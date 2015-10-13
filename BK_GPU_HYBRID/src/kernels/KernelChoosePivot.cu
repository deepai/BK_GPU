#include "../Device/NeighbourGraph.h"
#include "../Device/GPUStack.h"
#include "../Device/GPUCSR.h"
#include "kernels.cuh"

__device__
void swap(int &pos1, int &pos2) {
	pos1 = pos1 ^ pos2;
	pos2 = pos1 ^ pos2;
	pos1 = pos1 ^ pos2;
}

__device__
bool bsearch(int low, int high, int *array, int val) {
	int mid;
	while (low <= high) {
		mid = low + (high - low) / 2;
		if (array[mid] == val)
			return true;
		else if (val < array[mid])
			high = mid - 1;
		else
			low = mid + 1;
	}
	return false;
}

/**
 * This Kernel will choose the pivot element. The pivot element has the highest neighbours in P.
 *
 * @param Cliqes_per_block //Number of Cliques per block
 * @param Cliques_size //Size of each Clique in terms of multiple of 32
 * @param Neighbour_size // neighbour size(Max size of Clique)
 * @param g // Neighbour Graph
 * @param stack //The DFS stack used to simulate recursion
 * @param InputGraph //Contains the CSR format of Input Graph
 */__global__
void Kernel_ChoosePivot(int Cliqes_per_block, int Cliques_size,
		int Neighbour_size, BK_GPU::NeighbourGraph &graph,
		BK_GPU::GPU_Stack **stack, BK_GPU::GPU_CSR &InputGraph) {

	//shared memory
	__shared__ int currentVal[32], currentValCount[32], maxValCount[32];
	__shared__ int neighbours[1024];
	__shared__ int currentNodeAdjacentSize[32];
	__shared__ int adjacentListPosOffset[32];
	__shared__ int maxPos[32];

	//Thread Id of each thread
	int tid = threadIdx.x + blockIdx.x * blockDim.x;

	//works at warp levels. ThreadId within a Clique/
	int tid_perClique = tid % Cliques_size;

	/**CliqueID:
	 * The Clique selected by each thread is number of Cliques selected in previous
	 * blocks + ThreadId/Clique_size where Cliques_size is a multiple of 32(Warp Size)
	 */

	int localCliqueId = threadIdx.x / Cliques_size;
	int CliqueId = localCliqueId + Cliqes_per_block * blockIdx.x;

	if (tid_perClique == 0) {
		maxValCount[localCliqueId] = -1;
	}

	__syncthreads();

	/**PVertexoffset gives the value of the beginP present in the top of the stack
	 * currentVal is the actual value of the node. its obtained by graph.dataOffset[CliqueId] + beginP
	 */
	int PVertexStartOffset = stack[CliqueId]->topElement().beginP;
	int PVertexlength = stack[CliqueId]->topElement().currPSize;

	for (int idx = PVertexStartOffset;
			idx < (PVertexlength + PVertexStartOffset); idx++) {
		//First thread copies the value at position Idx from P into the shared memory currentVal
		if (tid_perClique == 0) {
			currentVal[localCliqueId] = graph.data[idx]; //Current Vertex contains the vertex P whose neighbour intersections we want to carry
			currentValCount[localCliqueId] = 0; //Count of neighbours in array P.
			currentNodeAdjacentSize[localCliqueId] =
					InputGraph.rowOffsets[currentVal[localCliqueId] + 1]
							- InputGraph.rowOffsets[currentVal[localCliqueId]]; //Adjacency size of currentVal
			adjacentListPosOffset[localCliqueId] =
					InputGraph.rowOffsets[currentVal[localCliqueId]];
		}

		__syncthreads();

		/**
		 *This loop brings in the adjacency list of the currentVal.If the size exceeds Clique_size, it brings in Clique_size neighbours
		 *at each iteration
		 */
		for (int neighbourIter = 0;
				neighbourIter
						< (int) (ceil(
								(double) currentNodeAdjacentSize[localCliqueId]
										/ Cliques_size)); neighbourIter++) {
			//Position of the adjacencyList of currentVal
			int fetchNeighbourPos = neighbourIter * Cliques_size
					+ tid_perClique;

			if (fetchNeighbourPos < currentNodeAdjacentSize[localCliqueId]) {
				neighbours[tid] =
						InputGraph.Columns[adjacentListPosOffset[localCliqueId]
								+ fetchNeighbourPos];

			} else
				neighbours[tid] = INT_MAX;

			__syncthreads();

			/**
			 * Loop Through Neighbour List of current Node.
			 */
			for (int IndexInP = 0;
					IndexInP
							< (int(ceil((double) PVertexlength / Cliques_size)));
					IndexInP++) {
				if ((IndexInP * Cliques_size + tid_perClique)
						< (PVertexlength)) {
					int PVertex = graph.data[PVertexStartOffset
							+ IndexInP * Cliques_size + tid_perClique];

					//Now check if PVertex is contained in the neighbourhood of currentVal
					bool isPVertexANeighbourOfCurrentVal = bsearch(
							localCliqueId * Cliques_size,
							(localCliqueId + 1) * Cliques_size - 1, neighbours,
							PVertex);

					//If Vertex Exists as a neighbour then increase the count
					if (isPVertexANeighbourOfCurrentVal)
						atomicAdd(&currentValCount[localCliqueId], 1);

				}
			}

		}

		if (tid_perClique == 0) {
			if (currentValCount[localCliqueId] > maxValCount[localCliqueId]) {
				maxValCount[localCliqueId] = currentValCount[localCliqueId];
				maxPos[localCliqueId] = idx;
			}
		}
	}

	if (tid_perClique == 0) {
//		printf("Node=%d, pos=%d count=%d\n",
//				graph.data[maxPos[localCliqueId]] + 1, maxPos[localCliqueId],
//				maxValCount[localCliqueId]); //Debugging purposes
		int beginR = tos(stack,CliqueId).beginR;
		/**
		 * Swap values of maxPos[localCliqueId] and beginR-1. i.e. copy the pivot value and append it to the beginning
		 * of R values.
		 */
		swap(graph.data[maxPos[localCliqueId]],graph.data[beginR-1]);

		stack[CliqueId]->push(tos(stack,CliqueId).beginX,tos(stack,CliqueId).currXSize,tos(stack,CliqueId).beginP,tos(stack,CliqueId).currPSize-1,tos(stack,CliqueId).beginR-1,tos(stack,CliqueId).currRSize+1,maxPos[localCliqueId],tos(stack,CliqueId).remainingNonNeighbour,tos(stack,CliqueId).direction);
	}
	}

extern "C" int GpuPivotSelect(BK_GPU::NeighbourGraph &graph,
		BK_GPU::GPU_Stack **stack, BK_GPU::GPU_CSR &InputGraph) {

	Kernel_ChoosePivot<<<1, 96>>>(3, 32, 5, graph, stack, InputGraph);

	DEV_SYNC;

	return 0;
}
