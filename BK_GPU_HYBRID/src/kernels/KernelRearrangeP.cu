#include "kernels.cuh"

/**This kernel is used to swap the values and bring the values having 1 in the darray towards the
 * start_offset and the values having 0 towards the end_offset
 * This Kernel acts as a gatherer.
 *
 * @param darray //array containing the prefixSum
 * @param d_temp //array to store the gathered values
 * @param start_offset //startOffset in the graph->data
 * @param end_offset //endoffset
 * @param graph //input graph
 * @param stack //input stack
 */
__global__
void kernelRearrangeGatherP(uint *darray,uint *d_temp,int start_offset,int end_offset,BK_GPU::NeighbourGraph *graph,BK_GPU::GPU_Stack* stack)
{
	int tid=threadIdx.x + blockDim.x*blockIdx.x;

	//Exceeds limit hence return
	if(tid+start_offset > end_offset)
		return;

	//get the current prefixsum value
	uint currVal=darray[tid+start_offset];

	//get the next prefixsum value
	uint nextVal=darray[tid+start_offset+1];

	int destination; //store destination here

	/**If( nextVal - currVal ) == 1, indicates that tid+start_offset is a neighbour. Hence
	 * ,its destination will be start_offset + currVal - 1(currval indicates number of 1s obtained previously)
	 *
	 * else , it indicates the tid+start_offset is not a neighbour. hence currVal - tid+start_offset
	 * indicates number of 0s preceding it.
	 */

	if(nextVal - currVal == 1)
		destination = start_offset + currVal - 1;
	else
		destination = end_offset - ( currVal - (tid + start_offset ));

	//Copy the current Element before swapping
	int currElement=graph->data[tid+start_offset];

	d_temp[destination - start_offset] = currElement;

}

/**
 * In this kernel elements in the d_temp array are copied back into the graph->data
 * starting from start_offset.
 *
 * @param d_temp //temporary storage
 * @param start_offset //start_offset in graph->data
 * @param end_offset //end_offset in graph->data
 * @param graph //graph
 */
__global__
void KernelRearrangeScatterP(uint *d_temp,int start_offset,int end_offset,BK_GPU::NeighbourGraph *graph)
{
	int tid = threadIdx.x + blockDim.x*blockIdx.x;

	if(tid+start_offset > end_offset)
		return;

	graph->data[tid + start_offset] = d_temp[tid];
}


extern "C"
void GpuArrayRearrangeP(BK_GPU::NeighbourGraph *graph,
		BK_GPU::GPU_Stack* stack,BK_GPU::GPU_CSR *InputGraph,unsigned int *darray,int start_offset,int end_offset)
{
	int numElements = end_offset - start_offset + 1;

	uint* d_temp;

	gpuErrchk(cudaMalloc(&d_temp,sizeof(uint)*numElements));

	kernelRearrangeGatherP<<<ceil((double)numElements/128),128>>>(darray,d_temp,start_offset,end_offset,graph,stack);

	DEV_SYNC;

	KernelRearrangeScatterP<<<ceil((double)numElements/128),128>>>(d_temp,start_offset,end_offset,graph);

	DEV_SYNC;

	gpuErrchk(cudaFree(d_temp));
}
