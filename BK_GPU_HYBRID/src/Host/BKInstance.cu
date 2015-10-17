/*
 * BKInstance.cpp
 *
 *  Created on: 06-Oct-2015
 *      Author: debarshi
 */

#include "BKInstance.h"
#include "../moderngpu/util/mgpucontext.h"
#include "../kernels/kernels.cuh"
#include "../moderngpu/mgpuhost.cuh"
#include "../cub/device/device_scan.cuh"

using namespace mgpu;

namespace BK_GPU {

BKInstance::BKInstance(Graph *host_graph, BK_GPU::GPU_CSR *gpuGraph,
		BK_GPU::NeighbourGraph *Ng, BK_GPU::GPU_Stack *stack,cudaStream_t &stream) {
	// TODO Auto-generated constructor stub
	this->Ng = Ng;
	this->gpuGraph = gpuGraph;
	this->stack = stack;
	this->topElement = stack->topElement();
	this->hostGraph = new NeighbourGraph();
	this->hostGraph = Ng;

	this->Stream= &stream;

	this->Context = mgpu::CreateCudaDeviceAttachStream(0,*(this->Stream));
	this->host_graph = host_graph;

	this->tracker = new BK_GPU::RecursionStack(topElement.currPSize);
}

int BKInstance::processPivot(BK_GPU::StackElement &element) {
	/**Step 1: Find the pivot element
	 */
	int currP = topElement.currPSize; //Size of Number of Elements in P
	int currX = topElement.currXSize;
	unsigned int *d_Sorted; //This is used to store the Unsorted elements initially

	void *d_temp_storage = NULL; //Auxillary array required for temporary Storage
	size_t d_temp_size = sizeof(unsigned) * currP * 2; //size of auxillary array is 2*N

	//Allocate Auxillary Array
	gpuErrchk(cudaMallocManaged(&d_temp_storage, sizeof(unsigned)* 2 * (currP + currX)));

	//Point to the unsorted input data
	unsigned int *d_unSorted = (unsigned *) &(Ng->data[topElement.beginP]);
	d_Sorted = d_unSorted;

	DEV_SYNC
	;
	//Store the Node Value for each value in the currPArray
	unsigned int *hptr = new unsigned[currP];

	//Kernel to copy the current P Values to the host in the hptr array.
	GpuCopyOffsetAddresses(Ng, stack, gpuGraph, hptr, currP,*(this->Stream));

	//This Array contains values 0 and 1 to store whether a value in the needle matches the haystack
	unsigned int* dptr;

	//size currP to allow prefixSums
	gpuErrchk(cudaMallocManaged(&dptr, sizeof(int) * 2 *(currP + currX)));

	DEV_SYNC
	;

	unsigned int *adata = d_Sorted;

	/** Max Index is used to store the index of value within P
	 *  Max Index lies between 0 and P-1.
	 */
	int max_index, numNeighbours = -1;

	int currNeighbour, non_neighbours;
	int acount = currP;

	/** For each value in the P array. Obtain the count of its neighbour amongst P.
	 *  The value with the maximum neighbour count is selected as the pivot. Other non-neighbours are also
	 *  selected after the pivot is completed.
	 *  This helps avoid unnecessary computations.
	 *
	 */
	for (int i = 0; i < currP; i++) {
		int adjacencySize = (host_graph->rowOffset[hptr[i] + 1] - host_graph->rowOffset[hptr[i]]);

		//std::cout << adjacencySize << ", " << host_graph->rowOffset[hptr[i] + 1] << " " << host_graph->rowOffset[hptr[i]]<< std::endl;

		unsigned int *bdata =
				&(gpuGraph->Columns[host_graph->rowOffset[hptr[i]]]);

		DEV_SYNC
		; //

		SortedSearch<MgpuBoundsLower, MgpuSearchTypeMatch, MgpuSearchTypeNone>(
				adata, acount, bdata, adjacencySize, dptr, dptr, *Context,
				&currNeighbour, &non_neighbours);


		if (currNeighbour > numNeighbours) {
			max_index = i;
			numNeighbours = currNeighbour;
		}
	}

	/**Swap the element(pivot) with the rightMost P element.
	 * New Size of P
	 */
	int endP = this->topElement.beginR - 1;

	GpuSwap(this->Ng,max_index+topElement.beginP, endP);

	d_unSorted = d_Sorted;
//
	gpuErrchk(cub::DeviceRadixSort::SortKeys(d_temp_storage, d_temp_size,
						d_unSorted, d_Sorted, currP-1,0,sizeof(uint)*8,*(this->Stream)));


	DEV_SYNC;

	int newBeginR = topElement.beginR - 1;
	int newRsize = topElement.currRSize + 1;


	//adjacency size of the neighbour array.
	int adjacencySize = host_graph->rowOffset[hptr[max_index] + 1]
			- host_graph->rowOffset[hptr[max_index]];

	//pointer to the beginning of the adjancy list for the maximum value
	unsigned *bdata =
			&(gpuGraph->Columns[host_graph->rowOffset[hptr[max_index]]]);

	//This calculates the number of remaining non-neighbours of pivot.
	SortedSearch<MgpuBoundsLower, MgpuSearchTypeMatch, MgpuSearchTypeNone>(
			adata, acount - 1, bdata, adjacencySize, dptr, dptr, *Context,
			&currNeighbour, &non_neighbours);

	int newPsize = currNeighbour;

	/**
	 * //Do a Scan on the current dptr array.
	 * //thrust::inclusive_scan(dptr, dptr + currP - 1, dptr);
	 */

	if(currP > 2)
	{
		size_t requiredmemSize;void *ptr=NULL;

		//Ist Invocation calculates the amount of memory required for the temporary array.
		gpuErrchk(cub::DeviceScan::InclusiveSum(ptr,requiredmemSize,dptr,dptr,currP - 1,*(this->Stream)));

		gpuErrchk(cudaMalloc(&ptr,requiredmemSize));

		//This step does the actual inclusiveSum
		gpuErrchk(cub::DeviceScan::InclusiveSum(ptr,requiredmemSize,dptr,dptr,currP - 1,*(this->Stream)));

		gpuErrchk(cudaFree(ptr));
	}

	DEV_SYNC;


	non_neighbours = currP - 1 - currNeighbour;

	//call Kernel Here to re-arrange P elements
	if((currNeighbour>0) && (currNeighbour < (currP-1)))
	{
		GpuArrayRearrangeP(this->Ng, this->stack, this->gpuGraph, dptr,
			topElement.beginP, topElement.beginP + currP - 2,non_neighbours,*(this->Stream));
	}

	//Repeat the steps for currX.
	//Intersection with X

	if (currX != 0) 
	{
		d_temp_size = 2 * currX * sizeof(int);

		//Pointer to the CurrX Values
		d_unSorted = (unsigned *) &(Ng->data[topElement.beginX]);
		d_Sorted = d_unSorted;

		//Output CurrX sorted into
		gpuErrchk(
				cub::DeviceRadixSort::SortKeys(d_temp_storage, d_temp_size,
						d_unSorted, d_Sorted, currX,0,sizeof(uint)*8,*(this->Stream)));

		adata = d_Sorted;
		int acount = topElement.currXSize;

		int NeighboursinX, nonNeighboursinX;

		SortedSearch<MgpuBoundsLower, MgpuSearchTypeMatch, MgpuSearchTypeNone>(
				adata, acount, bdata, adjacencySize, dptr, dptr, *Context,
				&NeighboursinX, &nonNeighboursinX);


		if(currX > 2)
		{
			/***
			 * * Do a Scan on the current dptr array. We can use the prefix sum to rearrange the neighbours and non-neighbours
			 */		//thrust::inclusive_scan(dptr, dptr + currX, dptr);
			size_t requiredmemSize = 0; void *ptr=NULL;

			gpuErrchk(cub::DeviceScan::InclusiveSum(ptr,requiredmemSize,dptr,dptr,currX,*(this->Stream)));

			gpuErrchk(cudaMalloc(&ptr,requiredmemSize));

			gpuErrchk(cub::DeviceScan::InclusiveSum(ptr,requiredmemSize,dptr,dptr,currX,*(this->Stream)));

			gpuErrchk(cudaFree(ptr));

			DEV_SYNC;
		}

		/***
		 * Scan Complete
		 */


		if((NeighboursinX > 0) && (NeighboursinX < currX ))
			GpuArrayRearrangeX(Ng,stack,gpuGraph,dptr,topElement.beginX,topElement.beginP-1,NeighboursinX,*(this->Stream));

		topElement.currXSize = NeighboursinX;
	}
	int trackerSize = tracker->size() ;

	stack->push(topElement.beginX, topElement.currXSize, topElement.beginP,
			newPsize, newBeginR, newRsize, max_index,trackerSize, non_neighbours, true);

	topElement.beginR = newBeginR;
	topElement.currPSize = newPsize;
	topElement.currRSize = newRsize;
	topElement.direction = true;
	topElement.pivot = hptr[max_index];
	topElement.remainingNonNeighbour = non_neighbours;

	//debug(max_index, hptr[max_index], numNeighbours);

	//debug(dptr[0],dptr[1],dptr[2],dptr[3]);

	/**Free the pointers **/
	gpuErrchk(cudaFree(d_temp_storage));
	gpuErrchk(cudaFree(dptr));


	delete[] hptr;

	return (non_neighbours + 1);

}

void BKInstance::printClique(int CliqueSize,int beginClique)
{
#ifdef PRINTCLIQUES
	for(int i=0;i<CliqueSize;i++)
		printf("%d ",Ng->data[beginClique+i]+1);

	printf("\n");
#endif
}

void BKInstance::RunCliqueFinder(int CliqueId) {

//	//topElement.printconfig();
//	if(topElement.currRSize%50==0)
//		topElement.printconfig();

	if ((topElement.currPSize == topElement.currXSize)
			&& (topElement.currXSize == 0)) {		//Obtained a Clique
		printf("%d) Clique of size %d, found!\n",CliqueId,topElement.currRSize);
		printClique(topElement.currRSize,topElement.beginR);
		return;
	} else if (topElement.currPSize == 0)
	{
		//printf("%d) Already contains a clique\n",CliqueId);
		return; //didn't obtain a Clique
	}
	else {
		int non_neighbours = processPivot(topElement);
		RunCliqueFinder(CliqueId);

		stack->pop();

		tracker->push(topElement.pivot);

		tracker->pop();
	}
}

} /* namespace BK_GPU */
