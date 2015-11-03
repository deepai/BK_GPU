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
		BK_GPU::NeighbourGraph *Ng, BK_GPU::GPU_Stack *stack,cudaStream_t &stream,mgpu::ContextPtr context) {

	this->maxCliqueSizeObtained = 1;
	// TODO Auto-generated constructor stub
	this->Ng = Ng; 								//Neighbor graph allocated in Device memory
	this->gpuGraph = gpuGraph; 					//CSR Graph allocated in Device memory
	this->stack = stack; 						//Stack for the current Clique , allocated in the device memory

	stack->topElement(&topElement); 	//StackElement resident in the CPU memory

	this->Stream= &stream;						//cudastream



	this->Context = context;


	this->host_graph = host_graph;				//Graph allocated in the host memory

	this->tracker = new BK_GPU::RecursionStack(topElement.currPSize,stream); //This device resident is used to keep a track of non_neighbours for a given evalution of BK

}

/**
 * Requires: Updated TopElement,Updated Stack,Sorted P and X segments.
 *
 * This element finds the pivot location.Once the pivot is located it is moved to the end of the current P segment.
 * Thus R segment is increased by 1 and P segment is reduced by 1. The beginning of R is shifted by 1 towards P.
 *
 * At the end of this method,
 * The topElement is updated,stack is updated with the new configuration.
 * In this new configuration,the pivot element is moved to the R segment and the P segment size is changed to
 * only contain the neighbors of Pivot in the P segment. i.e.
 *
 * X segment is updated to contain only the neighbors of the pivot,the current size of X is updated to reflect that.
 * Both P and X segments are sorted post this method.
 *
 * Tracker is updated but contains the previous value itself.
 *
 * @param element topElement is passed to this method.
 * @return
 */
int BKInstance::processPivot(BK_GPU::StackElement &element) {
	/**Step 1: Find the pivot element
	 *
	 */
	//This location is used to point to by a NULL reference
	int NullValue;

	int currP = topElement.currPSize; //Number of Elements in P
	int currX = topElement.currXSize; //Number of Elements in X
	unsigned int *d_Sorted; 		  //Pointer to Neighbour graph of a vertex

	//Point to the unsorted input data (i.e. P Array in the device graph)
	unsigned int *d_unSorted = (unsigned *)(Ng->data) + topElement.beginP;
	d_Sorted = d_unSorted;

	//cudaStreamSynchronize(*(this->Stream));
	//Host memory used to store the current P array in the host.To make pivoting faster.
	unsigned int *hptr = new unsigned[currP];

	//Kernel to copy the current P array to the host in the hptr array.
	GpuCopyOffsetAddresses(Ng, topElement.beginP, gpuGraph, hptr, currP,*(this->Stream));

	//Various uses of auxillary pointer
	unsigned int* auxillaryStorage;

	//set a maximum size of (curr*P)
	CudaError(cudaMalloc(&auxillaryStorage, sizeof(int) * (currP)));

	//DEV_SYNC;

	//Pointer to d_Sorted
	unsigned int *adata = d_Sorted;

	/** Max Index is used to store the index of value within P
	 *  Max Index lies between 0 and P-1.
	 */
	int max_index, numNeighbours = -1;

	int currNeighbour, non_neighbours;
	int acount = currP;

	/** For each value in the P segment. Obtain the count of its neighbors amongst P .
	 *  The value with the maximum neighbor count is selected as the pivot. Other non-neighbors are also
	 *  selected after the pivot is completed.
	 *  This helps avoid unnecessary computations.
	 *
	 */
	for (int i = 0; i < currP; i++) {
		int adjacencySize = (host_graph->rowOffset[hptr[i] + 1] - host_graph->rowOffset[hptr[i]]);

		//std::cout << adjacencySize << ", " << host_graph->rowOffset[hptr[i] + 1] << " " << host_graph->rowOffset[hptr[i]]<< std::endl;

		unsigned int *bdata =gpuGraph->Columns + host_graph->rowOffset[hptr[i]];

		//DEV_SYNC
		//; //

		SortedSearch<MgpuBoundsLower, MgpuSearchTypeMatch, MgpuSearchTypeNone>(
				adata, acount, bdata, adjacencySize, auxillaryStorage, auxillaryStorage, *Context,
				&currNeighbour, &non_neighbours);


		if (currNeighbour > numNeighbours) {
			max_index = i;
			numNeighbours = currNeighbour;
		}
	}

	/**Swap the element(pivot) with the rightMost P element.
	 * New Size of P
	 */
	int newBeginR = this->topElement.beginR - 1;

	//Swap the current element with the beginR - 1 position
	GpuSwap(this->Ng,max_index+topElement.beginP, newBeginR,*(this->Stream));

	//if beginR - 1 is not the end of the current P segment,then swap the current value with the end of the P segment.
	if(newBeginR != (topElement.beginP + topElement.currPSize - 1) )
	{
		GpuSwap(this->Ng,max_index + topElement.beginP,topElement.beginP + topElement.currPSize - 1,*(this->Stream));
	}

	d_unSorted = d_Sorted;
//
	//Sort the P segment is currPsize > 2
	if(currP > 2 )
	{
		//pointer and size variable to allocate temporary array.
		void *d_temp_storage=NULL;size_t d_temp_size=0;

		//One call to prefill the d_temp_size with required memory size.
		CudaError(cub::DeviceRadixSort::SortKeys(d_temp_storage, d_temp_size,
						d_unSorted, d_Sorted, currP-1,0,sizeof(uint)*8,*(this->Stream)));

		//Allocate appropiate memory for d_temp_storage required for radixSort.
		CudaError(cudaMalloc(&d_temp_storage,d_temp_size));

		if(d_temp_storage==NULL)
			d_temp_storage=&NullValue;

		//Perform the actual Radix Sort.
		CudaError(cub::DeviceRadixSort::SortKeys(d_temp_storage, d_temp_size,
								d_unSorted, d_Sorted, currP-1,0,sizeof(uint)*8,*(this->Stream)));

		//Synchronize the stream
		CudaError(cudaStreamSynchronize(*(this->Stream)));

		//Free memory.
		if(d_temp_storage!=&NullValue)
			CudaError(cudaFree(d_temp_storage));

	}

	//Rsize is incremented by 1.
	int newRsize = topElement.currRSize + 1;

	//adjacency size of the neighbor array.
	int adjacencySize = host_graph->rowOffset[hptr[max_index] + 1]
			- host_graph->rowOffset[hptr[max_index]];

	//pointer to the beginning of the adjacency list for the maximum value.
	unsigned *bdata = gpuGraph->Columns + host_graph->rowOffset[hptr[max_index]];

	//This calculates the number of remaining non-neighbors of pivot.
	SortedSearch<MgpuBoundsLower, MgpuSearchTypeMatch, MgpuSearchTypeNone>(
			adata, acount - 1, bdata, adjacencySize, auxillaryStorage, auxillaryStorage, *Context,
			&currNeighbour, &non_neighbours);

	int newPsize = currNeighbour;

	//Only if the P segment size is greater than 2, do an Inclusive Sum.
	if(currP > 2)
	{
		void *d_temp_storage=NULL;size_t d_temp_size=0;

		//Ist Invocation calculates the amount of memory required for the temporary array.
		CudaError(cub::DeviceScan::InclusiveSum(d_temp_storage,d_temp_size,auxillaryStorage,auxillaryStorage,currP - 1,*(this->Stream)));

		CudaError(cudaMalloc(&d_temp_storage,d_temp_size));

		if(d_temp_storage==NULL)
			d_temp_storage=&NullValue;

		//This step does the actual inclusiveSum
		CudaError(cub::DeviceScan::InclusiveSum(d_temp_storage,d_temp_size,auxillaryStorage,auxillaryStorage,currP - 1,*(this->Stream)));

		//Synchronize the stream
		CudaError(cudaStreamSynchronize(*(this->Stream)));

		//Free the allocated memory.
		if(d_temp_storage!=&NullValue)
			CudaError(cudaFree(d_temp_storage));
	}

	//Obtain the count of Non_Neighbours of the pivot.
	non_neighbours = currP - 1 - currNeighbour;

	//call Kernel Here to re-arrange P elements
	if((currNeighbour>0) && (currNeighbour < (currP-1)))
	{
		GpuArrayRearrangeP(this->Ng, this->stack, this->gpuGraph, auxillaryStorage,
			topElement.beginP, topElement.beginP + currP - 2,non_neighbours,*(this->Stream));
	}

	CudaError(cudaFree(auxillaryStorage));

	//Repeat the steps for currX.
	//Intersection with X

	if (currX != 0) 
	{
		//allocate memory for auxiliary space for X arrays
		CudaError(cudaMalloc(&auxillaryStorage, sizeof(int) * (currX)));

		//Pointer to the CurrX Values
		d_unSorted = (unsigned *)(Ng->data) + topElement.beginX;
		d_Sorted = d_unSorted;

		adata = d_Sorted;
		int acount = topElement.currXSize;

		int NeighboursinX, nonNeighboursinX;

		SortedSearch<MgpuBoundsLower, MgpuSearchTypeMatch, MgpuSearchTypeNone>(
				adata, acount, bdata, adjacencySize, auxillaryStorage, auxillaryStorage, *Context,
				&NeighboursinX, &nonNeighboursinX);

		//Scan only if currX size is greater than 1.
		if(currX > 1)
		{
			/***
			 * * Do a Scan on the current dptr array. We can use the prefix sum to rearrange the neighbours and non-neighbours
			 */		//thrust::inclusive_scan(dptr, dptr + currX, dptr);
			void *d_temp_storage=NULL;size_t d_temp_size=0;

			CudaError(cub::DeviceScan::InclusiveSum(d_temp_storage,d_temp_size,auxillaryStorage,auxillaryStorage,currX,*(this->Stream)));

			CudaError(cudaMalloc(&d_temp_storage,d_temp_size));

			if(d_temp_storage==NULL)
				d_temp_storage=&NullValue;

			CudaError(cub::DeviceScan::InclusiveSum(d_temp_storage,d_temp_size,auxillaryStorage,auxillaryStorage,currX,*(this->Stream)));

			CudaError(cudaStreamSynchronize(*(this->Stream)));

			if(d_temp_storage!=&NullValue)
				CudaError(cudaFree(d_temp_storage));

			//DEV_SYNC;
		}

		/***
		 * Scan Complete
		 */


		if((NeighboursinX > 0) && (NeighboursinX < currX ))
			GpuArrayRearrangeX(Ng,stack,gpuGraph,auxillaryStorage,topElement.beginX,topElement.beginX + topElement.currXSize - 1,NeighboursinX,*(this->Stream));

		topElement.currXSize = NeighboursinX;

		CudaError(cudaFree(auxillaryStorage));
	}
	int trackerSize = tracker->size() ;

	//CudaError(cudaStreamSynchronize(*(this->Stream)));
	topElement.beginR = newBeginR;
	topElement.currPSize = newPsize;
	topElement.currRSize = newRsize;
	topElement.direction = true;
	topElement.pivot = hptr[max_index];
	topElement.trackerSize = trackerSize;

	stack->push(&topElement);

	/**Free the pointers **/

	delete[] hptr;

	return (non_neighbours);

}

/**
 * Requirements:None
 *
 * This method is used to copy back the previously used P value to the X array.i.e. The value at position beginR
 * is moved to value at position beginX + currX -1
 *
 * The X segment is increased by 1,the R segment is decreased by 1.
 * beginP is shifted by 1 to accommodate the new X.
 * begin R is hence also shifted by 1.
 *
 * Both X and P Arrays are Sorted post execution of this method.
 * The previous stack configuration is popped and a new configuration is pushed into it.
 * The new stack configuration has the updated X and R values
 */
void BKInstance::moveToX()
{
	int NullValue;
	//Update the topElement.
	stack->topElement(&topElement);

	stack->secondElement(&secondElement);

	CudaError(cudaStreamSynchronize(*(this->Stream)));

	//Old_posElement is the last position of the P array.
	int old_posElement = topElement.beginR;

	//new position for the X value would be at topElement.beginX + topElement.currXSize
	int new_posElement = topElement.beginX + topElement.currXSize;



	//swap the positions
	GpuSwap(this->Ng,old_posElement,new_posElement,*(this->Stream));

	//If beginP is not swapped, swap it with the old_position to move back the X element into its previous position
	if(new_posElement!=topElement.beginP)
		GpuSwap(this->Ng,topElement.beginP,old_posElement,*(this->Stream));

	topElement.currXSize++;
	topElement.currRSize--;
	topElement.beginP++;
	topElement.beginR++;

	//Sort if currPSize > 1
	if(topElement.currPSize > 1)
	{
		int currP=topElement.currPSize;

		void *d_temp_storage=NULL;size_t d_temp_size=0;

		int *d_in=Ng->data + topElement.beginP;
		int *d_out=d_in;

		CudaError(cub::DeviceRadixSort::SortKeys(d_temp_storage,d_temp_size,d_in,d_out,currP,0,sizeof(uint)*8,*(this->Stream)));

		CudaError(cudaMalloc(&d_temp_storage,d_temp_size));

		if(d_temp_storage==NULL)
			d_temp_storage=&NullValue;

		//Sort the array.
		CudaError(cub::DeviceRadixSort::SortKeys(d_temp_storage,d_temp_size,d_in,d_out,currP,0,sizeof(uint)*8,*(this->Stream)));

		CudaError(cudaStreamSynchronize(*(this->Stream)));

		if(d_temp_storage!=&NullValue)
			CudaError(cudaFree(d_temp_storage));
	}

	//Sort if currXSize > 1
	if(topElement.currXSize > 1)
	{
		int currX=topElement.currXSize;

		void *d_temp_storage=NULL;size_t d_temp_size=0;

		int *d_in=Ng->data + topElement.beginX;
		int *d_out=d_in;

		CudaError(cub::DeviceRadixSort::SortKeys(d_temp_storage,d_temp_size,d_in,d_out,currX,0,sizeof(int)*8,*(this->Stream)));

		CudaError(cudaMalloc(&d_temp_storage,d_temp_size));

		if(d_temp_storage==NULL)
			d_temp_storage=&NullValue;

		//Sort the array.
		CudaError(cub::DeviceRadixSort::SortKeys(d_temp_storage,d_temp_size,d_in,d_out,currX,0,sizeof(int)*8,*(this->Stream)));

		CudaError(cudaStreamSynchronize(*(this->Stream)));

		if(d_temp_storage!=&NullValue)
			CudaError(cudaFree(d_temp_storage));
	}
	//Pop the values of the previous stack
	stack->pop();

	//Push the new configuration values into the stack.
	stack->push(&topElement);

	//Push the newly moved element into the tracker.
	tracker->push(topElement.pivot);

	//CudaError(cudaStreamSynchronize(*(this->Stream)));

}

void BKInstance::printClique(int CliqueSize,int beginClique)
{
#ifdef PRINTCLIQUES
	for(int i=0;i<CliqueSize;i++)
		printf("%d ",Ng->data[beginClique+i]+1);

	printf("\n");
#endif
}

/**
 * Requirements:None
 *
 * This method is used to copy back the previously tracked values in the current recursion step
 * from X to P array.
 *
 * Calculate number of Values tracked and needed to be moved into P from X.
 * Do a sorted Search for the values.
 * Do a prefix sum on the result of the sorted search on the entire X#### segment.
 * Do a GpuRearrange to rearrange the Tracked values towards the beginning of the P.
 * The rest of the values maintain their previous relative positions.
 *
 * Both X and P Arrays are Sorted post execution of this method.
 * Tracker size is removed by number of TrackedElements
 * Stack is then popped once to go back to the previous state before the current recursive invocation.
 *
 *
 */
void BKInstance::moveFromXtoP()
{
	int NullValue;
	//obtain the top of the stack first.
	stack->topElement(&topElement);

	//obtain the next value of the stack.
	stack->secondElement(&secondElement);

	//Current Number of Elements in the tracker.
	int currTrackerSize = tracker->size();

	CudaError(cudaStreamSynchronize(*(this->Stream)));

	//Number of elements Tracked which have been moved from P to X in the current recursive call.
	int NumValuesToMoveFromXToP = currTrackerSize - secondElement.trackerSize;

	//Pointer to the tracker elements to sort them.
	int *d_in=&(tracker->elements[currTrackerSize-NumValuesToMoveFromXToP]),*d_out=d_in;

	CudaError(cudaStreamSynchronize(*(this->Stream)));

	//Sorting is done only if NumValues > 1
	if(NumValuesToMoveFromXToP > 1)
	{
		//void *ptr;
		void *d_temp_storage=NULL; size_t d_temp_size=0;

		CudaError(cub::DeviceRadixSort::SortKeys(d_temp_storage,d_temp_size,d_in,d_out,NumValuesToMoveFromXToP,0,sizeof(int)*8,*(this->Stream)));

		CudaError(cudaMalloc(&d_temp_storage,d_temp_size));

		if(d_temp_storage==NULL)
			d_temp_storage=&NullValue;

		//Sort the Xvalues
		CudaError(cub::DeviceRadixSort::SortKeys(d_temp_storage,d_temp_size,d_in,d_out,NumValuesToMoveFromXToP,0,sizeof(int)*8,*(this->Stream)));

		CudaError(cudaStreamSynchronize(*(this->Stream)));

		if(d_temp_storage!=&NullValue)
			CudaError(cudaFree(d_temp_storage));

	}

	/**
	 * We need to now search for the tracked elements towards the left of P segment.
	 * Segment indicates X####|P###|R where # represents unused values
	 * Thus left of Segment is X####.
	 *
	 */
	int* d_flags;
	size_t dflagSize = sizeof(int)*2*(topElement.beginP - topElement.beginX + 1);

	int *adata=d_in;
	int acount=NumValuesToMoveFromXToP;

	int *bdata=Ng->data + topElement.beginX;
	int bcount=topElement.beginP - topElement.beginX; //The size of the bdata array is X####. We need to search it there

	int NeighboursinX,nonNeighboursinX;

	//Allocate memory for the flags
	CudaError(cudaMalloc(&d_flags,dflagSize));

	//Initialize the memory by 0
	CudaError(cudaMemset(d_flags,0,sizeof(int)*bcount));

	//Do a Sorted Search to check which values in bdata matches with values in  adata.
	SortedSearch<MgpuBoundsLower, MgpuSearchTypeMatch, MgpuSearchTypeNone>(
					adata, acount, bdata, topElement.currXSize, d_flags, d_flags, *Context,
					&NeighboursinX, &nonNeighboursinX);

	//if bcount > 1 , do an inclusive sum.
	if(bcount > 1)
	{
		void *d_temp_storage=NULL;size_t d_temp_size=0;

		//Inclusive Sum
		CudaError(cub::DeviceScan::InclusiveSum(d_temp_storage,d_temp_size,d_flags,d_flags,bcount,*(this->Stream)));

		CudaError(cudaMalloc(&d_temp_storage,d_temp_size));

		if(d_temp_storage==NULL)
			d_temp_storage=&NullValue;

		CudaError(cub::DeviceScan::InclusiveSum(d_temp_storage,d_temp_size,d_flags,d_flags,bcount,*(this->Stream)));

		CudaError(cudaStreamSynchronize(*(this->Stream)));

		if(d_temp_storage!=&NullValue)
			CudaError(cudaFree(d_temp_storage));

	}

	//CudaError(cudaStreamSynchronize(*(this->Stream)));

	//This kernel is used to rearrange back the X values towards P.
	GpuArrayRearrangeXtoP(Ng,d_flags,topElement.beginX,topElement.beginP-1,NeighboursinX,*(this->Stream));

	d_in=Ng->data + topElement.beginP - NumValuesToMoveFromXToP;

	//Sort the NumValuesToMoveFromXToP + CurrPSize elements.
	if(topElement.currPSize + NumValuesToMoveFromXToP > 1)
		CudaError(cub::DeviceRadixSort::SortKeys(d_flags,dflagSize,d_in,d_in,topElement.currPSize + NumValuesToMoveFromXToP,0,sizeof(int)*8,*(this->Stream)));

	CudaError(cudaFree(d_flags));

	//remove the nodes from the tracker
	tracker->pop(NumValuesToMoveFromXToP);

	//pop the current stack value.
	stack->pop();

}

/**
 * RunCliqueFinder is the main recursive call that emulates the BK_Tomita algorithm in the Heteregenous setting.
 * This algorithm works independently on a single cuda stream.
 *
 * Further plans to use CPU when currPSize becomes smaller than 128.
 *
 * Initial requirements of this algorithm are
 * 1)TopElement
 * 2)Neighbor graph in GPU
 * 3)CSR graph in GPU
 * 4)Stack Element in GPU
 *
 *
 * @param CliqueId
 */
void BKInstance::RunCliqueFinder(int CliqueId) {

//	//topElement.printconfig();
//	if(CliqueId==4)
//		topElement.printconfig();

	if ((topElement.currPSize == topElement.currXSize)
			&& (topElement.currXSize == 0)) {		//Obtained a Clique

		if(topElement.currRSize > maxCliqueSizeObtained)
			maxCliqueSizeObtained = topElement.currRSize;

		printf("%d) Clique of size %d, found!\n",CliqueId,topElement.currRSize);
		printClique(topElement.currRSize,topElement.beginR);
		return;
	} else if (topElement.currPSize == 0)
	{
		//printf("%d) Already contains a clique\n",CliqueId);
		return; //didn't obtain a Clique
	}
	else {

		//obtain the pivot element and adjust the vertices
		int non_neighbours = processPivot(topElement);
		int pivot = topElement.pivot;

		//On Expansion the current configuration would only result in a smaller CliqueSize.
		if(topElement.currRSize + topElement.currPSize > maxCliqueSizeObtained)
			RunCliqueFinder(CliqueId);

		//Move the pivot element to Reject List.
		moveToX();

		//While there are non_neighbours, continue invoking the recursive function.
		while(non_neighbours)
		{
			//Obtains the nextNonPivot Element
			nextNonPivot();

			//On Expansion the current configuration would only result in a smaller CliqueSize.
			if(topElement.currRSize + topElement.currPSize > maxCliqueSizeObtained)
				RunCliqueFinder(CliqueId);

			//Move elements back to X.
			moveToX();
			non_neighbours--;
		}

		//Move All elements back from X to P.
		moveFromXtoP();

		//Bring Back all values used in X into currP array.
	}
}

} /* namespace BK_GPU */
