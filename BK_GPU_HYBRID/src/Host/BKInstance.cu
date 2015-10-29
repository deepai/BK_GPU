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
	this->Ng = Ng; 								//Neighbor graph allocated in Device memory
	this->gpuGraph = gpuGraph; 					//CSR Graph allocated in Device memory
	this->stack = stack; 						//Stack for the current Clique , allocated in the device memory
	this->topElement = stack->topElement(); 	//StackElement resident in the CPU memory

	this->Stream= &stream;						//cudastream

	this->Context = mgpu::CreateCudaDeviceAttachStream(0,*(this->Stream));
	this->host_graph = host_graph;				//Graph allocated in the host memory

	this->tracker = new BK_GPU::RecursionStack(topElement.currPSize); //This device resident is used to keep a track of non_neighbours for a given evalution of BK
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
 * @param element
 * @return
 */
int BKInstance::processPivot(BK_GPU::StackElement &element) {
	/**Step 1: Find the pivot element
	 */
	int currP = topElement.currPSize; //Number of Elements in P
	int currX = topElement.currXSize; //Number of Elements in X
	unsigned int *d_Sorted; 		  //Pointer to Neighbour graph of a vertex

	void *d_temp_storage = NULL; 	  //Auxillary array required for temporary Storage
	size_t d_temp_size = sizeof(unsigned) * currP * 2; //size of auxillary array is 2*(X+P)

	//Allocate Auxillary Array
	gpuErrchk(cudaMallocManaged(&d_temp_storage, sizeof(unsigned)* 2 * (currP + currX)));

	//Point to the unsorted input data (i.e. P Array in the device graph)
	unsigned int *d_unSorted = (unsigned *) &(Ng->data[topElement.beginP]);
	d_Sorted = d_unSorted;

	DEV_SYNC;
	//Host memory used to store the current P array in the host.To make pivoting faster.
	unsigned int *hptr = new unsigned[currP];

	//Kernel to copy the current P array to the host in the hptr array.
	GpuCopyOffsetAddresses(Ng, stack, gpuGraph, hptr, currP,*(this->Stream));

	//Various uses of dptr pointer
	unsigned int* dptr;

	//set a maximum size of 2*(currP + currX)
	gpuErrchk(cudaMallocManaged(&dptr, sizeof(int) * 2 *(currP + currX)));

	DEV_SYNC;

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
	if(currP > 2 )
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
		if(currX > 1)
			gpuErrchk(
				cub::DeviceRadixSort::SortKeys(d_temp_storage, d_temp_size,
						d_unSorted, d_Sorted, currX,0,sizeof(uint)*8,*(this->Stream)));

		adata = d_Sorted;
		int acount = topElement.currXSize;

		int NeighboursinX, nonNeighboursinX;

		SortedSearch<MgpuBoundsLower, MgpuSearchTypeMatch, MgpuSearchTypeNone>(
				adata, acount, bdata, adjacencySize, dptr, dptr, *Context,
				&NeighboursinX, &nonNeighboursinX);

		//Scan only if currX size is greater than 1.
		if(currX > 1)
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
			newPsize, newBeginR, newRsize, hptr[max_index],trackerSize, non_neighbours, true);

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

	return (non_neighbours);

}

/**
 * Requirements:None
 *
 * This method is used to copy back the previously used P value to the X array.i.e. The value at position beginR
 * is moved to value at position beginX + currX -1
 *
 * The X segment is increased by 1,the R segment is decreased by 1.
 * beginP is shifted by 1 to accomodate the new X.
 * begin R is hence also shifted by 1.
 *
 * Both X and P Arrays are Sorted post execution of this method.
 * The previous stack configuration is popped and a new configuration is pushed into it.
 * The new stack configuration has the updated X and R values
 */
void BKInstance::moveToX()
{
	//Update the topElement.
	topElement=stack->topElement();

	secondElement=stack->secondElement();

	//Old_posElement is the last position of the P array.
	int old_posElement = topElement.beginR;

	//new position for the X value would be at topElement.beginX + topElement.currXSize
	int new_posElement = topElement.beginX + topElement.currXSize;



	//swap the positions
	GpuSwap(this->Ng,old_posElement,new_posElement);

	//If beginP is not swapped, swap it with the old_position to move back the X element into its previous position
	if(new_posElement!=topElement.beginP)
		GpuSwap(this->Ng,topElement.beginP,old_posElement);

	topElement.currXSize++;
	topElement.currRSize--;
	topElement.beginP++;
	topElement.beginR++;

	//Sort if currPSize > 1
	if(topElement.currPSize > 1)
	{
		int currP=topElement.currPSize;

		void *aux_ptr;
		size_t aux_size=sizeof(uint)*2*currP;

		gpuErrchk(cudaMalloc(&aux_ptr,aux_size));

		int *d_in=&(Ng->data[topElement.beginP]);
		int *d_out=d_in;

		//Sort the array.
		gpuErrchk(cub::DeviceRadixSort::SortKeys(aux_ptr,aux_size,d_in,d_out,currP,0,sizeof(uint)*8,*(this->Stream)));

		gpuErrchk(cudaFree(aux_ptr));
	}

	//Sort if currXSize > 1
	if(topElement.currXSize > 1)
	{
		int currX=topElement.currXSize;

		void *aux_ptr;
		size_t aux_size=sizeof(uint)*2*currX;

		gpuErrchk(cudaMalloc(&aux_ptr,aux_size));

		DEV_SYNC;

		int *d_in=&(Ng->data[topElement.beginX]);
		int *d_out=d_in;

		//Sort the array.
		gpuErrchk(cub::DeviceRadixSort::SortKeys(aux_ptr,aux_size,d_in,d_out,currX,0,sizeof(uint)*8,*(this->Stream)));

		gpuErrchk(cudaFree(aux_ptr));
	}

	DEV_SYNC;

	//Pop the values of the previous stack
	stack->pop();

	//Push the new configuration values into the stack.
	stack->push(topElement.beginX,topElement.currXSize,topElement.beginP,topElement.currPSize,topElement.beginR,topElement.currRSize,topElement.pivot,topElement.trackerSize+1,topElement.remainingNonNeighbour,topElement.direction);

	//Push the newly moved element into the tracker.
	tracker->push(topElement.pivot);

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
	//obtain the top of the stack first.
	topElement=stack->topElement();

	//obtain the next value of the stack.
	secondElement=stack->secondElement();

	//Current Number of Elements in the tracker.
	int currTrackerSize = tracker->size();

	//Number of elements Tracked which have been moved from P to X in the current recursive call.
	int NumValuesToMoveFromXToP = currTrackerSize - secondElement.trackerSize;

	//Pointer to the tracker elements to sort them.
	int *d_in=&(tracker->elements[currTrackerSize-NumValuesToMoveFromXToP]),*d_out=d_in;

	//Sorting is done only if NumValues > 1
	if(NumValuesToMoveFromXToP > 1)
	{
		void *ptr;
		size_t reserved_space=sizeof(int)*currTrackerSize*2;

		gpuErrchk(cudaMalloc(&ptr,reserved_space));

		//Sort the Xvalues
		gpuErrchk(cub::DeviceRadixSort::SortKeys(ptr,reserved_space,d_in,d_out,NumValuesToMoveFromXToP,0,sizeof(int)*8,*(this->Stream)));

		gpuErrchk(cudaFree(ptr));

		DEV_SYNC;

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

	int *bdata=&(Ng->data[topElement.beginX]);
	int bcount=topElement.beginP - topElement.beginX; //The size of the bdata array is X####. We need to search it there

	int NeighboursinX,nonNeighboursinX;

	//Allocate memory for the flags
	gpuErrchk(cudaMalloc(&d_flags,dflagSize));

	//Initialize the memory by 0
	gpuErrchk(cudaMemset(d_flags,0,sizeof(int)*bcount));

	//Do a Sorted Search to check which values in bdata matches with values in  adata.
	SortedSearch<MgpuBoundsLower, MgpuSearchTypeMatch, MgpuSearchTypeNone>(
					adata, acount, bdata, topElement.currXSize, d_flags, d_flags, *Context,
					&NeighboursinX, &nonNeighboursinX);

	//if bcount > 1 , do an inclusive sum.
	if(bcount > 1)
	{
		void *ptr=NULL;
		size_t requiredSpace;

		//Inclusive Sum
		gpuErrchk(cub::DeviceScan::InclusiveSum(ptr,requiredSpace,d_flags,d_flags,bcount,*(this->Stream)));

		gpuErrchk(cudaMalloc(&ptr,requiredSpace));

		gpuErrchk(cub::DeviceScan::InclusiveSum(ptr,requiredSpace,d_flags,d_flags,bcount,*(this->Stream)));

		gpuErrchk(cudaFree(ptr));

	}

	DEV_SYNC;

	//This kernel is used to rearrange back the X values towards P.
	GpuArrayRearrangeXtoP(Ng,d_flags,topElement.beginX,topElement.beginP-1,NeighboursinX,*(this->Stream));

	d_in=&(Ng->data[topElement.beginP - NumValuesToMoveFromXToP]);

	//Sort the NumValuesToMoveFromXToP + CurrPSize elements.
	if(topElement.currPSize + NumValuesToMoveFromXToP > 1)
		gpuErrchk(cub::DeviceRadixSort::SortKeys(d_flags,dflagSize,d_in,d_in,topElement.currPSize + NumValuesToMoveFromXToP,0,sizeof(int)*8,*(this->Stream)));

	gpuErrchk(cudaFree(d_flags));

	//remove the nodes from the tracker
	DEV_SYNC;

	tracker->pop(NumValuesToMoveFromXToP);

	//pop the current stack value.
	stack->pop();


}

void BKInstance::RunCliqueFinder(int CliqueId) {

//	//topElement.printconfig();
//	if(CliqueId==4)
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
		int pivot = topElement.pivot;

		RunCliqueFinder(CliqueId);

		//topElement=stack->topElement();

		//stack->pop();

		moveToX();

		while(non_neighbours)
		{
			//moveToX();
			non_neighbours--;
		}

		moveFromXtoP();

		//Bring Back all values used in X into currP array.
	}
}

} /* namespace BK_GPU */
