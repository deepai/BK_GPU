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
		BK_GPU::NeighbourGraph *Ng, BK_GPU::GPU_Stack *stack,cudaStream_t &stream,mgpu::ContextPtr *context,int numThreads) {

	this->maxCliqueSizeObtained = 1;
	// TODO Auto-generated constructor stub
	this->Ng = Ng; 								//Neighbor graph allocated in Device memory
	this->gpuGraph = gpuGraph; 					//CSR Graph allocated in Device memory
	this->stack = stack; 						//Stack for the current Clique , allocated in the device memory

	stack->topElement(&(this->topElement)); 	//StackElement resident in the CPU memory

	this->Stream= &stream;						//cudastream

	this->MaxThreads = numThreads;

	this->Context = context;


	this->host_graph = host_graph;				//Graph allocated in the host memory

	this->tracker = new BK_GPU::RecursionStack(topElement.currPSize); //This device resident is used to keep a track of non_neighbours for a given evalution of BK

	testInstance = new BKInstanceTest();
}

/**
 * This method is used to calculate the maximum buffer size for auxilliary and working array
 * It takes the max corresponding to both the Inclusive Sum and Exclusive Sum.
 */


BKInstance::~BKInstance()
{
	delete this->tracker;
	delete this->stack;
	delete this->testInstance;

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

	#ifdef TEST_ON
	{
		testInstance->TestProcessPivot(Ng,stack,host_graph);
	}
	#endif

	/**Step 1: Find the pivot element
	 *
	 */
	//This location is used to point to by a NULL reference
	int NullValue;

	unsigned int *PsegmentOutput; 		  //Pointer to Neighbour graph of a vertex

	//Point to the unsorted input data (i.e. P Array in the device graph)
	unsigned int *PsegmentInput = (Ng->data) + topElement.beginP;
	PsegmentOutput = PsegmentInput;

	//cudaStreamSynchronize(*(this->Stream));
	//Host memory used to store the current P array in the host.To make pivoting faster.
	unsigned int *CpuPsegment = new unsigned[topElement.currPSize];

	//Kernel to copy the current P array to the host in the hptr array.
	GpuCopyOffsetAddresses(Ng, topElement.beginP, gpuGraph, CpuPsegment, topElement.currPSize,*(this->Stream));

	//Various uses of auxillary pointer
	unsigned int* auxillaryStorage;

	//set a maximum size of (curr*P)
	CudaError(cudaMalloc(&auxillaryStorage, sizeof(int) * (topElement.currPSize)));

	//DEV_SYNC;


	/** Max Index is used to store the index of value within P
	 *  Max Index lies between 0 and P-1.
	 */
	int max_index=0;

	int currNeighbour=0, non_neighbours;

	/** For each value in the P segment. Obtain the count of its neighbors amongst P .
	 *  The value with the maximum neighbor count is selected as the pivot. Other non-neighbors are also
	 *  selected after the pivot is completed.
	 *  This helps avoid unnecessary computations.
	 *
	 */
	int nsize,nonnsize;
	#pragma omp parallel private(nsize,nonnsize)
	{
		int currNeighbourSize=0;
		int best_index=0;

		#pragma omp for nowait
		for (int i = 1; i < topElement.currPSize; i++)
		{

			int threadIdx=omp_get_thread_num();

			int adjacencySize = (host_graph->rowOffset[CpuPsegment[i] + 1] - host_graph->rowOffset[CpuPsegment[i]]);

				//std::cout << adjacencySize << ", " << host_graph->rowOffset[hptr[i] + 1] << " " << host_graph->rowOffset[hptr[i]]<< std::endl;

			unsigned int *adjacencyList =gpuGraph->Columns + host_graph->rowOffset[CpuPsegment[i]];

				//DEV_SYNC
				//; //

			SortedSearch<MgpuBoundsLower, MgpuSearchTypeMatch, MgpuSearchTypeNone>(
						PsegmentInput,topElement.currPSize, adjacencyList, adjacencySize, auxillaryStorage, auxillaryStorage,*(Context[threadIdx]),
						&nsize, &nonnsize);

			CudaError(cudaStreamSynchronize(((Context[threadIdx]->Stream()))));

			if(nsize > currNeighbourSize)
			{
				currNeighbourSize = nsize;
				best_index=i;
			}
		}
		#pragma omp critical
		{
			if (currNeighbourSize > currNeighbour) {
				max_index = best_index;
				currNeighbour = currNeighbourSize;
			}
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

	//Rsize is incremented by 1.
	int newRsize = topElement.currRSize + 1;
	int newPsize;

	//adjacency size of the neighbor array.
	int adjacencySize = host_graph->rowOffset[CpuPsegment[max_index] + 1]
			- host_graph->rowOffset[CpuPsegment[max_index]];

	//pointer to the beginning of the adjacency list for the maximum value.
	unsigned *adjacencyListPivot = gpuGraph->Columns + host_graph->rowOffset[CpuPsegment[max_index]];

//Use different Threads for computation of P and X segments
	#pragma omp parallel num_threads(2)
	{
		int threadId = omp_get_thread_num();

		if(threadId == 0)
		{
			cudaStream_t currStream = Context[threadId]->Stream();

			if(topElement.currPSize > 2 )
			{
				//pointer and size variable to allocate temporary array.
				void *d_temp_storage=NULL;size_t d_temp_size=0;

				//One call to prefill the d_temp_size with required memory size.
				CudaError(cub::DeviceRadixSort::SortKeys(d_temp_storage, d_temp_size,
								PsegmentInput, PsegmentOutput, topElement.currPSize - 1,0,sizeof(uint)*8,currStream));

				//Allocate appropiate memory for d_temp_storage required for radixSort.
				CudaError(cudaMalloc(&d_temp_storage,d_temp_size));

				if(d_temp_storage==NULL)
					d_temp_storage=&NullValue;

				//Perform the actual Radix Sort.
				CudaError(cub::DeviceRadixSort::SortKeys(d_temp_storage, d_temp_size,
										PsegmentInput, PsegmentOutput, topElement.currPSize - 1,0,sizeof(uint)*8,currStream));

				//Synchronize the stream
				CudaError(cudaStreamSynchronize(currStream));

				//Free memory.
				if(d_temp_storage!=&NullValue)
					CudaError(cudaFree(d_temp_storage));

			}



			//This calculates the number of remaining non-neighbors of pivot.
			SortedSearch<MgpuBoundsLower, MgpuSearchTypeMatch, MgpuSearchTypeNone>(
					PsegmentInput,topElement.currPSize - 1, adjacencyListPivot, adjacencySize, auxillaryStorage, auxillaryStorage, *(Context[threadId]),
					&currNeighbour, &non_neighbours);

			CudaError(cudaStreamSynchronize(currStream));

			newPsize = currNeighbour;

			//Only if the P segment size is greater than 2, do an Inclusive Sum.
			if(topElement.currPSize > 2)
			{
				void *d_temp_storage=NULL;size_t d_temp_size=0;

				//Ist Invocation calculates the amount of memory required for the temporary array.
				CudaError(cub::DeviceScan::InclusiveSum(d_temp_storage,d_temp_size,auxillaryStorage,auxillaryStorage,topElement.currPSize - 1,currStream));

				CudaError(cudaMalloc(&d_temp_storage,d_temp_size));

				if(d_temp_storage==NULL)
					d_temp_storage=&NullValue;

				//This step does the actual inclusiveSum
				CudaError(cub::DeviceScan::InclusiveSum(d_temp_storage,d_temp_size,auxillaryStorage,auxillaryStorage,topElement.currPSize - 1,currStream));

				//Synchronize the stream
				CudaError(cudaStreamSynchronize(currStream));

				//Free the allocated memory.
				if(d_temp_storage!=&NullValue)
					CudaError(cudaFree(d_temp_storage));
			}

			//Obtain the count of Non_Neighbours of the pivot.
			non_neighbours = topElement.currPSize - 1 - currNeighbour;

			//call Kernel Here to re-arrange P elements
			if((currNeighbour>0) && (currNeighbour < (topElement.currPSize-1)))
			{
				GpuArrayRearrangeP(this->Ng, this->stack, this->gpuGraph, auxillaryStorage,
					topElement.beginP, topElement.beginP + topElement.currPSize - 2,non_neighbours,currStream);
			}

			//Declare a vector of size non_neighbours + 1;
			unsigned *list_non_neighbour = new unsigned[max(1,non_neighbours)];


			//Copy back the non_neighbours from device to the list.
			CudaError(cudaMemcpyAsync(list_non_neighbour,Ng->data + topElement.beginP + currNeighbour,sizeof(unsigned) * non_neighbours,cudaMemcpyDeviceToHost,currStream));

			CudaError(cudaStreamSynchronize(currStream));

			//push main pivot
			tracker->push(CpuPsegment[max_index]);

			//push non_neighbours of pivot
			for(int i=0;i<non_neighbours;i++)
				tracker->push(list_non_neighbour[i]);

			//delete the auxilliary list
			delete list_non_neighbour;

		}
		else if(threadId == 1)
		{
			//Repeat the steps for currX.
			//Intersection with X
			cudaStream_t currStream = Context[threadId]->Stream();

			if (topElement.currXSize != 0)
			{
				unsigned *auxStorage;

				//allocate memory for auxiliary space for X arrays
				CudaError(cudaMalloc(&auxStorage, sizeof(unsigned) * (topElement.currXSize)));

				unsigned *XsegmentInput = (Ng->data) + topElement.beginX;;

				int NeighboursinX, nonNeighboursinX;

				SortedSearch<MgpuBoundsLower, MgpuSearchTypeMatch, MgpuSearchTypeNone>(
						XsegmentInput, topElement.currXSize, adjacencyListPivot, adjacencySize, auxStorage, auxStorage, *(Context[threadId]),
						&NeighboursinX, &nonNeighboursinX);

				CudaError(cudaStreamSynchronize(currStream));

				//Scan only if currX size is greater than 1.
				if(topElement.currXSize > 1)
				{
					/***
					 * * Do a Scan on the current dptr array. We can use the prefix sum to rearrange the neighbours and non-neighbours
					 */		//thrust::inclusive_scan(dptr, dptr + currX, dptr);
					void *d_temp_storage=NULL;size_t d_temp_size=0;

					CudaError(cub::DeviceScan::InclusiveSum(d_temp_storage,d_temp_size,auxStorage,auxStorage,topElement.currXSize,currStream));

					CudaError(cudaMalloc(&d_temp_storage,d_temp_size));

					if(d_temp_storage==NULL)
						d_temp_storage=&NullValue;

					CudaError(cub::DeviceScan::InclusiveSum(d_temp_storage,d_temp_size,auxStorage,auxStorage,topElement.currXSize,currStream));

					CudaError(cudaStreamSynchronize(Context[threadId]->Stream()));

					if(d_temp_storage!=&NullValue)
						CudaError(cudaFree(d_temp_storage));

					//DEV_SYNC;
				}

				/***
				 * Scan Complete
				 */
				nonNeighboursinX = topElement.currXSize - NeighboursinX;

				if((NeighboursinX > 0) && (NeighboursinX < topElement.currXSize ))
					GpuArrayRearrangeX(Ng,stack,gpuGraph,auxStorage,topElement.beginX,topElement.beginX + topElement.currXSize - 1,nonNeighboursinX,currStream);

				topElement.currXSize = NeighboursinX;

				CudaError(cudaFree(auxStorage));
			}
		}
	}

	CudaError(cudaFree(auxillaryStorage));
//
	//Sort the P segment is currPsize > 2
	int trackerSize = tracker->size() ;

	//CudaError(cudaStreamSynchronize(*(this->Stream)));
	topElement.beginR = newBeginR;
	topElement.currPSize = newPsize;
	topElement.currRSize = newRsize;
	topElement.direction = true;
	topElement.pivot = CpuPsegment[max_index];
	topElement.trackerSize = trackerSize;

	stack->push(&topElement);

	#ifdef TEST_ON
	{
		testInstance->TestPivotEnd(topElement,Ng,stack,host_graph);
	}
	#endif

	/**Free the pointers **/

	delete[] CpuPsegment;

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
void BKInstance::moveToX(int pivot)
{

	int NullValue;

	stack->pop();

	//Update the topElement.
	stack->topElement(&topElement);

	#ifdef TEST_ON
	{
		testInstance->TestMoveToX(Ng,stack,host_graph,topElement.pivot);
	}
	#endif

	CudaError(cudaStreamSynchronize(*(this->Stream)));

	//Old_posElement is the last position of the P array.
	int old_posElementPivot = topElement.beginR - 1;

	//new position for the X value would be at secondElement.beginX + secondElement.currXSize
	int new_posElementPivot = topElement.beginX + topElement.currXSize;

	//swap the positions
	GpuSwap(this->Ng,old_posElementPivot,new_posElementPivot,*(this->Stream));

	//If beginP is not swapped, swap it with the old_position to move back the X element into its previous position
	if(new_posElementPivot!=topElement.beginP)
	{
		//swap the positions.
		GpuSwap(this->Ng,topElement.beginP,old_posElementPivot,*(this->Stream));

		//Since P segment might not extend till beginR, an extra swap might be required to correctly set the values
		if((topElement.beginP + topElement.currPSize - 1)!= (topElement.beginR - 1))
		{
			GpuSwap(this->Ng,topElement.beginP+topElement.currPSize - 1,topElement.beginR - 1,*(this->Stream));
		}
	}

	//beginR will shift right
	//beginP will shift towards right
	//currXSize will increase
	//currRSize will decrease
	//currPSize remains the same.

	topElement.currXSize = topElement.currXSize + 1;
	topElement.beginP = topElement.beginP + 1;
	topElement.currPSize = topElement.currPSize - 1;
	topElement.pivot = pivot;
	//topElement.beginR = topElement.beginR + 1;
	//topElement.currRSize = topElement.currRSize - 1;

	//pop the current top of the stack
	//stack->pop();

	#pragma omp parallel num_threads(2)
	{
		int threadId = omp_get_thread_num();

		if(threadId == 0)
		{
			cudaStream_t currStream = Context[threadId]->Stream();
			//Sort if currPSize > 1
			if(topElement.currPSize > 1)
			{

				void *d_temp_storage=NULL;size_t d_temp_size=0;

				unsigned *PsegmentInput=Ng->data + topElement.beginP;
				unsigned *PsegmentOutput=PsegmentInput;

				CudaError(cub::DeviceRadixSort::SortKeys(d_temp_storage,d_temp_size,PsegmentInput,PsegmentOutput,topElement.currPSize,0,sizeof(uint)*8,currStream));

				CudaError(cudaMalloc(&d_temp_storage,d_temp_size));

				if(d_temp_storage==NULL)
					d_temp_storage=&NullValue;

				//Sort the array.
				CudaError(cub::DeviceRadixSort::SortKeys(d_temp_storage,d_temp_size,PsegmentInput,PsegmentOutput,topElement.currPSize,0,sizeof(uint)*8,currStream));

				CudaError(cudaStreamSynchronize(currStream));

				if(d_temp_storage!=&NullValue)
					CudaError(cudaFree(d_temp_storage));
			}
		}
		else
		{
			cudaStream_t currStream = Context[threadId]->Stream();

			//Sort if currXSize > 1
			if(topElement.currXSize > 1)
			{

				void *d_temp_storage=NULL;size_t d_temp_size=0;

				unsigned *XsegmentInput=Ng->data + topElement.beginX;
				unsigned *XsegmentOutput=XsegmentInput;

				CudaError(cub::DeviceRadixSort::SortKeys(d_temp_storage,d_temp_size,XsegmentInput,XsegmentOutput,topElement.currXSize,0,sizeof(int)*8,currStream));

				CudaError(cudaMalloc(&d_temp_storage,d_temp_size));

				if(d_temp_storage==NULL)
					d_temp_storage=&NullValue;

				//Sort the array.
				CudaError(cub::DeviceRadixSort::SortKeys(d_temp_storage,d_temp_size,XsegmentInput,XsegmentOutput,topElement.currXSize,0,sizeof(int)*8,currStream));

				CudaError(cudaStreamSynchronize(currStream));

				if(d_temp_storage!=&NullValue)
					CudaError(cudaFree(d_temp_storage));
			}
		}
	}

	topElement.trackerSize = tracker->size();

	//Pop the values of the secondElement
	//stack->pop();

	//Push the new configuration values into the stack.
	stack->push(&topElement);

	//Push the newly moved element into the tracker.
	//tracker->push(topElement.pivot);

	#ifdef TEST_ON
	{
		testInstance->TestMoveToXEnd(topElement,Ng,stack,host_graph);
	}
	#endif

	//CudaError(cudaStreamSynchronize(*(this->Stream)));

}

void BKInstance::printClique(int CliqueSize,int beginClique)
{
#ifdef PRINTCLIQUES

	unsigned *Clique=new unsigned[CliqueSize];

	CudaError(cudaMemcpy(Clique,Ng->data+beginClique,sizeof(unsigned)*CliqueSize,cudaMemcpyDeviceToHost));

	for(int i=0;i<CliqueSize;i++)
		printf("%d ",Clique[i]+1);

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
	#ifdef TEST_ON
	{
		testInstance->TestMoveFromXToP(Ng,stack,host_graph);
	}
	#endif

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
	unsigned *trackerInput,*trackerOutput;

	CudaError(cudaMalloc(&trackerInput,sizeof(unsigned)*NumValuesToMoveFromXToP));

	trackerOutput = trackerInput;

	//CudaError(cudaMemcpy(d_in,tracker->elements->data() + secondElement.trackerSize,sizeof(unsigned) * NumValuesToMoveFromXToP,cudaMemcpyHostToDevice));
	unsigned *Non_neighbours = new unsigned[NumValuesToMoveFromXToP];

	//Fill from the vector from the last
	for(int i=0;i<NumValuesToMoveFromXToP;i++)
	{
		Non_neighbours[NumValuesToMoveFromXToP - 1 - i] = tracker->getTopElement();
		tracker->pop(1);
	}

	//Memory Allocation memcpy
	CudaError(cudaMemcpyAsync(trackerInput,Non_neighbours,sizeof(unsigned) * NumValuesToMoveFromXToP,cudaMemcpyHostToDevice,*(this->Stream)));

	CudaError(cudaStreamSynchronize(*(this->Stream)));

	delete[] Non_neighbours;


	//Sorting is done only if NumValues > 1
	if(NumValuesToMoveFromXToP > 1)
	{
		//void *ptr;
		void *d_temp_storage=NULL; size_t d_temp_size=0;

		CudaError(cub::DeviceRadixSort::SortKeys(d_temp_storage,d_temp_size,trackerInput,trackerOutput,NumValuesToMoveFromXToP,0,sizeof(int)*8,*(this->Stream)));

		CudaError(cudaMalloc(&d_temp_storage,d_temp_size));

		if(d_temp_storage==NULL)
			d_temp_storage=&NullValue;

		//Sort the Xvalues
		CudaError(cub::DeviceRadixSort::SortKeys(d_temp_storage,d_temp_size,trackerInput,trackerOutput,NumValuesToMoveFromXToP,0,sizeof(int)*8,*(this->Stream)));

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
	unsigned* SearchResults;
	size_t dflagSize = sizeof(int)*(topElement.beginP - topElement.beginX);


	unsigned *XsegmentInput=Ng->data + topElement.beginX;
	int XToPSegment=topElement.beginP - topElement.beginX; //The size of the bdata array is X####. We need to search it there

	int NeighboursinX,nonNeighboursinX;

	//Allocate memory for the flags
	CudaError(cudaMalloc(&SearchResults,dflagSize));

	//Initialize the memory by 0
	CudaError(cudaMemset(SearchResults,0,sizeof(unsigned)*XToPSegment));

	//Do a Sorted Search to check which values in bdata matches with values in  adata.
	SortedSearch<MgpuBoundsLower, MgpuSearchTypeMatch, MgpuSearchTypeNone>(
					trackerInput,NumValuesToMoveFromXToP , XsegmentInput, topElement.currXSize, SearchResults, SearchResults, **Context,
					&NeighboursinX, &nonNeighboursinX);

	//if bcount > 1 , do an inclusive sum.|bcount represents the whole X#### segment.
	if(XToPSegment > 1)
	{
		void *d_temp_storage=NULL;size_t d_temp_size=0;

		//Inclusive Sum
		CudaError(cub::DeviceScan::InclusiveSum(d_temp_storage,d_temp_size,SearchResults,SearchResults,XToPSegment,*(this->Stream)));

		CudaError(cudaMalloc(&d_temp_storage,d_temp_size));

		if(d_temp_storage==NULL)
			d_temp_storage=&NullValue;

		CudaError(cub::DeviceScan::InclusiveSum(d_temp_storage,d_temp_size,SearchResults,SearchResults,XToPSegment,*(this->Stream)));

		CudaError(cudaStreamSynchronize(*(this->Stream)));

		if(d_temp_storage!=&NullValue)
			CudaError(cudaFree(d_temp_storage));

	}

	//CudaError(cudaStreamSynchronize(*(this->Stream)));

	//This kernel is used to rearrange back the X values towards P.
	GpuArrayRearrangeXtoP(Ng,SearchResults,topElement.beginX,topElement.beginP-1,NeighboursinX,*(this->Stream));

	CudaError(cudaFree(SearchResults));
	CudaError(cudaFree(trackerInput));

	#pragma omp parallel num_threads(2)
	{
		int threadId = omp_get_thread_num();

		if(threadId == 0)
		{
			unsigned *PsegmentInputData =Ng->data + secondElement.beginP;

			//Sort the NumValuesToMoveFromXToP + CurrPSize elements = secondElement.currPSize.
			if(secondElement.currPSize > 1)
			{
				void *d_temp_storage=NULL;size_t d_temp_size=0;

				CudaError(cub::DeviceRadixSort::SortKeys(d_temp_storage,d_temp_size,PsegmentInputData,PsegmentInputData,secondElement.currPSize,0,sizeof(int)*8,*(this->Stream)));

				CudaError(cudaMalloc(&d_temp_storage,d_temp_size));

				if(d_temp_storage==NULL)
					d_temp_storage=&NullValue;

				CudaError(cub::DeviceRadixSort::SortKeys(d_temp_storage,d_temp_size,PsegmentInputData,PsegmentInputData,secondElement.currPSize,0,sizeof(int)*8,*(this->Stream)));

				CudaError(cudaStreamSynchronize(*(this->Stream)));

				if(d_temp_storage!=&NullValue)
					CudaError(cudaFree(d_temp_storage));
			}
		}
		else
		{
			unsigned *XsegmentInputData =Ng->data + secondElement.beginX;

			if(secondElement.currXSize > 1)
			{
				void *d_temp_storage=NULL;size_t d_temp_size=0;

				CudaError(cub::DeviceRadixSort::SortKeys(d_temp_storage,d_temp_size,XsegmentInputData,XsegmentInputData,secondElement.currXSize,0,sizeof(int)*8,*(this->Stream)));

				CudaError(cudaMalloc(&d_temp_storage,d_temp_size));

				if(d_temp_storage==NULL)
					d_temp_storage=&NullValue;

				CudaError(cub::DeviceRadixSort::SortKeys(d_temp_storage,d_temp_size,XsegmentInputData,XsegmentInputData,secondElement.currXSize,0,sizeof(int)*8,*(this->Stream)));

				CudaError(cudaStreamSynchronize(*(this->Stream)));

				if(d_temp_storage!=&NullValue)
					CudaError(cudaFree(d_temp_storage));
			}

		}

	}

	stack->pop();

	//remove the nodes from the tracker
	//tracker->pop(NumValuesToMoveFromXToP);

	#ifdef TEST_ON
	{
		testInstance->finalElement.TestEquality(secondElement);
	}
	#endif


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
		moveToX(pivot);

		//While there are non_neighbours, continue invoking the recursive function.
		for(int i=0;i<non_neighbours && (topElement.currPSize!=0) ;i++)
		{
			//Obtains the nextNonPivot Element
			nextNonPivot(pivot,i);

			int nextPivot = topElement.pivot;

			//On Expansion the current configuration would only result in a smaller CliqueSize.
			if(topElement.currRSize + topElement.currPSize > maxCliqueSizeObtained )
				RunCliqueFinder(CliqueId);

			//Move elements back to X.
			moveToX(nextPivot);
			//non_neighbours--;
		}

		//Move All elements back from X to P.
		moveFromXtoP();

		//Bring Back all values used in X into currP array.
	}
}

} /* namespace BK_GPU */
