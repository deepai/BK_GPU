#include "BKInstance.h"
#include "../moderngpu/util/mgpucontext.h"
#include "../kernels/kernels.cuh"
#include "../moderngpu/mgpuhost.cuh"
#include "../cub/device/device_scan.cuh"

using namespace mgpu;

namespace BK_GPU {

/**
 * Requires: Updated TopElement,Updated Stack,Sorted P and X segments.
 *
 * This element finds the next non_neighbour of the pivot.Once the non_neighbour is located it is moved to the end of the current P segment.
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
void BKInstance::nextNonPivot(int pivot,int index)
{

	cudaStream_t currStream = Context[threadIndex]->Stream();

	int NullValue;
	//Obtain the TopElement
	stack->topElement(&topElement,currStream);

	//next pivot to consider is obtained from the current top of the stack - index position
	unsigned nextCandidateNode = tracker->elements->at(tracker->size() - 1 - index);

	//Locate and swap the last zeroes.
	GpuArraySwapNonPivot(Ng,topElement.beginP,topElement.beginP + topElement.currPSize - 1,nextCandidateNode,topElement.beginR,currStream);

	// Check bounds and swap if the selected element is not at beginR - 1 position
	if((topElement.beginP + topElement.currPSize - 1) != (topElement.beginR - 1))
	{
		GpuSwap(Ng,topElement.beginP + topElement.currPSize - 1,topElement.beginR - 1,currStream);
	}


	//CudaError(cudaMemcpy(&nextCandidateNode,Ng->data + topElement.beginR - 1 ,sizeof(unsigned ),cudaMemcpyDeviceToHost));
	//adjacencyList of the Pivot
	unsigned *adjacencyListPivot  = this->gpuGraph->Columns + host_graph->rowOffset[nextCandidateNode];
	int adjacencySize = host_graph->rowOffset[nextCandidateNode+1] - host_graph->rowOffset[nextCandidateNode];

	//Psegment Input Data
	unsigned *PsegmentInputData = Ng->data + topElement.beginP;
	unsigned *PsegmentResults;

	//PsegmentResults are required to store the results of the Sorted Search
	CudaError(cudaMalloc(&PsegmentResults,sizeof(unsigned)*topElement.currPSize));


	if(topElement.currPSize > 1)
	{
		//update the size of acount

		//if P segment was greater than 2, then sort the remaining P segment.
		if(topElement.currPSize > 2)
		{
			void *d_temp_storage=NULL;size_t d_temp_size=0;

			CudaError(cub::DeviceRadixSort::SortKeys(d_temp_storage,d_temp_size,PsegmentInputData,PsegmentInputData,topElement.currPSize - 1,0,sizeof(uint)*8,currStream));

			CudaError(cudaMalloc(&d_temp_storage,d_temp_size));

			if(d_temp_storage==NULL)
						d_temp_storage=&NullValue;

			CudaError(cub::DeviceRadixSort::SortKeys(d_temp_storage,d_temp_size,PsegmentInputData,PsegmentInputData,topElement.currPSize - 1,0,sizeof(uint)*8,currStream));

			CudaError(cudaStreamSynchronize(currStream));

			if(d_temp_storage!=&NullValue)
				CudaError(cudaFree(d_temp_storage));
		}
	}

	int currNeighbour,non_neighbours;

	//Intersection of currP with the neighbors of nextCandidateNode
	SortedSearch<MgpuBoundsLower, MgpuSearchTypeMatch, MgpuSearchTypeNone>(
						PsegmentInputData, topElement.currPSize - 1, adjacencyListPivot, adjacencySize, PsegmentResults, PsegmentResults, *(Context[threadIndex]),
						&currNeighbour, &non_neighbours);

	CudaError(cudaStreamSynchronize(currStream));

	topElement.currPSize = currNeighbour;

	//Do an Inclusive Scan on the intersection values of the adata
	if(topElement.currPSize > 2)
	{
		void *d_temp_storage=NULL;size_t d_temp_size=0;

		//Ist Invocation calculates the amount of memory required for the temporary array.
		CudaError(cub::DeviceScan::InclusiveSum(d_temp_storage,d_temp_size,PsegmentResults,PsegmentResults,topElement.currPSize - 1,currStream));

		CudaError(cudaMalloc(&d_temp_storage,d_temp_size));

		//This step does the actual inclusiveSum
		CudaError(cub::DeviceScan::InclusiveSum(d_temp_storage,d_temp_size,PsegmentResults,PsegmentResults,topElement.currPSize - 1,currStream));

		CudaError(cudaStreamSynchronize(currStream));

		if(d_temp_storage!=&NullValue)
			CudaError(cudaFree(d_temp_storage));

	}

	//Non_neighbour of the current selected candidate Vertex
	non_neighbours = topElement.currPSize - 1 - currNeighbour;

	//if size of neighbors is atleast 1 and less than currPSize
	if((currNeighbour>0) && (currNeighbour < (topElement.currPSize - 1)))
	{
		GpuArrayRearrangeP(this->Ng, this->stack, this->gpuGraph, PsegmentResults,
			topElement.beginP, topElement.beginP + topElement.currPSize - 2,non_neighbours,currStream);
	}


	unsigned *XsegmentInputData = Ng->data + topElement.beginX;
	unsigned *auxXsegmentData;

	//Current X size

	if(topElement.currXSize!=0)
	{
		//Auxilliary X segment Data
		CudaError(cudaMalloc(&auxXsegmentData,sizeof(int)*topElement.currXSize));

		int NeighboursinX, nonNeighboursinX;

		SortedSearch<MgpuBoundsLower, MgpuSearchTypeMatch, MgpuSearchTypeNone>(
						XsegmentInputData, topElement.currXSize, adjacencyListPivot, adjacencySize, auxXsegmentData,auxXsegmentData, *(Context[threadIndex]),
						&NeighboursinX, &nonNeighboursinX);

		CudaError(cudaStreamSynchronize(currStream));

		if(topElement.currXSize > 1)
		{
			/***
			 * * Do a Scan on the current dptr array. We can use the prefix sum to rearrange the neighbours and non-neighbours
			 */		//thrust::inclusive_scan(dptr, dptr + currX, dptr);
			void *d_temp_storage=NULL;size_t d_temp_size=0;

			CudaError(cub::DeviceScan::InclusiveSum(d_temp_storage,d_temp_size,auxXsegmentData,auxXsegmentData,topElement.currXSize,currStream));

			CudaError(cudaMalloc(&d_temp_storage,d_temp_size));

			CudaError(cub::DeviceScan::InclusiveSum(d_temp_storage,d_temp_size,auxXsegmentData,auxXsegmentData,topElement.currXSize,currStream));

			CudaError(cudaStreamSynchronize(currStream));

			if(d_temp_storage!=&NullValue)
				CudaError(cudaFree(d_temp_storage));
		}

		nonNeighboursinX = topElement.currXSize - NeighboursinX;

		if((NeighboursinX > 0) && (NeighboursinX < topElement.currXSize ))
			GpuArrayRearrangeX(Ng,stack,gpuGraph,auxXsegmentData,topElement.beginX,topElement.beginX + topElement.currXSize - 1,nonNeighboursinX,currStream);

		topElement.currXSize = NeighboursinX;

		CudaError(cudaFree(auxXsegmentData));
	}

	topElement.beginR = topElement.beginR - 1;
	topElement.currRSize = topElement.currRSize + 1;
	topElement.pivot = nextCandidateNode;
	topElement.direction = true;

	//CudaError(cudaStreamSynchronize(*(this->Stream)));

	stack->push(&topElement);

		//CudaError(cudaStreamSynchronize(*(this->Stream)));

	}



}


