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
void BKInstance::nextNonPivot()
{
	int NullValue;
	//Obtain the TopElement
	stack->topElement(&topElement);

	//CudaError(cudaStreamSynchronize(*(this->Stream)));

	//obtain the pivot element
	int pivot=topElement.pivot;

	//AdjacencySize
	int adjacencyPivotSize = host_graph->rowOffset[pivot+1] - host_graph->rowOffset[pivot];

	//Obtain the elements for the bdata
	unsigned int *bdata = gpuGraph->Columns + host_graph->rowOffset[pivot];
	int  bcount = adjacencyPivotSize;

	//obtain the elements for the adata
	unsigned int *adata = (unsigned *)(Ng->data) +topElement.beginP;
	int  acount = topElement.currPSize;

	unsigned *ptr;

	size_t requiredSize = sizeof(uint)*(topElement.currPSize);

	//Allocate memory of size 2*currP
	CudaError(cudaMalloc(&ptr,requiredSize));

//	//Sort P segment once
//	if(topElement.currPSize > 1)
//		CudaError(cub::DeviceRadixSort::SortKeys(ptr,requiredSize,adata,adata,acount,0,sizeof(uint)*8,*(this->Stream)));

	int currNeighbour,non_neighbours;

	//This sorted search is used to know the values which are non-neighbours with respect to pivot.
	//This values are indicated with 0s
	SortedSearch<MgpuBoundsLower, MgpuSearchTypeMatch, MgpuSearchTypeNone>(
					adata, acount, bdata, bcount, ptr, ptr, **Context,
					&currNeighbour, &non_neighbours);

	//Locate and swap the last zeroes.
	GpuArraySwapNonPivot(Ng,(int *)ptr,topElement.beginP,topElement.beginP + topElement.currPSize - 2,currNeighbour,*(this->Stream));

	int nextCandidateNode; //= Ng->data[topElement.beginR-1];

	CudaError(cudaMemcpy(&nextCandidateNode,Ng->data + topElement.beginP + topElement.currPSize - 1 ,sizeof(int),cudaMemcpyDeviceToHost));


	bdata  = (unsigned *)(Ng->data) + host_graph->rowOffset[nextCandidateNode];
	bcount = host_graph->rowOffset[nextCandidateNode+1] - host_graph->rowOffset[nextCandidateNode];

	if(topElement.currPSize > 1)
	{
		//update the size of acount
		acount = topElement.currPSize - 1;

		//if P segment was greater than 2, then sort the remaining P segment.
		if(topElement.currPSize > 2)
		{
			void *d_temp_storage=NULL;size_t d_temp_size=0;

			CudaError(cub::DeviceRadixSort::SortKeys(d_temp_storage,d_temp_size,adata,adata,acount,0,sizeof(uint)*8,*(this->Stream)));

			CudaError(cudaMalloc(&d_temp_storage,d_temp_size));

			if(d_temp_storage==NULL)
						d_temp_storage=&NullValue;

			CudaError(cub::DeviceRadixSort::SortKeys(d_temp_storage,d_temp_size,adata,adata,acount,0,sizeof(uint)*8,*(this->Stream)));

			CudaError(cudaStreamSynchronize(*(this->Stream)));

			if(d_temp_storage!=&NullValue)
				CudaError(cudaFree(d_temp_storage));
		}

		//Intersection of currP with the neighbors of nextCandidateNode
		SortedSearch<MgpuBoundsLower, MgpuSearchTypeMatch, MgpuSearchTypeNone>(
							adata, acount, bdata, bcount, ptr, ptr, **Context,
							&currNeighbour, &non_neighbours);

		//Do an Inclusive Scan on the intersection values of the adata
		if(topElement.currPSize > 2)
		{
			void *d_temp_storage=NULL;size_t d_temp_size=0;

			//Ist Invocation calculates the amount of memory required for the temporary array.
			CudaError(cub::DeviceScan::InclusiveSum(d_temp_storage,d_temp_size,ptr,ptr,topElement.currPSize - 1,*(this->Stream)));

			CudaError(cudaMalloc(&d_temp_storage,d_temp_size));

			//This step does the actual inclusiveSum
			CudaError(cub::DeviceScan::InclusiveSum(d_temp_storage,d_temp_size,ptr,ptr,topElement.currPSize - 1,*(this->Stream)));

			CudaError(cudaStreamSynchronize(*(this->Stream)));

			if(d_temp_storage!=&NullValue)
				CudaError(cudaFree(d_temp_storage));

		}

		//Non-neighbours of the P values.
		non_neighbours = topElement.currPSize - 1 - currNeighbour;

		if((currNeighbour>0) && (currNeighbour < (topElement.currPSize - 1)))
		{
			GpuArrayRearrangeP(this->Ng, this->stack, this->gpuGraph, ptr,
				topElement.beginP, topElement.beginP + topElement.currPSize - 2,non_neighbours,*(this->Stream));
		}

		CudaError(cudaFree(ptr));

		if(topElement.currXSize!=0)
		{

			CudaError(cudaMalloc(&ptr,sizeof(int)*topElement.currXSize));

			adata = (unsigned *)(Ng->data) + topElement.beginX;
			acount = topElement.currXSize;

			int NeighboursinX, nonNeighboursinX;

			SortedSearch<MgpuBoundsLower, MgpuSearchTypeMatch, MgpuSearchTypeNone>(
							adata, acount, bdata, bcount, ptr, ptr, **Context,
							&NeighboursinX, &nonNeighboursinX);

			if(topElement.currXSize > 1)
			{
				/***
				 * * Do a Scan on the current dptr array. We can use the prefix sum to rearrange the neighbours and non-neighbours
				 */		//thrust::inclusive_scan(dptr, dptr + currX, dptr);
				void *d_temp_storage=NULL;size_t d_temp_size=0;

				CudaError(cub::DeviceScan::InclusiveSum(d_temp_storage,d_temp_size,ptr,ptr,topElement.currXSize,*(this->Stream)));

				CudaError(cudaMalloc(&d_temp_storage,d_temp_size));

				CudaError(cub::DeviceScan::InclusiveSum(d_temp_storage,d_temp_size,ptr,ptr,topElement.currXSize,*(this->Stream)));

				CudaError(cudaStreamSynchronize(*(this->Stream)));

				if(d_temp_storage!=&NullValue)
					CudaError(cudaFree(d_temp_storage));
			}

			if((NeighboursinX > 0) && (NeighboursinX < topElement.currXSize ))
				GpuArrayRearrangeX(Ng,stack,gpuGraph,ptr,topElement.beginX,topElement.beginP-1,NeighboursinX,*(this->Stream));

			topElement.currXSize = NeighboursinX;

			CudaError(cudaFree(ptr));
		}

		topElement.currPSize = currNeighbour;
		topElement.beginR = topElement.beginR - 1;
		topElement.currRSize = topElement.currRSize + 1;
		topElement.pivot = nextCandidateNode;
		topElement.direction = true;

		//CudaError(cudaStreamSynchronize(*(this->Stream)));

		stack->push(&topElement);

		//CudaError(cudaStreamSynchronize(*(this->Stream)));

	}



}

}
