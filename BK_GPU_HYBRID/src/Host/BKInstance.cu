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

using namespace mgpu;

namespace BK_GPU {

BKInstance::BKInstance(Graph *host_graph, BK_GPU::GPU_CSR *gpuGraph,
		BK_GPU::NeighbourGraph *Ng, BK_GPU::GPU_Stack *stack) {
	// TODO Auto-generated constructor stub
	this->Ng = Ng;
	this->gpuGraph = gpuGraph;
	this->stack = stack;
	this->topElement = stack->topElement();
	this->hostGraph = new NeighbourGraph();
	this->hostGraph = Ng;
	this->Context = mgpu::CreateCudaDevice(0);
	this->host_graph = host_graph;
}

void BKInstance::processP(BK_GPU::StackElement &element)
{
	/**Step 1: Find the pivot element
			 */
	int currP = topElement.currPSize; //Size of Number of Elements in P
	unsigned int *d_SortedP; //This is used to store the Unsorted elements initially

	gpuErrchk(cudaMallocManaged(&d_SortedP, sizeof(unsigned int) * currP));

	void *d_temp_storage = NULL; //Auxillary array required for temporary Storage
	size_t d_temp_size = sizeof(unsigned) * currP * 2; //size of auxillary array is 2*N

	//Allocate Auxillary Array
	gpuErrchk(cudaMallocManaged(&d_temp_storage, d_temp_size));

	//Point to the unsorted input data
	unsigned int *d_unSortedP = (unsigned *) &(Ng->data[topElement.beginP]);

	gpuErrchk(cudaDeviceSynchronize());

	//This step does the actual sorting.
	gpuErrchk(
			cub::DeviceRadixSort::SortKeys(d_temp_storage, d_temp_size,
					d_unSortedP, d_SortedP, currP));

	gpuErrchk(cudaDeviceSynchronize());

	//Store the Node Value for each value in the currPArray
	unsigned int *hptr = new unsigned[currP];

	//Kernel to copy the current P Values to the host in the hptr array.
	GpuCopyOffsetAddresses(Ng, stack, gpuGraph, hptr, topElement.currPSize);

	unsigned int* dptr;

	gpuErrchk(cudaMallocManaged(&dptr, sizeof(uint) * currP));

	gpuErrchk(cudaDeviceSynchronize());

	unsigned int *adata = d_SortedP;

	int max_index, numNeighbours = 0;

	int currNeighbour, Helper;
	int acount = currP;

	for (int i = 0; i < currP; i++) {
		int adjacencySize = host_graph->rowOffset[hptr[i] + 1]
				- host_graph->rowOffset[hptr[i]];

		unsigned int *bdata =
				&(gpuGraph->Columns[host_graph->rowOffset[hptr[0]]]);

		gpuErrchk(cudaDeviceSynchronize()); //

		SortedSearch<MgpuBoundsLower, MgpuSearchTypeMatch,
				MgpuSearchTypeNone>(adata, acount, bdata, adjacencySize,
				dptr, dptr, *Context, &currNeighbour, &Helper);

		if (currNeighbour > numNeighbours) {
			max_index = i;
			numNeighbours = currNeighbour;
		}
	}

	int adjacencySize = host_graph->rowOffset[hptr[max_index] + 1]
					- host_graph->rowOffset[hptr[max_index]];

	unsigned *bdata = &(gpuGraph->Columns[host_graph->rowOffset[hptr[max_index]]]);

	SortedSearch<MgpuBoundsLower, MgpuSearchTypeMatch,
					MgpuSearchTypeNone>(adata, acount, bdata, adjacencySize,
					dptr, dptr, *Context, &currNeighbour, &Helper);

	debug(max_index, hptr[max_index], numNeighbours);

	gpuErrchk(cudaFree(d_temp_storage));
	gpuErrchk(cudaFree(d_SortedP));
	gpuErrchk(cudaFree(dptr));

	free(hptr);

}

void BKInstance::RunCliqueFinder(int CliqueId) {

	if ((topElement.currPSize == topElement.currXSize)
			&& (topElement.currXSize == 0)) {		//Obtained a Clique
		return;
	} else if (topElement.currPSize == 0)
		return; //didn't obtain a Clique
	else {
		processP(topElement);
	}
}

} /* namespace BK_GPU */
