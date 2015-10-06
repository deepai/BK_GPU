/*
 * BKInstance.cpp
 *
 *  Created on: 06-Oct-2015
 *      Author: debarshi
 */

#include "BKInstance.h"

namespace BK_GPU {

BKInstance::BKInstance(BK_GPU::GPU_CSR *gpuGraph,BK_GPU::NeighbourGraph *Ng,BK_GPU::GPU_Stack *stack){
	// TODO Auto-generated constructor stub
	this->Ng=Ng;
	this->gpuGraph=gpuGraph;
	this->stack=stack;
	this->topElement = stack->topElement();
	this->hostGraph = new NeighbourGraph();
	this->hostGraph = Ng;
}

void BKInstance::RunCliqueFinder(int CliqueId)
{
	if((topElement.currPSize == topElement.currXSize) && (topElement.currXSize == 0))
	{		//Obtained a Clique
		return ;
	}
	else if(topElement.currPSize == 0)
		return; //didn't obtain a Clique
	else
	{
		/**Step 1: Find the pivot element
		 */
		int *d_SortedP;
		gpuErrchk(cudaMalloc(&d_SortedP,sizeof(int)*topElement.currPSize));

		void *d_temp_storage = NULL;
		size_t d_temp_size = 0;

		int *d_unSortedP=&(Ng->data[topElement.beginP]);

		//This step is done to obtain the required size of the auxiliary array d_temp_storage

		cub::DeviceRadixSort::SortKeys(d_temp_storage,d_temp_size,d_unSortedP,d_SortedP,topElement.currPSize);

		//This step does the actual sorting.
		gpuErrchk(cudaMalloc(&d_temp_storage,d_temp_size));

		cub::DeviceRadixSort::SortKeys(d_temp_storage,d_temp_size,d_unSortedP,d_SortedP,topElement.currPSize);

		gpuErrchk(cudaFree(d_temp_storage));

	}
}

} /* namespace BK_GPU */
