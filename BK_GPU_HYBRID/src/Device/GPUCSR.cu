/*
 * GPUCSR.cpp
 *
 *  Created on: 14-Sep-2015
 *      Author: debarshi
 */

#include "GPUCSR.h"

namespace BK_GPU {

GPU_CSR::~GPU_CSR()
{
	CudaError(cudaFree(this->rowOffsets));
	CudaError(cudaFree(this->Columns));
}

GPU_CSR::GPU_CSR(Graph &graph) {
	// TODO Auto-generated constructor stub
	this->Nodes = graph.Nodes;
	this->Edges = graph.Edges;

	//copy from host memory to device_memory
	CudaError(
			cudaMalloc(&(this->rowOffsets),
					sizeof(int) * (this->Nodes + 1)));
	CudaError(
			cudaMemcpy(this->rowOffsets, graph.rowOffset.data(),
					sizeof(unsigned) * (this->Nodes + 1),
					cudaMemcpyHostToDevice));
	DEV_SYNC;

	CudaError(cudaMalloc(&(this->Columns), sizeof(int) * (this->Edges)));

	CudaError(
			cudaMemcpy(this->Columns, graph.columns.data(),
					sizeof(unsigned) * (this->Edges), cudaMemcpyHostToDevice));

	DEV_SYNC;

}

} /* namespace BK_GPU */
