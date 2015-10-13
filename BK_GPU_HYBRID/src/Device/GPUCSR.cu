/*
 * GPUCSR.cpp
 *
 *  Created on: 14-Sep-2015
 *      Author: debarshi
 */

#include "GPUCSR.h"

namespace BK_GPU {

GPU_CSR::GPU_CSR(Graph &graph) {
	// TODO Auto-generated constructor stub
	this->Nodes = graph.Nodes;
	this->Edges = graph.Edges;

	//copy from host memory to device_memory
	gpuErrchk(
			cudaMallocManaged(&(this->rowOffsets),
					sizeof(int) * (this->Nodes + 1)));
	gpuErrchk(
			cudaMemcpy(this->rowOffsets, graph.rowOffset.data(),
					sizeof(unsigned) * (this->Nodes + 1),
					cudaMemcpyHostToDevice));
	DEV_SYNC;

	gpuErrchk(cudaMallocManaged(&(this->Columns), sizeof(int) * (this->Edges)));

	gpuErrchk(
			cudaMemcpy(this->Columns, graph.columns.data(),
					sizeof(unsigned) * (this->Edges), cudaMemcpyHostToDevice));

	DEV_SYNC;

}

void *GPU_CSR::operator new(size_t len) {
	void *ptr;
	gpuErrchk(cudaMallocManaged(&ptr, len * sizeof(BK_GPU::GPU_CSR)));
	DEV_SYNC;
	return ptr;
}

void GPU_CSR::operator delete(void *ptr) {
	DEV_SYNC;
	gpuErrchk(cudaFree(ptr));
}

} /* namespace BK_GPU */
