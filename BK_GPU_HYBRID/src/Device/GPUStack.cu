/*
 * GPUStack.cpp
 *
 *  Created on: 09-Sep-2015
 *      Author: debarshi
 */

#include "GPUStack.h"

namespace BK_GPU {

GPU_Stack::GPU_Stack(int size) {
	// TODO Auto-generated constructor stub
	top = -1;
	gpuErrchk(cudaMallocManaged(&elements, sizeof(StackElement) * size));
	gpuErrchk(cudaDeviceSynchronize());
}

GPU_Stack::~GPU_Stack() {
	// TODO Auto-generated destructor stub
}

void *GPU_Stack::operator new(size_t len) {
	void *ptr;
	gpuErrchk(cudaMallocManaged(&ptr, sizeof(GPU_Stack) * len));
	gpuErrchk(cudaDeviceSynchronize());
	return ptr;
}

void *GPU_Stack::operator new[](std::size_t count) {
	void *ptr;
	gpuErrchk(cudaMallocManaged(&ptr, sizeof(GPU_Stack*) * count))
	gpuErrchk(cudaDeviceSynchronize());
	return ptr;
}

void GPU_Stack::operator delete(void *ptr) {
	gpuErrchk(cudaDeviceSynchronize());
	gpuErrchk(cudaFree(ptr));
}

StackElement& GPU_Stack::operator [](int x) {
	return elements[x];
}

} /* namespace BK_GPU */
