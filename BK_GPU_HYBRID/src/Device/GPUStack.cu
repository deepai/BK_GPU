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
	CudaError(cudaMallocManaged(&elements, sizeof(StackElement) * size));
	DEV_SYNC;
}


GPU_Stack::~GPU_Stack() {
	// TODO Auto-generated destructor stub
}

void *GPU_Stack::operator new(size_t len) {
	void *ptr;
	CudaError(cudaMallocManaged(&ptr, sizeof(GPU_Stack) * len));
	DEV_SYNC;
	return ptr;
}

void *GPU_Stack::operator new[](std::size_t count) {
	void *ptr;
	CudaError(cudaMallocManaged(&ptr, sizeof(GPU_Stack*) * count))
	DEV_SYNC;
	return ptr;
}

void GPU_Stack::operator delete(void *ptr) {
	DEV_SYNC;
	CudaError(cudaFree(ptr));
}

StackElement& GPU_Stack::operator [](int x) {
	return elements[x];
}

} /* namespace BK_GPU */
