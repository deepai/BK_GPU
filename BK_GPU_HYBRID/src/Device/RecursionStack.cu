/*
 * RecursionStack.cpp
 *
 *  Created on: 16-Oct-2015
 *      Author: debarshi
 */

#include "RecursionStack.h"

namespace BK_GPU {

RecursionStack::RecursionStack(int size) {
	// TODO Auto-generated constructor stub
	top=0;
	CudaError(cudaMallocManaged(&elements,sizeof(int)*size));
	//DEV_SYNC;
}

void *RecursionStack::operator new(size_t len) {
	void *ptr;
	CudaError(cudaMallocManaged(&ptr, sizeof(RecursionStack) * len));
	//DEV_SYNC;
	return ptr;
}

void *RecursionStack::operator new[](std::size_t count) {
	void *ptr;
	CudaError(cudaMallocManaged(&ptr, sizeof(RecursionStack) * count))
	//DEV_SYNC;
	return ptr;
}

void RecursionStack::operator delete(void *ptr) {
	DEV_SYNC;
	CudaError(cudaFree(ptr));
}

} /* namespace BK_GPU */
