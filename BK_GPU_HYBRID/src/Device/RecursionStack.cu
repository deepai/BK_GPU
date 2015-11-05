/*
 * RecursionStack.cpp
 *
 *  Created on: 16-Oct-2015
 *      Author: debarshi
 */

#include "RecursionStack.h"

namespace BK_GPU {

RecursionStack::RecursionStack(int size,cudaStream_t &stream) {
	// TODO Auto-generated constructor stub
	top=0;
	CudaError(cudaMalloc(&elements,sizeof(int)*size));

}

RecursionStack::~RecursionStack()
{
	CudaError(cudaFree(this->elements));
}

} /* namespace BK_GPU */
