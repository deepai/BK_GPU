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
	top = 0;
	maxCliqueSize = size;
	CudaError(cudaMalloc(&elements, sizeof(StackElement) * (size+1)));

}


GPU_Stack::~GPU_Stack() {
	// TODO Auto-generated destructor stub
	CudaError(cudaFree(elements));
}


} /* namespace BK_GPU */
