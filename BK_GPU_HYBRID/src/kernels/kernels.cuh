/*
 * kernels.cuh
 *
 *  Created on: 14-Sep-2015
 *      Author: debarshi
 */

#ifndef KERNELS_CUH_
#define KERNELS_CUH_

#include <cassert>

#define tos(stack,i) (stack[i]->topElement())

extern "C" int GpuPivotSelect(BK_GPU::NeighbourGraph &graph,
		BK_GPU::GPU_Stack **stack, BK_GPU::GPU_CSR &InputGraph);

//extern "C" void GpuChoosePivotNeighbours(BK_GPU::NeighbourGraph &graph,
	//	BK_GPU::GPU_Stack **stack, BK_GPU::GPU_CSR &InputGraph);

#endif /* KERNELS_CUH_ */
