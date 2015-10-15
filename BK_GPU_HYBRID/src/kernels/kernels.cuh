/*
 * kernels.cuh
 *
 *  Created on: 14-Sep-2015
 *      Author: debarshi
 */

#ifndef KERNELS_CUH_
#define KERNELS_CUH_

#include <cassert>
#include "../Device/GPUStack.h"
#include "../Device/NeighbourGraph.h"
#include "../Device/GPUCSR.h"

#define tos(stack,i) (stack[i]->topElement())

extern "C" int GpuPivotSelect(BK_GPU::NeighbourGraph &graph,
		BK_GPU::GPU_Stack **stack, BK_GPU::GPU_CSR &InputGraph);

extern "C"
void GpuCopyOffsetAddresses(BK_GPU::NeighbourGraph *graph,
		BK_GPU::GPU_Stack *stack, BK_GPU::GPU_CSR *InputGraph,unsigned int *host,int currPSize);

extern "C"
void GpuArrayRearrangeP(BK_GPU::NeighbourGraph *graph,
		BK_GPU::GPU_Stack* stack,BK_GPU::GPU_CSR *InputGraph,unsigned int *darray,int start_offset,int end_offset);

extern "C"
void GpuArraySwap(BK_GPU::NeighbourGraph *Graph,BK_GPU::GPU_Stack* stack,int swapstart,int swapend);
//extern "C" void GpuChoosePivotNeighbours(BK_GPU::NeighbourGraph &graph,
	//	BK_GPU::GPU_Stack **stack, BK_GPU::GPU_CSR &InputGraph);

#endif /* KERNELS_CUH_ */
