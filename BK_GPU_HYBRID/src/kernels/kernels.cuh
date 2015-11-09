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


extern "C"
void GpuCopyOffsetAddresses(BK_GPU::NeighbourGraph *graph,int beginP, BK_GPU::GPU_CSR *InputGraph,unsigned int *host,int currPSize,cudaStream_t &stream);

extern "C"
void GpuArrayRearrangeP(BK_GPU::NeighbourGraph *graph,
		BK_GPU::GPU_Stack* stack,BK_GPU::GPU_CSR *InputGraph,unsigned int *darray,int start_offset,int end_offset,int countZeroes,cudaStream_t &stream);

extern "C"
void GpuArrayRearrangeX(BK_GPU::NeighbourGraph *graph,
    BK_GPU::GPU_Stack* stack,BK_GPU::GPU_CSR *InputGraph,unsigned int *darray,int start_offset,int end_offset,int countZeroes,cudaStream_t &stream);

extern "C"
void GpuSwap(BK_GPU::NeighbourGraph *graph,int swapstart,int swapend,cudaStream_t &stream);
//extern "C" void GpuChoosePivotNeighbours(BK_GPU::NeighbourGraph &graph,
	//	BK_GPU::GPU_Stack **stack, BK_GPU::GPU_CSR &InputGraph);

extern "C"
void GpuArrayRearrangeXtoP(BK_GPU::NeighbourGraph *graph,unsigned *darray,int start_offset,int end_offset,int countOnes,cudaStream_t &stream);

extern "C"
void GpuArraySwapNonPivot(BK_GPU::NeighbourGraph *graph,unsigned *darray,int start_offset,int end_offset,int countOnes,cudaStream_t &stream);

#endif /* KERNELS_CUH_ */
