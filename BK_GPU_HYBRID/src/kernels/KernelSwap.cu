#include "../Device/NeighbourGraph.h"
#include "../Device/GPUStack.h"
#include "../Device/GPUCSR.h"
#include "kernels.cuh"

__global__
void KernelSwap(BK_GPU::NeighbourGraph *graph,int swapstart,int swapend)
{
	__shared__ int val[2];

	int tid=threadIdx.x;

	if(tid > 1)
		return;

	int offset=(1-tid)*swapstart + tid*swapend;

	val[1-tid]=graph->data[offset];

	graph->data[offset] = val[tid];
}

extern "C"
void GpuSwap(BK_GPU::NeighbourGraph *graph,int swapstart,int swapend)
{
	KernelSwap<<<1,32>>>(graph,swapstart,swapend);
	DEV_SYNC;
}
