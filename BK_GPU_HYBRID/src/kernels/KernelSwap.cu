#include "../Device/NeighbourGraph.h"
#include "../Device/GPUStack.h"
#include "../Device/GPUCSR.h"
#include "kernels.cuh"

/**
 * This kernel is used to swap two values of the data array in the graph.
 * The values are first copied into the shared memory.
 * Each thread picks up the values from alternate locations in the shared memory  and copies them to their
 * current locations.
 *
 * @param graph Input Neighbour graph
 * @param swapstart 1st swap position
 * @param swapend 2nd swap position
 */
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

/**
 *
 * @param graph Input Neighbour graph
 * @param swapstart 1st swap position
 * @param swapend 2nd swap position
 */
extern "C"
void GpuSwap(BK_GPU::NeighbourGraph *graph,int swapstart,int swapend)
{
	KernelSwap<<<1,32>>>(graph,swapstart,swapend);
	DEV_SYNC;
}
