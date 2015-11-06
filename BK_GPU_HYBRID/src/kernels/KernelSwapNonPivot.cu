#include "kernels.cuh"

/**
 * This kernel is used to identify the last element corresponding to a 0 in the darray(inclusive sum).
 *
 * @param graph //Input Graph
 * @param dPos
 * @param darray
 * @param start_offset
 * @param end_offset
 * @param countOnes
 */
__global__
void KernelSwapNonPivot(unsigned *dPos,unsigned* darray,int start_offset,int end_offset,int countOnes)
{
	__shared__ unsigned pos;

	int numElements = end_offset - start_offset + 1;

	//ThreadIndex
	int tid=threadIdx.x + blockIdx.x*blockDim.x;

	if(tid==0)
		pos=start_offset;

	__syncthreads();

	if(tid + start_offset > end_offset)
		return;

	//get the current prefixsum value
	unsigned currVal=darray[tid];

	if(currVal == 0)
		atomicMax(&pos,(unsigned)tid);

	__syncthreads();

	//update from the local shared memory to the global
	if(threadIdx.x == 0)
	{
		atomicMax(dPos,pos);
	}
}

extern "C"
void GpuArraySwapNonPivot(BK_GPU::NeighbourGraph *graph,unsigned* darray,int start_offset,int end_offset,int countOnes,cudaStream_t &stream)
{
	int numElements = end_offset - start_offset + 1;

	if(numElements < 2)
		return;

	unsigned *d_pos;

	CudaError(cudaMalloc(&d_pos,sizeof(int)));

	KernelSwapNonPivot<<<ceil((double)numElements/128),128,0,stream>>>(d_pos,darray,start_offset,end_offset,countOnes);

	CudaError(cudaStreamSynchronize(stream));

	unsigned pos_offset;

	CudaError(cudaMemcpy(&pos_offset,d_pos,sizeof(int),cudaMemcpyDeviceToHost));

	if(pos_offset == end_offset)
		return;

	GpuSwap(graph,pos_offset,end_offset,stream);

	CudaError(cudaFree(d_pos));

	return;

}











