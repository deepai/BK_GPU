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
void KernelSwapNonPivot(unsigned *data,int start_offset,int end_offset,int beginR,unsigned currentNode)
{
	__shared__ unsigned pos;

	int numElements = end_offset - start_offset + 1;

	//ThreadIndex
	int tid=threadIdx.x + blockIdx.x*blockDim.x;

	if(tid + start_offset > end_offset)
		return;

	int destination;
	unsigned currValue = data[tid + start_offset];

	//set the destination for each value.
	if(currValue < currentNode) //destination for values less than the pivot remains same
	{
		destination = tid + start_offset;
	}
	else if(currValue > currentNode) // destination for values greater than pivot is decreased by 1
	{
		destination = tid + start_offset - 1;
	}
	else //pivot element is copied at the end_offset
	{
		destination = end_offset;
	}

	__syncthreads(); //synchronize

	data[destination] = currValue; //copy the current value to the destination array.

}

extern "C"
void GpuArraySwapNonPivot(BK_GPU::NeighbourGraph *graph,int start_offset,int end_offset,unsigned Node,int beginR,cudaStream_t &stream)
{
	int numElements = end_offset - start_offset + 1;

	if(numElements < 2)
		return;

	//CudaError(cudaMalloc(&d_pos,sizeof(int)));

	KernelSwapNonPivot<<<ceil((double)numElements/128),128,0,stream>>>(graph->data,start_offset,end_offset,beginR,Node);

	CudaError(cudaStreamSynchronize(stream));

}











