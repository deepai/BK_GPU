#include "kernels.cuh"

__global__
void kernelRearrangeGatherXToP(unsigned *darray,unsigned *d_temp,int start_offset,int end_offset,int countOnes,unsigned *data)
{
  int tid=threadIdx.x + blockDim.x*blockIdx.x;

  //Exceeds limit hence return
  if(tid+start_offset > end_offset)
    return;

  //get the current prefixsum value
  unsigned currVal=darray[tid];

  //get the next prefixsum value
  unsigned prevVal=(tid==0)?0:darray[tid-1];

  unsigned destination; //store destination here

  if(currVal - prevVal == 1)
  {
	  destination = end_offset - start_offset - (countOnes - currVal);
  }
  else
  {
	  destination = tid + 1 -currVal;
  }

  //Copy the current Element before swapping
  unsigned currElement=data[tid+start_offset];

  d_temp[destination] = currElement;

}

__global__
void KernelRearrangeScatterXToP(unsigned *d_temp,int start_offset,int end_offset,unsigned *data)
{
  int tid = threadIdx.x + blockDim.x*blockIdx.x;

  if(tid+start_offset > end_offset)
    return;

  data[tid + start_offset] = d_temp[tid];
}

extern "C"
void GpuArrayRearrangeXtoP(BK_GPU::NeighbourGraph *graph,unsigned *darray,int start_offset,int end_offset,int countOnes,cudaStream_t &stream)
{
	int NumElements=end_offset - start_offset + 1;

	unsigned *d_temp;
	CudaError(cudaMalloc(&d_temp,sizeof(int)*NumElements));

	if(NumElements < 2)
		return;

	kernelRearrangeGatherXToP<<<ceil((double)NumElements/128),128,0,stream>>>(darray,d_temp,start_offset,end_offset,countOnes,graph->data);

	CudaError(cudaStreamSynchronize(stream));

	KernelRearrangeScatterXToP<<<ceil((double)NumElements/128),128,0,stream>>>(d_temp,start_offset,end_offset,graph->data);

	CudaError(cudaStreamSynchronize(stream));

	CudaError(cudaFree(d_temp));
}

