#include "kernels.cuh"

__global__
void kernelRearrangeGatherXToP(int *darray,int *d_temp,int start_offset,int end_offset,int countOnes,BK_GPU::NeighbourGraph *graph)
{
  int tid=threadIdx.x + blockDim.x*blockIdx.x;

  //Exceeds limit hence return
  if(tid+start_offset > end_offset)
    return;

  //get the current prefixsum value
  int currVal=darray[tid];

  //get the next prefixsum value
  int prevVal=(tid==0)?0:darray[tid-1];

  int destination; //store destination here

  if(currVal - prevVal == 1)
  {
	  destination = end_offset - start_offset - (countOnes - currVal);
  }
  else
  {
	  destination = tid + 1 -currVal;
  }

  //Copy the current Element before swapping
  int currElement=graph->data[tid+start_offset];

  d_temp[destination] = currElement;

}

__global__
void KernelRearrangeScatterXToP(int *d_temp,int start_offset,int end_offset,BK_GPU::NeighbourGraph *graph)
{
  int tid = threadIdx.x + blockDim.x*blockIdx.x;

  if(tid+start_offset > end_offset)
    return;

  graph->data[tid + start_offset] = d_temp[tid];
}

extern "C"
void GpuArrayRearrangeXtoP(BK_GPU::NeighbourGraph *graph,int* darray,int start_offset,int end_offset,int countOnes,cudaStream_t &stream)
{
	int NumElements=end_offset - start_offset + 1;

	int *d_temp;
	gpuErrchk(cudaMalloc(&d_temp,sizeof(int)*NumElements));

	kernelRearrangeGatherXToP<<<ceil((double)NumElements/128),128,0,stream>>>(darray,d_temp,start_offset,end_offset,countOnes,graph);

	DEV_SYNC;

	KernelRearrangeScatterXToP<<<ceil((double)NumElements/128),128,0,stream>>>(d_temp,start_offset,end_offset,graph);

	DEV_SYNC;

	gpuErrchk(cudaFree(d_temp));
}

