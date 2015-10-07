#include "kernels.cuh"

/**
 * This kernel is used to prefill the array of pointers with the corresponding location of the
 * neighbours from the CSR graph.
 *
 * @param graph Neighbour graph
 * @param stack Stack
 * @param InputGraph Input Graph in CSR format
 * @param dptr array of Pointers
 * @param currPSize currPSize
 */
__global__
void kernel_CopyAddress(BK_GPU::NeighbourGraph *graph,
		BK_GPU::GPU_Stack *stack, BK_GPU::GPU_CSR *InputGraph,unsigned int **dptr,int currPSize)
{
	int tid=threadIdx.x + blockIdx.x*blockDim.x;
	if(tid < currPSize)
		return;
	else
	{
		dptr[tid]=&(InputGraph->Columns[InputGraph->rowOffsets[graph->data[stack->topElement().beginP+tid]]]);
	}
}

/**
 * A wrapper function that invokes the kernel which contains kernels to copy the addresses of the neighbours starting
 * address.
 *
 * @param graph Neighbour graph
 * @param stack Stack
 * @param InputGraph Input Graph in CSR format
 * @param dptr array of Pointers
 * @param currPSize currPSize
 */
void GpuCopyOffsetAddresses(BK_GPU::NeighbourGraph *graph,
		BK_GPU::GPU_Stack *stack, BK_GPU::GPU_CSR *InputGraph,unsigned int **host,int currPSize)
{
	unsigned int **dptr;
	gpuErrchk(cudaMalloc(&dptr,sizeof(unsigned int*)*currPSize));

	kernel_CopyAddress<<<(ceil((double)currPSize/128)),128>>>(graph,stack,InputGraph,dptr,currPSize);
	gpuErrchk(cudaDeviceSynchronize());

	//Copy back the values from the dptr to host memory
	gpuErrchk(cudaMemcpy(host,dptr,sizeof(unsigned int *)*currPSize,cudaMemcpyDeviceToHost));

	gpuErrchk(cudaFree(dptr));
}
