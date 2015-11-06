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
void kernel_CopyAddress(int *data,int beginP,unsigned int *dptr,int currPSize)
{
	int tid=threadIdx.x + blockIdx.x*blockDim.x;
	if(tid >= currPSize)
		return;
	else
	{
		dptr[tid]=data[beginP+tid];
	}
}

/**
 * A function that copies the currentP segment in the host memory.
 *
 * @param graph Neighbour graph
 * @param beginP Starting location of P segment
 * @param InputGraph Input Graph in CSR format
 * @param dptr array of Pointers
 * @param currPSize currPSize
 */
void GpuCopyOffsetAddresses(BK_GPU::NeighbourGraph *graph,int beginP, BK_GPU::GPU_CSR *InputGraph,unsigned int *host,int currPSize,cudaStream_t &stream)
{
	CudaError(cudaStreamSynchronize(stream));
	//Copy back the values from the dptr to host memory
	CudaError(cudaMemcpy(host,graph->data + beginP, sizeof(unsigned)*currPSize,cudaMemcpyDeviceToHost));

}
