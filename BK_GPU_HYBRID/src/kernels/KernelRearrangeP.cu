#include "kernels.cuh"

/**This kernel is used to swap the values and bring the values having 1 in the darray towards the
 * start_offset and the values having 0 towards the end_offset
 * This Kernel acts as a gatherer.
 *
 * @param darray //array containing the prefixSum
 * @param d_temp //array to store the gathered values
 * @param start_offset //startOffset in the graph->data
 * @param end_offset //endoffset
 * @param graph //input graph
 * @param stack //input stack
 */
__global__
void kernelRearrangeGatherP(unsigned int *darray,int *d_temp,int start_offset,int end_offset,int countZeroes,int *data,BK_GPU::GPU_Stack* stack)
{
	int tid=threadIdx.x + blockDim.x*blockIdx.x;

	//Exceeds limit hence return
	if(tid+start_offset > end_offset)
		return;

	//get the current prefixsum value
	int currVal=darray[tid];

	//get the previous prefixSum value
    int prevVal=(tid==0)?0:darray[tid-1];

	int destination; //store destination here

	/**If( nextVal - currVal ) == 1, indicates that tid+start_offset is a neighbour. Hence
	 * ,its destination will be start_offset + currVal - 1(currval indicates number of 1s obtained previously)
	 *
	 * else , it indicates the tid+start_offset is not a neighbour. hence currVal - tid+start_offset
	 * indicates number of 0s preceding it.
	 */

	if(currVal - prevVal == 1)
		destination = currVal - 1;
	else
		destination = end_offset - (countZeroes - (tid+1 - currVal)) - start_offset;

	//Copy the current Element before swapping
	int currElement=data[tid+start_offset];

	d_temp[destination] = currElement;

}

/**
 * In this kernel elements in the d_temp array are copied back into the graph->data
 * starting from start_offset.
 *
 * @param d_temp //temporary storage
 * @param start_offset //start_offset in graph->data
 * @param end_offset //end_offset in graph->data
 * @param graph //graph
 */
__global__
void KernelRearrangeScatterP(int *d_temp,int start_offset,int end_offset,int *data)
{
	int tid = threadIdx.x + blockDim.x*blockIdx.x;

	if(tid+start_offset > end_offset)
		return;

	data[tid + start_offset] = d_temp[tid];
}


extern "C"
void GpuArrayRearrangeP(BK_GPU::NeighbourGraph *graph,
		BK_GPU::GPU_Stack* stack,BK_GPU::GPU_CSR *InputGraph,unsigned int *darray,int start_offset,int end_offset,int countZeroes,cudaStream_t &stream)
{
	int numElements = end_offset - start_offset + 1;

	int* d_temp;

	CudaError(cudaMalloc(&d_temp,sizeof(int)*numElements));

	kernelRearrangeGatherP<<<ceil((double)numElements/128),128,0,stream>>>(darray,d_temp,start_offset,end_offset,countZeroes,graph->data,stack);

	CudaError(cudaStreamSynchronize(stream));

	KernelRearrangeScatterP<<<ceil((double)numElements/128),128,0,stream>>>(d_temp,start_offset,end_offset,graph->data);

	CudaError(cudaStreamSynchronize(stream));

	CudaError(cudaFree(d_temp));
}
