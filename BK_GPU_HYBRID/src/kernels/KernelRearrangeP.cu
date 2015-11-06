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
void kernelRearrangeGatherP(unsigned int *darray,unsigned int *d_temp,int start_offset,int end_offset,int countZeroes,unsigned *data,BK_GPU::GPU_Stack* stack)
{
	int tid=threadIdx.x + blockDim.x*blockIdx.x;

	//Exceeds limit hence return
	if(tid+start_offset > end_offset)
		return;

	//get the current prefixsum value
	unsigned currVal=darray[tid];

	//get the previous prefixSum value
	unsigned prevVal=(tid==0)?0:darray[tid-1];

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
	unsigned currElement=data[tid+start_offset];

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
void KernelRearrangeScatterP(unsigned *d_temp,int start_offset,int end_offset,unsigned *data)
{
	int tid = threadIdx.x + blockDim.x*blockIdx.x;

	if(tid+start_offset > end_offset)
		return;

	data[tid + start_offset] = d_temp[tid];
}

/**
 * This method is a wrapper method.This method invokes invokes two kernel i.e. KernelRearrangeGatherP and
 * kernelRearrangeScatterP.
 *
 * The purpose of this method call is to Re-arrange the P segment. We are also given as input a Prefix-Sum of an array consisting of Values
 * 0 and 1.The Prefix-Sum array is of the same size as that of the current P segment. The indexes having 1 corresponding to the Prefix-Sum array
 * (i.e subtraction of ith and (i-1)th value is 1) are moved towards the left end i.e. towards the beginP,whereas the values having 0 are moved towards the
 * right while keeping their relative positions same.
 *
 * kernelRearrangeGatherP is used to gather and determine the scatter offset corresponding to each thread and stores it in an auxillary array.
 * KernelRearrangeScatterP is used to copy back the values into the scatter offset from the auxillary array.
 *
 *
 * @param graph Input Neighbor Graph
 * @param stack Current Stack
 * @param InputGraph Graph in CSR format
 * @param darray Auxillary array to store the scatter offset.
 * @param start_offset start_offset of the segment
 * @param end_offset end_offset of the segment
 * @param countZeroes Number of zeroes present in the segment
 * @param stream stream
 */
extern "C"
void GpuArrayRearrangeP(BK_GPU::NeighbourGraph *graph,
		BK_GPU::GPU_Stack* stack,BK_GPU::GPU_CSR *InputGraph,unsigned int *darray,int start_offset,int end_offset,int countZeroes,cudaStream_t &stream)
{
	//Calculate the number of elements present
	int numElements = end_offset - start_offset + 1;

	//Used for auxillary storage.
	unsigned* d_temp;

	//allocate the memory
	CudaError(cudaMalloc(&d_temp,sizeof(unsigned)*numElements));

	///Invoke the Gather Kernel
	kernelRearrangeGatherP<<<ceil((double)numElements/128),128,0,stream>>>(darray,d_temp,start_offset,end_offset,countZeroes,graph->data,stack);

	///Synchronize the previous kernel
	CudaError(cudaStreamSynchronize(stream));

	///Invoke the Scatter Kernel
	KernelRearrangeScatterP<<<ceil((double)numElements/128),128,0,stream>>>(d_temp,start_offset,end_offset,graph->data);

	///Synchronize
	CudaError(cudaStreamSynchronize(stream));

	//Free the memory
	CudaError(cudaFree(d_temp));
}
