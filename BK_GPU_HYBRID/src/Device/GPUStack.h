/*
 * GPUStack.h
 *
 *  Created on: 09-Sep-2015
 *      Author: debarshi
 */

#ifndef GPUSTACK_H_
#define GPUSTACK_H_

#include "StackElement.h"

namespace BK_GPU {

class GPU_Stack {

public:

	int top;
	int maxCliqueSize;
	StackElement *elements;

	GPU_Stack(int size);
	~GPU_Stack();
	//void* operator new[](std::size_t count);

	void topElement(StackElement *topElement,cudaStream_t &stream) {
		CudaError(cudaMemcpyAsync(topElement,elements+top-1,sizeof(StackElement),cudaMemcpyDeviceToHost,stream));
		CudaError(cudaStreamSynchronize(stream));
	}

	void secondElement(StackElement *secondElement,cudaStream_t &stream)
	{
		CudaError(cudaMemcpyAsync(secondElement,elements+top-2,sizeof(StackElement),cudaMemcpyDeviceToHost,stream));
		CudaError(cudaStreamSynchronize(stream));
	}

	/**
	 * Push topElement into the stack, top is incrememented by 1 place.
	 * @param topElement
	 */
	void push(StackElement *topElement,cudaStream_t &stream) //true indicates forward and false indicates backward)
			{
		CudaError(cudaMemcpyAsync(elements+top,topElement,sizeof(StackElement),cudaMemcpyHostToDevice,stream));
		CudaError(cudaStreamSynchronize(stream));
		++top;
	}

	void push(StackElement *topElement) //true indicates forward and false indicates backward)
	{
			CudaError(cudaMemcpy(elements+top,topElement,sizeof(StackElement),cudaMemcpyHostToDevice));
			++top;
	}

	/**
	 * Reduces the value of stack by 1.
	 */
	void pop() {
		//DEV_SYNC;
		top--;
		//DEV_SYNC;
	}
};

} /* namespace BK_GPU */
#endif /* GPUSTACK_H_ */
