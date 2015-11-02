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

	void topElement(StackElement *topElement) {
		CudaError(cudaMemcpy(topElement,elements+top,sizeof(StackElement),cudaMemcpyDeviceToHost));
	}

	void secondElement(StackElement *secondElement)
	{
		CudaError(cudaMemcpy(secondElement,elements+top-1,sizeof(StackElement),cudaMemcpyDeviceToHost));
	}

	/**
	 * Push topElement into the stack, top is incrememented by 1 place.
	 * @param topElement
	 */
	void push(StackElement *topElement) //true indicates forward and false indicates backward)
			{
		++top;
		CudaError(cudaMemcpy(elements+top,topElement,sizeof(StackElement),cudaMemcpyHostToDevice));

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
