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
		//CudaError(cudaMemcpyAsync(topElement,elements+top-1,sizeof(StackElement),cudaMemcpyDeviceToHost,stream));
		//CudaError(cudaStreamSynchronize(stream));
		topElement->beginP 	  = elements[top - 1].beginP;
		topElement->currPSize = elements[top - 1].currPSize;
		topElement->beginX = elements[top - 1].beginX;
		topElement->currRSize = elements[top - 1].currRSize;
		topElement->beginR = elements[top - 1].beginR;
		topElement->currXSize = elements[top - 1].currXSize;
		topElement->direction = elements[top - 1].direction;
		topElement->pivot = elements[top - 1].pivot;
		topElement->trackerSize = elements[top - 1].trackerSize;
	}

	void secondElement(StackElement *secondElement,cudaStream_t &stream)
	{
		secondElement->beginP 	  = elements[top - 2].beginP;
		secondElement->currPSize = elements[top - 2].currPSize;
		secondElement->beginX = elements[top - 2].beginX;
		secondElement->currRSize = elements[top - 2].currRSize;
		secondElement->beginR = elements[top - 2].beginR;
		secondElement->currXSize = elements[top - 2].currXSize;
		secondElement->direction = elements[top - 2].direction;
		secondElement->pivot = elements[top - 2].pivot;
		secondElement->trackerSize = elements[top - 2].trackerSize;
	}

	/**
	 * Push topElement into the stack, top is incrememented by 1 place.
	 * @param topElement
	 */
	void push(StackElement *topElement,cudaStream_t &stream) //true indicates forward and false indicates backward)
	{
		//CudaError(cudaMemcpyAsync(elements+top,topElement,sizeof(StackElement),cudaMemcpyHostToDevice,stream));
		//CudaError(cudaStreamSynchronize(stream));
		elements[top].beginP = topElement->beginP;
		elements[top].beginR = topElement->beginR;
		elements[top].beginX = topElement->beginX;
		elements[top].currXSize = topElement->currXSize;
		elements[top].currPSize = topElement->currPSize;
		elements[top].currRSize = topElement->currRSize;
		elements[top].pivot = topElement->pivot;
		elements[top].direction = topElement->direction;
		elements[top].trackerSize = topElement->trackerSize;

		++top;
	}

	void push(StackElement *topElement) //true indicates forward and false indicates backward)
	{
			//CudaError(cudaMemcpy(elements+top,topElement,sizeof(StackElement),cudaMemcpyHostToDevice));
		elements[top].beginP = topElement->beginP;
		elements[top].beginR = topElement->beginR;
		elements[top].beginX = topElement->beginX;
		elements[top].currXSize = topElement->currXSize;
		elements[top].currPSize = topElement->currPSize;
		elements[top].currRSize = topElement->currRSize;
		elements[top].pivot = topElement->pivot;
		elements[top].direction = topElement->direction;
		elements[top].trackerSize = topElement->trackerSize;

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
