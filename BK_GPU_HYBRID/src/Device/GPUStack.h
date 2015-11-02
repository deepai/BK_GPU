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
	void *operator new(size_t len);
	//void* operator new[](std::size_t count);

	void operator delete(void *ptr);

	__host__
	void attachStream(cudaStream_t &stream);

	__host__ __device__
	StackElement& operator[](int x);

	__host__ __device__
	StackElement& topElement() {
		return elements[top];
	}

	__host__ __device__
	StackElement& secondElement()
	{
		return elements[top-1];
	}

	/**
	 * This method adds the values to the current top of the Stack
	 *
	 * @param beginX starting index of X(Reject list)
	 * @param currXSize current size of X(Reject list)
	 * @param beginP starting index of P(candidates)
	 * @param currPSize current size of P(candidates)
	 * @param beginR starting index of R(partial cliques)
	 * @param currRSize current size of R(partial Cliques)
	 * @param pivot_index (selected pivot in the previous step)
	 * @param remainingNonNeighbour (number of non-neighbours still remaining to explore)
	 * @param direction (true for forward pass and false for backward pass)
	 */
	__host__ __device__
	void push(int beginX, int currXSize, int beginP, int currPSize, int beginR,
			int currRSize, int pivot_index,int stackSize, int remainingNonNeighbour,
			bool direction) //true indicates forward and false indicates backward)
			{
		++top;

		elements[top].beginX = beginX;
		elements[top].currXSize = currXSize;
		elements[top].beginP = beginP;
		elements[top].currPSize = currPSize;
		elements[top].beginR = beginR;
		elements[top].currRSize = currRSize;
		elements[top].pivot = pivot_index;
		elements[top].remainingNonNeighbour = remainingNonNeighbour;
		elements[top].direction = direction;
		elements[top].trackerSize = stackSize;

	}

	/**
	 * Reduces the value of stack by 1.
	 */
	__host__ __device__
	void pop() {
		//DEV_SYNC;
		top--;
		//DEV_SYNC;
	}
};

} /* namespace BK_GPU */
#endif /* GPUSTACK_H_ */
