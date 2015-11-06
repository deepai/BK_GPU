/*
 * RecursionStack.h
 *
 *  Created on: 16-Oct-2015
 *      Author: debarshi
 */

#ifndef RECURSIONSTACK_H_
#define RECURSIONSTACK_H_

#include "../utilities.h"

namespace BK_GPU {

class RecursionStack {
public:

	int top;
	unsigned *elements;

	RecursionStack(int size,cudaStream_t &stream);

	~RecursionStack();

	void push(int value)
	{
		CudaError(cudaMemcpy(elements+top,&value,sizeof(int),cudaMemcpyHostToDevice));
		top++;
	}

	void pop(int count)
	{
		top-=count;
		//DEV_SYNC;
	}

	int getTopElement()
	{
		int topVal;

		CudaError((cudaMemcpy(&topVal,elements+top,sizeof(int),cudaMemcpyHostToDevice)));

		return topVal;
	}


	int size()
	{
		return top;
	}

};

} /* namespace BK_GPU */
#endif /* RECURSIONSTACK_H_ */
