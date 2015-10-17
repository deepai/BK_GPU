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
	int *elements;

	RecursionStack(int size);

	void *operator new(size_t len);
	void* operator new[](std::size_t count);
	void operator delete(void *ptr);

	__host__ __device__
	void push(int value)
	{
		top++;
		elements[top]=value;
		DEV_SYNC;
	}

	__host__ __device__
	void pop()
	{
		top--;
		DEV_SYNC;
	}

	__host__ __device__
	int size()
	{
		int s=top+1;
		DEV_SYNC;
		return s;
	}

};

} /* namespace BK_GPU */
#endif /* RECURSIONSTACK_H_ */
