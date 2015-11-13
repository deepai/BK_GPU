/*
 * RecursionStack.h
 *
 *  Created on: 16-Oct-2015
 *      Author: debarshi
 */

#ifndef RECURSIONSTACK_H_
#define RECURSIONSTACK_H_

#include "../utilities.h"
#include <vector>

namespace BK_GPU {

class RecursionStack {
public:

	int top;
	std::vector<unsigned> *elements;

	RecursionStack(int size);
	~RecursionStack();


	void push(int value)
	{
		elements->push_back(value);
		top=elements->size();
	}

	void pop(int count)
	{
		elements->pop_back();
		top = elements->size();
		//DEV_SYNC;
	}

	unsigned getTopElement()
	{
		return elements->back();
	}


	int size()
	{
		return top;
	}

};

} /* namespace BK_GPU */
#endif /* RECURSIONSTACK_H_ */
