/*
 * RecursionStack.cpp
 *
 *  Created on: 16-Oct-2015
 *      Author: debarshi
 */

#include "RecursionStack.h"

namespace BK_GPU {

RecursionStack::RecursionStack(int size) {
	// TODO Auto-generated constructor stub
	top=0;
	elements=new std::vector<unsigned>();
	elements->reserve(size);

}

RecursionStack::~RecursionStack()
{
	elements->clear();
}

} /* namespace BK_GPU */
