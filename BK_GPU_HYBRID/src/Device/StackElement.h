/*
 * StackElement.h
 *
 *  Created on: 09-Sep-2015
 *      Author: debarshi
 */

#ifndef STACKELEMENT_H_
#define STACKELEMENT_H_

#include "../utilities.h"
#include "assert.h"

namespace BK_GPU {

class StackElement {
public:

	int beginX;
	int currXSize;
	int beginP;
	int currPSize;
	int beginR;
	int currRSize;
	int trackerSize;

	int pivot;
	bool direction; //true indicates forward and false indicates backward

	StackElement(int beginX,int currXSize,int beginP,int currPSize,int beginR,int currRSize,int trackerSize,int pivot,int direction);
	StackElement();
	~StackElement();

	void TestEquality(StackElement &otherElement)
	{
		assert(beginX == otherElement.beginX);
		assert(beginP == otherElement.beginP);
		assert(beginR == otherElement.beginR);
		assert(currXSize == otherElement.currXSize);
		assert(currPSize == otherElement.currPSize);
		assert(currRSize == otherElement.currRSize);
		assert(pivot == otherElement.pivot);
		assert(trackerSize == otherElement.trackerSize);
		assert(direction == otherElement.direction);

	}


	__host__
	void printconfig()
	{
		std::cout << "#############################################" << std::endl;
		std::cout << "currPsize : " << currPSize << std::endl;
		std::cout << "currRsize : " << currRSize << std::endl;
		std::cout << "currXsize : " << currXSize << std::endl;
		std::cout << "#############################################" << std::endl;
	}

};

} /* namespace BK_GPU */
#endif /* STACKELEMENT_H_ */
