/*
 * StackElement.h
 *
 *  Created on: 09-Sep-2015
 *      Author: debarshi
 */

#ifndef STACKELEMENT_H_
#define STACKELEMENT_H_

#include "../utilities.h"

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
