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

	int pivot_index;
	int remainingNonNeighbour;
	bool direction; //true indicates forward and false indicates backward

	StackElement();
	~StackElement();

};

} /* namespace BK_GPU */
#endif /* STACKELEMENT_H_ */
