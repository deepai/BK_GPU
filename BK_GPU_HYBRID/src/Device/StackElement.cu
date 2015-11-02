/*
 * StackElement.cpp
 *
 *  Created on: 09-Sep-2015
 *      Author: debarshi
 */

#include "StackElement.h"

namespace BK_GPU {

StackElement::StackElement() {
	// TODO Auto-generated constructor stub

}

StackElement::~StackElement() {
	// TODO Auto-generated destructor stub
}

StackElement::StackElement(int beginX,int currXSize,int beginP,int currPSize,int beginR,int currRSize,int trackerSize,int pivot,int direction)
{
	this->beginX=beginX;
	this->currXSize=currXSize;
	this->beginP=beginP;
	this->currPSize=currPSize;
	this->beginR=beginR;
	this->currRSize=currRSize;
	this->trackerSize=trackerSize;
	this->pivot=pivot;
	this->direction=direction;
}

}
