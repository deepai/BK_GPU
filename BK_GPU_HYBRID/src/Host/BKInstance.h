/*
 * BKInstance.h
 *
 *  Created on: 06-Oct-2015
 *      Author: debarshi
 */

#ifndef BKINSTANCE_H_
#define BKINSTANCE_H_

#include "../Device/GPUCSR.h"
#include "../Device/NeighbourGraph.h"
#include "../Device/GPUStack.h"
#include "../utilities.h"
#include "../cub/cub.cuh"
#include "../moderngpu/moderngpu.cuh"
#include "../Device/RecursionStack.h"
#include "BKInstanceTest.h"



namespace BK_GPU {

class BKInstance {
public:

	int maxCliqueSizeObtained;

	//Auxiliary memory required for Sorting,InclusiveSum

	BK_GPU::GPU_CSR *gpuGraph;
	BK_GPU::NeighbourGraph *Ng;
	BK_GPU::GPU_Stack *stack;
	BK_GPU::StackElement topElement;
	BK_GPU::StackElement secondElement;
	BK_GPU::RecursionStack *tracker;
	mgpu::ContextPtr *Context;
	Graph *host_graph;
	int MaxThreads;

	int threadIndex;

	BKInstanceTest *testInstance;

	BKInstance(Graph *host_graph,BK_GPU::GPU_CSR *gpuGraph,BK_GPU::NeighbourGraph *Ng,BK_GPU::GPU_Stack *stack,mgpu::ContextPtr *context,int numThreads,int threadIndex);
	~BKInstance();

	void RunCliqueFinder(int CliqueId);
	int processPivot(BK_GPU::StackElement &element);
	void printClique(int CliqueSize,int beginClique);
	void moveToX(int pivot);
	void moveFromXtoP();
	void nextNonPivot(int pivot,int index);
};


} /* namespace BK_GPU */
#endif /* BKINSTANCE_H_ */

