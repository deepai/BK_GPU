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



namespace BK_GPU {

class BKInstance {
public:

	BK_GPU::GPU_CSR *gpuGraph;
	BK_GPU::NeighbourGraph *Ng;
	BK_GPU::GPU_Stack *stack;
	BK_GPU::StackElement topElement;
	BK_GPU::NeighbourGraph *hostGraph;
	BK_GPU::RecursionStack *tracker;
	mgpu::ContextPtr Context;
	Graph *host_graph;
	cudaStream_t *Stream;


	BKInstance(Graph *host_graph,BK_GPU::GPU_CSR *gpuGraph,BK_GPU::NeighbourGraph *Ng,BK_GPU::GPU_Stack *stack,cudaStream_t &stream);
	void RunCliqueFinder(int CliqueId);
	int processPivot(BK_GPU::StackElement &element);
	void printClique(int CliqueSize,int beginClique);
};

} /* namespace BK_GPU */
#endif /* BKINSTANCE_H_ */

