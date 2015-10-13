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



namespace BK_GPU {

class BKInstance {
public:

	BK_GPU::GPU_CSR *gpuGraph;
	BK_GPU::NeighbourGraph *Ng;
	BK_GPU::GPU_Stack *stack;
	BK_GPU::StackElement topElement;
	BK_GPU::NeighbourGraph *hostGraph;
	mgpu::ContextPtr Context;
	Graph *host_graph;

	BKInstance(Graph *host_graph,BK_GPU::GPU_CSR *gpuGraph,BK_GPU::NeighbourGraph *Ng,BK_GPU::GPU_Stack *stack);
	void RunCliqueFinder(int CliqueId);
	void processPivot(BK_GPU::StackElement &element);
};

} /* namespace BK_GPU */
#endif /* BKINSTANCE_H_ */

