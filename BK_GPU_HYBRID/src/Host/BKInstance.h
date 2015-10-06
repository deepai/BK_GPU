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


namespace BK_GPU {

class BKInstance {
public:

	BK_GPU::GPU_CSR *gpuGraph;
	BK_GPU::NeighbourGraph *Ng;
	BK_GPU::GPU_Stack *stack;
	BK_GPU::StackElement topElement;
	BK_GPU::NeighbourGraph *hostGraph;

	BKInstance(BK_GPU::GPU_CSR *gpuGraph,BK_GPU::NeighbourGraph *Ng,BK_GPU::GPU_Stack *stack);
	void RunCliqueFinder(int CliqueId);
};

} /* namespace BK_GPU */
#endif /* BKINSTANCE_H_ */

