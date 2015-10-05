/*
 * GPUCSR.h
 *
 *  Created on: 14-Sep-2015
 *      Author: debarshi
 */

#ifndef GPUCSR_H_
#define GPUCSR_H_

#include "../Host/CsrGraph.h"
#include "../utilities.h"

namespace BK_GPU {

class GPU_CSR {
public:
	int Nodes;
	int Edges;

	unsigned *rowOffsets;
	unsigned *Columns;

	GPU_CSR(Graph &host_graph);

	void *operator new(size_t len);
	void operator delete(void *ptr);

};

} /* namespace BK_GPU */
#endif /* GPUCSR_H_ */
