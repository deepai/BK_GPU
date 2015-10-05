/*
 * NeighbourGraph.h
 *
 *  Created on: 08-Sep-2015
 *      Author: debarshi
 */

#ifndef NEIGHBOURGRAPH_H_
#define NEIGHBOURGRAPH_H_

#include "../utilities.h"
#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <thrust/gather.h>
#include <thrust/scan.h>
#include <thrust/fill.h>
#include <thrust/copy.h>
#include <iterator>

namespace BK_GPU {

class NeighbourGraph {
public:

	int cliqueSize;
	int nodes;
	int totallength;
	int *data;
	int *dataOffset;
	bool *filter;
	int* key;
	int* prefixArray;

	void *operator new(size_t len);
	void operator delete(void *ptr);

	void copy(int nodeindex, int offset, int *neighbours, int size);

	template<typename InputIterator1, typename InputIterator2,
			typename OutputIterator>
	OutputIterator expand(InputIterator1 first1, InputIterator1 last1,
			InputIterator2 first2, OutputIterator output);

	void computeKeyArray(int cliqueSize, int totalSize);

	NeighbourGraph();
	NeighbourGraph(int nodes, int neighbours);

	~NeighbourGraph();
};

} /* namespace BK_GPU */
#endif /* NEIGHBOURGRAPH_H_ */
