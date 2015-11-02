/*
 * NeighbourGraph.cpp
 *
 *  Created on: 08-Sep-2015
 *      Author: debarshi
 */

#include "NeighbourGraph.h"

namespace BK_GPU {

NeighbourGraph::NeighbourGraph() {
	// TODO Auto-generated constructor stub

}

/**
 * The Input parameters
 *
 * @param nodes //indicates Number of Cliques
 * @param neighbours //indicates Totallength of all CliqueSize. i.e. Length of each Clique*nodes
 */
NeighbourGraph::NeighbourGraph(int nodes, int neighbours) {
	this->cliqueSize = neighbours / nodes;
	this->nodes = nodes;
	this->totallength = neighbours;

	CudaError(cudaMalloc(&dataOffset, sizeof(int) * (this->nodes + 1)));
	CudaError(cudaMalloc(&data, sizeof(int) * (this->totallength)));

}


NeighbourGraph::~NeighbourGraph() {
	// TODO Auto-generated destructor stub
	CudaError(cudaFree(data));
	CudaError(cudaFree(dataOffset));
}

/**
 * This input array passed to this method is of the type  R|P|X where R contains a single Node, P contains the neighbours of R's vertex and X is empty.
 * @param nodeindex //Index of the Nodes in the neighbour graph whose cliques we need to search.
 * @param offset //offset is the starting position of the gpulist where we copy the values.
 * @param neighbour //neighbour array containing the vertices which we shall copy.
 * @param size //size contains the length of the |1+P| vertices. The rowoffset shall be filled by this value.
 */
void NeighbourGraph::copy(int nodeindex, int offset, int *neighbours, int Psize,int *rejectLists,int Rsize) {

	CudaError(
			cudaMemcpy(data + offset, rejectLists, sizeof(int) * Rsize,
					cudaMemcpyHostToDevice));
	CudaError(cudaMemcpy(data + offset + Rsize,neighbours,sizeof(int) * Psize,cudaMemcpyHostToDevice));

	//DEV_SYNC;

	int lastValue=offset + Psize + Rsize;

	CudaError(cudaMemcpy(dataOffset+nodeindex,&offset,sizeof(int),cudaMemcpyHostToDevice));
	CudaError(cudaMemcpy(dataOffset+nodeindex+1,&lastValue,sizeof(int),cudaMemcpyHostToDevice));

}

template<typename InputIterator1, typename InputIterator2,
		typename OutputIterator>
OutputIterator NeighbourGraph::expand(InputIterator1 first1,
		InputIterator1 last1, InputIterator2 first2, OutputIterator output) {
	typedef typename thrust::iterator_difference<InputIterator1>::type difference_type;

	difference_type input_size = thrust::distance(first1, last1);
	difference_type output_size = thrust::reduce(first1, last1);

	// scan the counts to obtain output offsets for each input element
	thrust::device_vector<difference_type> output_offsets(input_size, 0);
	thrust::exclusive_scan(first1, last1, output_offsets.begin());

	// scatter the nonzero counts into their corresponding output positions
	thrust::device_vector<difference_type> output_indices(output_size, 0);
	thrust::scatter_if(thrust::counting_iterator<difference_type>(0),
			thrust::counting_iterator<difference_type>(input_size),
			output_offsets.begin(), first1, output_indices.begin());

	// compute max-scan over the output indices, filling in the holes
	thrust::inclusive_scan(output_indices.begin(), output_indices.end(),
			output_indices.begin(), thrust::maximum<difference_type>());

	// gather input values according to index array (output = first2[output_indices])
	OutputIterator output_end = output;
	thrust::advance(output_end, output_size);
	thrust::gather(output_indices.begin(), output_indices.end(), first2,
			output);

	// return output + output_size
	thrust::advance(output, output_size);
	return output;
}



} /* namespace BK_GPU */
