/*
 * CsrGraph.h
 *
 *  Created on: 07-Aug-2015
 *      Author: debarshi
 */

#ifndef CSRGRAPH_H_
#define CSRGRAPH_H_

#include <vector>
#include <algorithm>
#include <assert.h>
#include <cstring>
#include <omp.h>
#include <iostream>
#include <set>
#include <map>
#include "../utilities.h"

class Graph {
public:

	struct EdgeSorter {
		bool operator ()(const std::pair<unsigned, unsigned> &a,
				const std::pair<unsigned, unsigned> &b) {
			if (a.first < b.first) {
				return true;
			} else if (a.first == b.first) {
				return (a.second < b.second);
			} else
				return false;
		}
	};

	struct comparatorByDegree {
		bool big_endian;

		std::vector<unsignedLL> *degree;

		bool operator()(const unsigned &a, const unsigned &b) const {
			unsigned currentDegreeA, currentDegreeB;

			unsigned *p1 = (unsigned *) &(degree->at(a)), *p2 =
					(unsigned *) &(degree->at(b));

			if (big_endian) {
				currentDegreeA = p1[1];
				currentDegreeB = p2[1];
			} else {
				currentDegreeA = p1[0];
				currentDegreeB = p2[0];
			}

			return (currentDegreeA < currentDegreeB);
		}

		comparatorByDegree(bool endian, std::vector<unsignedLL> *degreeGraph) {
			big_endian = endian;
			degree = degreeGraph;
		}
	};

	void extractUpperAndLower(unsignedLL val, unsigned &upper,
			unsigned &lower) {
		upper = (val >> 32) & (4294967295ul);
		lower = (val) & (4294967295ul);

	}

	/**
	 * Returns a unsignedLL Value which contains upper in its upper 32 bits and lower in its lower 32 bits.
	 * @param upper
	 * @param lower
	 */
	unsignedLL joinUpperAndLower(unsigned upper, unsigned lower) {
		unsignedLL val = 0;
		val = val | upper;
		val = val << 32;
		val = val | lower;
		return val;
	}

	void insertEdgeIfNotExist(int a, int b, bool dir, int &Edges);

	unsigned Nodes, Edges, build_Edges, CountInitialZeroNodes = 0;

	int highestCoreNumber;

	std::vector<unsigned int> rows; //rows of the graph ,     i.e. Endpoint of an edge.
	std::vector<unsigned int> rowOffset;
	std::vector<unsigned int> columns; //columns of the graph i.e. Endpoint of an edge.

	std::map<int, std::set<int>*> visited;

	std::vector<unsigned long long> degree; //degree of the vertices

	//This ordering contains the K-core value in its upper 32 bits
	//This ordering contains the end-index in its lower 32 bits
	std::vector<unsigned long long> indexKOrderingOffsets;

	std::vector<unsigned> KCoreValues;
	std::vector<unsigned> degeneracy;

	std::vector<std::vector<unsigned> > neighbourArray;

	Graph(int N, int E);
	Graph();

	void initialize(int N, int E);

	virtual ~Graph();

	void insertEdges(int a, int b, bool direction);

	void calculateKores(bool isEndian);

	void sortEdgeLists();

	void calculateNeighbourArray();

	int countCliques(Graph *g2);

	int binary_Search(int low, int high, int val);

	unsigned long long incrementDegree(unsigned long long degree);
	unsigned long long DecrementDegreeBy2(unsigned long long degree);

};

#endif /* CSRGRAPH_H_ */
