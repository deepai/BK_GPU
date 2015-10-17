/*
 * CsrGraph.cpp
 *
 *  Created on: 07-Aug-2015
 *      Author: debarshi
 */

#include "CsrGraph.h"
#include <cstdio>

#include <fstream>
#include <iostream>
#include <set>
#include <map>
#include <cstdlib>
#include <algorithm>

template<class T> const T& max(const T& a, const T& b) {
	return (a < b) ? b : a;     // or: return comp(a,b)?b:a; for version (2)
}

template<class T> const T& min(const T& a, const T& b) {
	return (a > b) ? a : b;     // or: return comp(a,b)?b:a; for version (2)
}

Graph::Graph(int N, int E) {
	// TODO Auto-generated constructor stub
	Nodes = N;
	Edges = E;
	build_Edges = 0;

	rows.resize(Edges);
	columns.resize(Edges);

	degree.resize(Nodes);
	rowOffset.resize(Nodes + 1);

	degree.assign(Nodes, 0);
	rowOffset.assign(Nodes + 1, 0);

}

Graph::Graph() {

}

void Graph::initialize(int N, int E) {
	Nodes = N;
	Edges = E;
	build_Edges = 0;

	rows.resize(Edges);
	columns.resize(Edges);

	degree.resize(Nodes);
	rowOffset.resize(Nodes + 1);

	degree.assign(Nodes, 0);
	rowOffset.assign(Nodes + 1, 0);
}

Graph::~Graph() {
	// TODO Auto-generated destructor stub
}

void Graph::insertEdges(int a, int b, bool direction) {

	rows[build_Edges] = a;
	columns[build_Edges] = b;

	//assert(a!=Nodes && b!=Nodes);

	build_Edges++;

	if (!direction)
		insertEdges(b, a, true);
}

unsigned long long Graph::incrementDegree(unsigned long long degree) {
	unsigned upper, lower;
	extractUpperAndLower(degree, upper, lower);

	lower++;

	return joinUpperAndLower(lower, lower);
}
unsigned long long Graph::DecrementDegreeBy2(unsigned long long degree) {
	unsigned upper, lower;
	extractUpperAndLower(degree, upper, lower);

	lower = lower / 2;

	return joinUpperAndLower(lower, lower);
}

void Graph::calculateKores(bool isBigEndian) {
	KCoreValues.resize(Nodes);
	degeneracy.resize(Nodes);

	std::vector<bool>* finishedVertices = new std::vector<bool>(Nodes);
	std::fill(finishedVertices->begin(), finishedVertices->end(), false);

	std::vector<unsignedLL> *degreeValues = &degree;

	unsigned Kcore = 1;

	unsigned EdgeCount = 0;

	int UsedNodes = 0;

	int globalCount = 0;

	while ((EdgeCount < build_Edges) || (UsedNodes < Nodes)) {
		//printf("%d\n",UsedNodes);
		std::vector<unsigned> *_globalSelectedNodes =
				new std::vector<unsigned>();
		//#pragma omp parallel
		{
			std::vector<unsigned> *_selectedNodes = new std::vector<unsigned>();

			//#pragma omp for
			for (unsigned i = 0; i < Nodes; i++) {
				if (!finishedVertices->at(i)) {
					unsigned currentDegree, originalDegree;
					extractUpperAndLower(degree[i], originalDegree,
							currentDegree);

					if (currentDegree < Kcore) {
						_selectedNodes->push_back(i);
					}
				}
			}

			if (_selectedNodes->size() > 0) {
				//#pragma omp critical
				_globalSelectedNodes->insert(_globalSelectedNodes->begin(),
						_selectedNodes->begin(), _selectedNodes->end()); //insert the vectors

			}

			_selectedNodes->clear();
		}

		if (_globalSelectedNodes->size() == 0) {
			Kcore++;
			continue;
		} else {

			if (_globalSelectedNodes->size() > 1)
				sort(_globalSelectedNodes->begin(), _globalSelectedNodes->end(),
						comparatorByDegree(isBigEndian, degreeValues));

			unsigned countEdges = 0;

			//#pragma omp parallel for reduction(+:countEdges)
			for (unsigned i = 0; i < _globalSelectedNodes->size(); i++) {
				unsigned currentDegree, originalDegree;
				extractUpperAndLower(degree[i], originalDegree, currentDegree);

				//countEdges += currentDegree;

				unsigned nodeID = _globalSelectedNodes->at(i);

				unsigned start_offset = rowOffset[nodeID];
				unsigned end_offset = (
						(nodeID == (Nodes - 1)) ? Edges : rowOffset[nodeID + 1]);

				for (unsigned neighbours = start_offset;
						neighbours < end_offset; neighbours++) {

					assert((neighbours >= 0) && (neighbours < Edges));

					if (!((columns[neighbours] >= 0)
							&& (columns[neighbours] < Nodes)))
						printf("%d", columns[neighbours]);

					assert(
							(columns[neighbours] >= 0)
									&& (columns[neighbours] < Nodes));

					if (finishedVertices->at(columns[neighbours]))
						continue;

					//#pragma omp atomic
					degree[columns[neighbours]]--;

					countEdges++;
				}
			}

			//#pragma omp parallel for reduction(+:countEdges)
			for (unsigned i = 0; i < _globalSelectedNodes->size(); i++) {
				//#pragma omp atomic
				globalCount++;

				unsigned nodeID = _globalSelectedNodes->at(i);

				finishedVertices->at(nodeID) = true;

				unsigned *p = (unsigned *) &(degree[nodeID]);

				if (isBigEndian) {
					countEdges += p[1];
				} else
					countEdges += p[0];
			}

			for (unsigned i = 0; i < _globalSelectedNodes->size(); i++) {

				unsigned nodeID = _globalSelectedNodes->at(i);

				KCoreValues[nodeID] = Kcore - 1;
				degeneracy[nodeID] = UsedNodes + (i);

				highestCoreNumber = max(highestCoreNumber, (int) Kcore - 1);

			}

			UsedNodes += _globalSelectedNodes->size();

			EdgeCount += countEdges;

			//printf("Edges = %d,Core=%u , UsedNodes= %d\n",EdgeCount,Kcore,UsedNodes);

			_globalSelectedNodes->clear();
		}

	}

	//debug("globalcount is ",globalCount);
}

/**
 *
 */
void Graph::sortEdgeLists() {
	//assert(build_Edges == Edges);

	std::vector<std::pair<unsigned, unsigned> > *temporary = new std::vector<
			std::pair<unsigned, unsigned> >(Edges);
	for (int i = 0; i < Edges; i++) {
		temporary->at(i).first = rows[i];
		temporary->at(i).second = columns[i];
	}

	std::sort(temporary->begin(), temporary->end(), EdgeSorter());

	for (int i = 0; i < Edges; i++) {
		rows[i] = temporary->at(i).first;
		columns[i] = temporary->at(i).second;

		assert(columns[i] != Nodes);

		rowOffset[rows[i]]++;

		degree[rows[i]] = incrementDegree(degree[rows[i]]);
		degree[columns[i]] = incrementDegree(degree[columns[i]]);
	}
	int prev = 0, curr;
	for (int i = 0; i < Nodes; i++) {
		curr = rowOffset[i];
		rowOffset[i] = prev;
		prev += curr;
		degree[i] = DecrementDegreeBy2(degree[i]);

		if (degree[i] == 0)
			CountInitialZeroNodes++;
	}

	rowOffset[Nodes] = Edges;

	unsigned *p = (unsigned *) &degree[Nodes - 1];

	printf("Edges are %u\n", build_Edges);

	//assert((rowOffset[Nodes - 1] + p[1]) == Edges);

	temporary->clear();

	//delete temporary;

}
void Graph::calculateNeighbourArray() {
	printf("Edges after doubling = %u\n", Edges);
	neighbourArray.resize(Nodes);
	preDegeneracyVertices.resize(Nodes);

	std::set<unsigned> visited;

	for (int i = 0; i < Nodes; i++) {
		int src = i;

		int offset_start = rowOffset[i];
		int offset_end = ((i == (Nodes - 1)) ? Edges : rowOffset[i + 1]);

		unsigned degeneracyOrderSrc = degeneracy[src];

		assert((degeneracyOrderSrc >= 0) && (degeneracyOrderSrc < Nodes));

		assert(visited.find(degeneracyOrderSrc) == visited.end());

		visited.insert(degeneracyOrderSrc);

		neighbourArray[degeneracyOrderSrc].push_back(src);

		for (int j = offset_start; j < offset_end; j++) {
			int dest = columns[j];

			int coreNumberSrc = KCoreValues[src];
			int coreNumberDest = KCoreValues[dest];

			int degeneracyOrderDest = degeneracy[dest];

			if (coreNumberDest >= coreNumberSrc) {
				if (degeneracyOrderDest > degeneracyOrderSrc) {
					neighbourArray[degeneracyOrderSrc].push_back(dest);
				}
				else
				{
					preDegeneracyVertices[degeneracyOrderSrc].push_back(dest);
				}
			}
			else
			{
				preDegeneracyVertices[degeneracyOrderSrc].push_back(dest);
			}
		}
	}

	for(int i=0;i<neighbourArray.size();i++)
	{
		int size=neighbourArray[i].size();

		if(neighbourArray[i].size()>1)
			std::swap(neighbourArray[i][0],neighbourArray[i][size-1]);

		//Sort the array till the second last value. The second last value contains the initial Clique Vertex.
		sort(neighbourArray[i].begin(),neighbourArray[i].begin()+size-1);

		//Sort the Reject list values.
		sort(preDegeneracyVertices[i].begin(),preDegeneracyVertices[i].end());
	}

	printf("Calculating neighbour done\n");


}

void Graph::insertEdgeIfNotExist(int a, int b, bool dir, int &Edges) {

	if ((KCoreValues[a] != highestCoreNumber)
			|| (KCoreValues[b] != highestCoreNumber)) {
		return;
	}

	printf("Edge %d - %d\n", a + 1, b + 1);

	if (visited.find(a) == visited.end()) {
		std::set<int> *temp = new std::set<int>;
		temp->insert(b);
		visited[a] = temp;
		Edges++;
		//g->insertEdges(a,b,true);

		//printf("Edges added = %d - %d\n",a+1,b+1);
	} else {
		if (visited[a]->find(b) == visited[a]->end()) {
			visited[a]->insert(b);
			Edges++;
			//g->insertEdges(a,b,true);

			//printf("Edges added = %d - %d\n",a+1,b+1);

		} else
			return;
	}
	if (!dir) {
		insertEdgeIfNotExist(b, a, true, Edges);
	}

}

int Graph::binary_Search(int low, int high, int val) {

	while (low <= high) {
		int mid = low + (high - low) / 2;
		if (columns[mid] == val)
			return mid;
		else if (val < columns[mid]) {
			high = mid - 1;
		} else
			low = mid + 1;
	}
	return -1;
}

int Graph::countCliques(Graph *g2) {
	int count = 0;

	int EdgesInNewGraph = 0;

	int lesserSizeNodes = 0;
	int correctSizeNodes = 0;
	int extraSizeNodes = 0;

	std::ofstream fout("./Cliques.txt", std::ios::app);

	int max_val = -1, max_index;

	std::map<int, std::set<int> > visited;

	for (int i = 0; i < Nodes; i++) {
		assert(neighbourArray[i].size() > 0);
		int coreNumber = KCoreValues[neighbourArray[i][0]];
		if (coreNumber >= 2) {
			int size = neighbourArray[i].size();

			if (size < (coreNumber + 1)) {
				lesserSizeNodes++;
				continue;
			} else if (size == (coreNumber + 1)) {
				correctSizeNodes++;
			} else
				extraSizeNodes++;

			if (size < (coreNumber + 1)) {
				int val1, val2;
				for (int t = 0; t < (size - 1); t++) {
					val1 = neighbourArray[i][t];
					for (int s = t + 1; s < size; s++) {
						val2 = neighbourArray[i][s];
						int pos = binary_Search(rowOffset[val1],
								(val1 == (Nodes - 1)) ?
										Edges - 1 : rowOffset[val1 + 1] - 1,
								val2);
						if (pos != -1) //there is an edge between val1 and val2
								{
							degree[t]++;
							degree[s]++;

							//insertEdgeIfNotExist(val1,val2,false,EdgesInNewGraph);

						}
					}

				}
				continue;
			}

			int degree[size];

			memset(degree, 0, sizeof(int) * size);

			int val1, val2;
			for (int t = 0; t < (size - 1); t++) {
				val1 = neighbourArray[i][t];
				for (int s = t + 1; s < size; s++) {
					val2 = neighbourArray[i][s];
					int pos = binary_Search(rowOffset[val1],
							(val1 == (Nodes - 1)) ?
									Edges - 1 : rowOffset[val1 + 1] - 1, val2);
					if (pos != -1) {
						degree[t]++;
						degree[s]++;
					}
				}

			}
			int countminDegreeVertices = 0;
			for (int t = 0; t < size; t++) {
				if (degree[t] >= coreNumber)
					countminDegreeVertices++;
			}
			if (countminDegreeVertices >= (coreNumber + 1)) //contains a Clique
					{
				if (max_val < coreNumber) {
					max_val = coreNumber;
					max_index = i;
				}

				for (int j = 0; j < neighbourArray[i].size(); j++) {
					fout << (neighbourArray[i][j] + 1) << " ";
				}
				fout << std::endl;

				count++;
			} else { //donot contain a clique

				for (int t = 0; t < (size - 1); t++) {
					val1 = neighbourArray[i][t];
					for (int s = t + 1; s < size; s++) {
						val2 = neighbourArray[i][s];
						int pos = binary_Search(rowOffset[val1],
								(val1 == (Nodes - 1)) ?
										Edges - 1 : rowOffset[val1 + 1] - 1,
								val2);
						if (pos != -1) //there is an edge between val1 and val2
								{
							degree[t]++;
							degree[s]++;

							insertEdgeIfNotExist(val1, val2, false,
									EdgesInNewGraph);

						}
					}

				}
			}
		}
	}

	//g2=new Graph(Nodes,EdgesInNewGraph);
	g2->initialize(Nodes, EdgesInNewGraph);

	for (std::map<int, std::set<int>*>::iterator itr = this->visited.begin();
			itr != this->visited.end(); itr++) {
		for (std::set<int>::iterator itr2 = itr->second->begin();
				itr2 != itr->second->end(); itr2++) {
			g2->insertEdges(itr->first, *itr2, true);
		}
	}

//	if(max_val > -1)
//	{
//		fout << "Max is " << max_val << std::endl;
//		for(int j=0;j<neighbourArray[max_index].size();j++)
//		{
//			fout << (neighbourArray[max_index][j] + 1) << " ";
//		}
//		fout << std::endl;
//	}

	fout.close();

	//printf("Less = %d,Normal = %d and Extra = %d,Edges added=%d\n",lesserSizeNodes,correctSizeNodes,extraSizeNodes,g2->build_Edges);

	return count;
}

