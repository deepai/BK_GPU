#include <cstdio>
#include <cstdlib>
#include "Host/CsrGraph.h"
#include "utilities.h"
#include "Device/GPUStack.h"
#include "Device/NeighbourGraph.h"
#include "Device/GPUCSR.h"
#include "kernels/kernels.cuh"
#include "Host/BKInstance.h"
#include "moderngpu/moderngpu.cuh"
#include "moderngpu/util/mgpucontext.h"
#include <iostream>
#include <algorithm>

#include "cub/cub.cuh"

//#include <cub/cub.cuh>

#define gc fgetc

inline int isSpaceChar(char c) {
	return (c == ' ' || c == '\n' || c == '\r' || c == ',');
}
inline int FAST_IO(FILE* fp) {
	char ch;
	int val = 0;
	ch = gc(fp);
	while (isSpaceChar(ch))
		ch = gc(fp);
	val = 0;
	while (!isSpaceChar(ch)) {
		val = (val * 10) + (ch - '0');
		ch = gc(fp);
	}
	return val;
}

debugger dbg;

bool isBigEndian() {
	unsigned int i = 1;
	char *c = (char*) &i;
	if (*c)
		return false;
	else
		return true;
}

struct listbyPsize
{
	int Psize;
	int index;

	bool operator < (const listbyPsize &rhs) const
	{
		return (Psize > rhs.Psize);
	}

};

int main(int argc, char * argv[]) {
	if (argc < 5) {
		printf(
				"Argument 1 should be path of the Input Matrix File.\n"
						"Argument 2 should be 0 for undirected and 1 for directed.\n"
						"Argument 3 should be 0 for 0 index-based or 1 index based.\n"
						"Argument 4 should be 0 for mtx format or 1 for normal format.\n");
		exit(1);
	}

	FILE* fp = fopen(argv[1], "r");

	bool undirected = atoi(argv[2]);
	bool oneIndexBased = atoi(argv[3]);
	bool nonmtxFormat = atoi(argv[4]);

	int N, E;
	int a, b;

	char c;

	if (!nonmtxFormat)
		while ((c = fgetc(fp)) != '\n')
			;

	if (!nonmtxFormat)
		N = FAST_IO(fp);
	N = FAST_IO(fp);
	E = FAST_IO(fp);

	Graph *g1 = new Graph(N, (!undirected) ? 2 * E : E);

	printf("Edges= %d,Nodes= %d\n", g1->Edges, g1->Nodes);

	for (int i = 0; i < E; i++) {
		a = FAST_IO(fp);
		b = FAST_IO(fp);
		if (!oneIndexBased)
			g1->insertEdges(a - 1, b - 1, undirected);
		else
			g1->insertEdges(a, b, undirected);

	}

	g1->sortEdgeLists();

	g1->calculateKores(isBigEndian()); //Added

	g1->calculateNeighbourArray();

	int Core =
			g1->KCoreValues[g1->neighbourArray[g1->neighbourArray.size() - 1][0]];
	int loc = g1->neighbourArray.size() - 1;

	printf("CoreSize = %d\n", Core);

	int totalSize = 0;
	int countNodes = 0;

	for (int i = g1->neighbourArray.size() - 1; i >= 0; i--) {

		int lastindex = g1->neighbourArray[i].size()-1;

		if (g1->KCoreValues[g1->neighbourArray[i][lastindex]] == Core) {
			loc = i;
			countNodes++;
			totalSize += g1->neighbourArray[loc].size();
			totalSize += g1->preDegeneracyVertices[loc].size();
			//printf("%d \n",totalSize);
		} else
			break;
	}

	BK_GPU::NeighbourGraph *Ng = new BK_GPU::NeighbourGraph(countNodes,
			totalSize);

	printf("Number of Elements = %d,totalSize = %d\n", countNodes, totalSize);

	int offset = 0;

	int nodeIndex = 0;
	int countofStack = countNodes;

	BK_GPU::GPU_Stack **stack;

	cudaMallocManaged(&stack, sizeof(BK_GPU::GPU_Stack*) * countofStack);

	DEV_SYNC;

	//Count of stack = Count Nodes which has Corenumber as Core
	//

	DEV_SYNC;



	std::vector<listbyPsize> L(countNodes);

	for (int i = loc; i < g1->neighbourArray.size(); i++) {


		int Psize = g1->neighbourArray[i].size();
		int Rsize = g1->preDegeneracyVertices[i].size();

		stack[nodeIndex] = new BK_GPU::GPU_Stack(Psize);

		L[nodeIndex].Psize = Psize-1;
		L[nodeIndex].index = nodeIndex;

		Ng->copy(nodeIndex++, offset, (int *) g1->neighbourArray[i].data(),Psize,(int *)g1->preDegeneracyVertices[i].data(),Rsize);

		stack[nodeIndex-1]->push(offset,Rsize ,offset + Rsize, Psize - 1 , offset + Rsize + Psize - 1, 1,
						offset + Rsize, 0, true);

		offset += (Psize + Rsize);
	}

	std::sort(L.begin(),L.end());

	int Cliquesize = Ng->cliqueSize;

	BK_GPU::GPU_CSR *gpuGraph = new BK_GPU::GPU_CSR(*g1);

	//GpuPivotSelect(*Ng, stack, *gpuGraph);

//	BK_GPU::BKInstance *instance=new BK_GPU::BKInstance(g1,gpuGraph,Ng,stack[0]);
//
//	instance->RunCliqueFinder(0);

	//debug("hello");
	fclose(fp);

	return 0;
}

