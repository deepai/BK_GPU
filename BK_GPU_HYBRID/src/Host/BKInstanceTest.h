/*
 * BKInstanceTest.h
 *
 *  Created on: 10-Nov-2015
 *      Author: debarshi
 */

#ifndef BKINSTANCETEST_H_
#define BKINSTANCETEST_H_

#include "../Device/GPUStack.h"
#include "../Device/NeighbourGraph.h"
#include "../Device/StackElement.h"
#include "../Host/CsrGraph.h"

namespace BK_GPU {

class BKInstanceTest {
public:

		StackElement currentTopElement;
		StackElement secondElement;

		StackElement finalElement;

		BKInstanceTest();

		void TestProcessPivot(BK_GPU::NeighbourGraph *Ng,BK_GPU::GPU_Stack *stack,Graph *host_graph)
		{
			#ifdef  TEST_ON
			{
				stack->topElement(&currentTopElement);

				int max_index= 0;
				int maxNeighbourinP=0;

				unsigned *Psegment = new unsigned[currentTopElement.currPSize];
				unsigned *Xsegment = new unsigned[currentTopElement.currXSize];

				CudaError(cudaMemcpy(Psegment,Ng->data+currentTopElement.beginP,sizeof(unsigned)*currentTopElement.currPSize,cudaMemcpyDeviceToHost));
				CudaError(cudaMemcpy(Xsegment,Ng->data+currentTopElement.beginX,sizeof(unsigned)*currentTopElement.currXSize,cudaMemcpyDeviceToHost))

				for(int i=0;i<currentTopElement.currPSize;i++)
				{
					unsigned currNode = Psegment[i];
					int numNeighbours = 0;
					for(int j=0;j<currentTopElement.currPSize;j++)
					{
						if(j==i)
							continue;
						else
						{
							int adjacencySize = host_graph->rowOffset[currNode + 1] - host_graph->rowOffset[currNode];

							int beginOffset =  host_graph->rowOffset[currNode];

							if(std::binary_search(host_graph->columns.begin() + beginOffset, host_graph->columns.begin() + beginOffset + adjacencySize,Psegment[j]))
								numNeighbours++;
						}
						if(numNeighbours > maxNeighbourinP)
						{
							maxNeighbourinP = numNeighbours;
							max_index = i;
						}
					}
				}

				unsigned pivot = Psegment[max_index];
				int beginOffset = host_graph->rowOffset[pivot];
				int adjacencySize = host_graph->rowOffset[pivot + 1] - host_graph->rowOffset[pivot];

				int numNeighboursInX=0;

				for(int i=0;i<currentTopElement.currXSize;i++)
				{
					if(std::binary_search(host_graph->columns.begin() + beginOffset, host_graph->columns.begin() + beginOffset + adjacencySize,Xsegment[i]))
						numNeighboursInX++;
				}

				finalElement.beginX = currentTopElement.beginX;
				finalElement.currXSize = numNeighboursInX;
				finalElement.beginP = currentTopElement.beginP;
				finalElement.currPSize = maxNeighbourinP;
				finalElement.beginR = currentTopElement.beginR - 1;
				finalElement.currRSize = currentTopElement.currRSize + 1;

				finalElement.pivot = pivot;
				finalElement.trackerSize = currentTopElement.trackerSize;
				finalElement.direction = true;

				delete Psegment;
				delete Xsegment;

			}

			#endif
			return;
		}

		void TestMoveToX(BK_GPU::NeighbourGraph *Ng,BK_GPU::GPU_Stack *stack,Graph *host_graph,int pivot)
		{
			stack->topElement(&currentTopElement);

			finalElement.beginX = currentTopElement.beginX;
			finalElement.currXSize = currentTopElement.currXSize + 1;
			finalElement.beginP = currentTopElement.beginP + 1;
			finalElement.currPSize = currentTopElement.currPSize;
			finalElement.beginR = currentTopElement.beginR;
			finalElement.currRSize = currentTopElement.currRSize - 1;

			finalElement.pivot = currentTopElement.pivot;
			finalElement.trackerSize = currentTopElement.trackerSize + 1;
			finalElement.direction = true;

		}
};

} /* namespace BK_GPU */
#endif /* BKINSTANCETEST_H_ */
