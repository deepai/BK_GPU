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
#include <assert.h>

#include <vector>
#include <utility>

namespace BK_GPU {

class BKInstanceTest {
public:

		StackElement currentTopElement;
		StackElement secondElement;

		StackElement finalElement;

		unsigned *Psegment;
		unsigned *Xsegment;

		BKInstanceTest();

		struct Compare
		{
			bool operator()(const std::pair<unsigned,bool> &a,const std::pair<unsigned,bool> &b) const
			{
				if(a.second > b.second)
					return true;
				else if(a.second < b.second)
					return false;
				else
				{
					return (a.first < b.first);
				}
			}
		};


		void TestProcessPivot(BK_GPU::NeighbourGraph *Ng,BK_GPU::GPU_Stack *stack,Graph *host_graph)
		{
			#ifdef  TEST_ON
			{
				stack->topElement(&currentTopElement);

				int max_index= 0;
				int maxNeighbourinP=-1;

				Psegment = new unsigned[currentTopElement.currPSize];
				Xsegment = new unsigned[currentTopElement.currXSize];

				CudaError(cudaMemcpy(Psegment,Ng->data+currentTopElement.beginP,sizeof(unsigned)*currentTopElement.currPSize,cudaMemcpyDeviceToHost));
				CudaError(cudaMemcpy(Xsegment,Ng->data+currentTopElement.beginX,sizeof(unsigned)*currentTopElement.currXSize,cudaMemcpyDeviceToHost))

				//obtain the pivot element;
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
					}

					if(numNeighbours > maxNeighbourinP)
					{
						maxNeighbourinP = numNeighbours;
						max_index = i;
					}
				}

				unsigned pivot = Psegment[max_index];
				int beginOffset = host_graph->rowOffset[pivot];
				int adjacencySize = host_graph->rowOffset[pivot + 1] - host_graph->rowOffset[pivot];

				//Used to store the pair, P value and a boolean which indicates if its a neighbour of the pivot.
				std::vector<std::pair<unsigned,bool> > auxillaryVec(currentTopElement.currPSize);

				//ReArrange P values with Neighbors towards the left and sorted
				for(int i=0;i<currentTopElement.currPSize;i++)
				{
					bool isNeighbor=std::binary_search(host_graph->columns.begin() + beginOffset,host_graph->columns.begin() + beginOffset + adjacencySize,Psegment[i]);
					auxillaryVec[i]=std::make_pair<int,bool>(Psegment[i],isNeighbor);
				}

				//Sort the Values such that neighbors are towards the left and sorted
				std::sort(auxillaryVec.begin(),auxillaryVec.end(),Compare());

				for(int i=0;i<currentTopElement.currPSize; i++)
					Psegment[i] = auxillaryVec[i].first;

				auxillaryVec.clear();

				//Resize the array to fill currXsize
				auxillaryVec.resize(currentTopElement.currXSize);

				int numNeighboursInX=0;

				for(int i=0;i<currentTopElement.currXSize;i++)
				{
					bool isNeighbor = std::binary_search(host_graph->columns.begin() + beginOffset, host_graph->columns.begin() + beginOffset + adjacencySize,Xsegment[i]);
					auxillaryVec[i]=std::make_pair<int,bool>(Xsegment[i],isNeighbor);

					if(isNeighbor)
						numNeighboursInX++;
				}

				//Sort the Values such that neighbors are towards the left and sorted
				std::sort(auxillaryVec.begin(),auxillaryVec.end(),Compare());

				for(int i=0;i<currentTopElement.currXSize; i++)
					Xsegment[i] = auxillaryVec[i].first;


				auxillaryVec.clear();


				finalElement.beginX = currentTopElement.beginX;
				finalElement.currXSize = numNeighboursInX;
				finalElement.beginP = currentTopElement.beginP;
				finalElement.currPSize = maxNeighbourinP;
				finalElement.beginR = currentTopElement.beginR - 1;
				finalElement.currRSize = currentTopElement.currRSize + 1;

				finalElement.pivot = pivot;
				finalElement.trackerSize = currentTopElement.trackerSize;
				finalElement.direction = true;

				//delete Psegment;
				//delete Xsegment;

			}

			#endif
			return;
		}

		void TestPivotEnd(StackElement &topElement,BK_GPU::NeighbourGraph *Ng,BK_GPU::GPU_Stack *stack,Graph *host_graph)
		{
			finalElement.TestEquality(topElement);

			unsigned *PdeviceSegment = new unsigned[max(1,finalElement.currPSize)];
			unsigned *XdeviceSegment = new unsigned[max(1,finalElement.currXSize)];

			CudaError(cudaMemcpy(PdeviceSegment,Ng->data+finalElement.beginP,sizeof(unsigned)*finalElement.currPSize,cudaMemcpyDeviceToHost));
			CudaError(cudaMemcpy(XdeviceSegment,Ng->data+finalElement.beginX,sizeof(unsigned)*finalElement.currXSize,cudaMemcpyDeviceToHost))

			for(int i=0;i<finalElement.currPSize;i++)
				assert(PdeviceSegment[i]==Psegment[i]);

			for(int i=0;i<finalElement.currXSize;i++)
				assert(XdeviceSegment[i]==Xsegment[i]);

			delete Psegment;
			delete Xsegment;
			delete PdeviceSegment;
			delete XdeviceSegment;

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

		void TestMoveFromXToP(BK_GPU::NeighbourGraph *Ng,BK_GPU::GPU_Stack *stack,Graph *host_graph)
		{
			stack->topElement(&currentTopElement);
			stack->secondElement(&secondElement);

			finalElement.beginX = secondElement.beginX;
			finalElement.currXSize = secondElement.currXSize;
			finalElement.beginP = secondElement.beginP;
			finalElement.currPSize = secondElement.currPSize;
			finalElement.beginR = secondElement.beginR;
			finalElement.currRSize = secondElement.currRSize;

			finalElement.pivot = secondElement.pivot;
			finalElement.trackerSize = secondElement.trackerSize;
			finalElement.direction = true;

		}
};

} /* namespace BK_GPU */
#endif /* BKINSTANCETEST_H_ */
