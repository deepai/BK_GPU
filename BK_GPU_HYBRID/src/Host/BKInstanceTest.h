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

		std::vector<unsigned> Psegment;
		std::vector<unsigned> Xsegment;

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

				//Allocate space for PSegment and Xsegment
				Psegment.resize(currentTopElement.currPSize);
				Xsegment.resize(currentTopElement.currXSize);

				//copy into the previous arrays the value from the cuda device
				CudaError(cudaMemcpy(Psegment.data(),Ng->data+currentTopElement.beginP,sizeof(unsigned)*currentTopElement.currPSize,cudaMemcpyDeviceToHost));
				CudaError(cudaMemcpy(Xsegment.data(),Ng->data+currentTopElement.beginX,sizeof(unsigned)*currentTopElement.currXSize,cudaMemcpyDeviceToHost))

				//obtain the pivot element
				//Loop throught the elements and obtain the number of its neighbours in the array
				for(int i=0;i<currentTopElement.currPSize;i++)
				{
					unsigned currNode = Psegment[i]; //Get Current Node ID
					int numNeighbours = 0; //Initialise number of neighbours
					for(int j=0;j<currentTopElement.currPSize;j++)
					{
						if(j==i) //skip if the index represent the same node as the current node
							continue;
						else
						{
							//obtain the adjacency size
							int adjacencySize = host_graph->rowOffset[currNode + 1] - host_graph->rowOffset[currNode];

							//beginoffset of the adjacency list
							int beginOffset =  host_graph->rowOffset[currNode];

							//binary search returns whether the node j exist in the adjancency list of node i. IF yes, Increment the neighbour count
							if(std::binary_search(host_graph->columns.begin() + beginOffset, host_graph->columns.begin() + beginOffset + adjacencySize,Psegment[j]))
								numNeighbours++;
						}
					}
					//if number of neighbours is greater than maxneighbour till now,update the maxneighbour
					if(numNeighbours > maxNeighbourinP)
					{
						maxNeighbourinP = numNeighbours;
						max_index = i;
					}
				}

				//pivot element
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

				//Copy the elements back into Psegment
				for(int i=0;i<currentTopElement.currPSize; i++)
					Psegment[i] = auxillaryVec[i].first;

				//Clear the vector
				auxillaryVec.clear();

				//Resize the array to fill currXsize
				auxillaryVec.resize(currentTopElement.currXSize);

				int numNeighboursInX=0;

				//Search for the neighbours of the Xsegment
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
			#ifdef TEST_ON
			{
				//Test whether the configuration parameters matches
				finalElement.TestEquality(topElement);

				//Make two auxillary array to store the results from the device
				std::vector<unsigned> PdeviceSegment(finalElement.currPSize);
				std::vector<unsigned> XdeviceSegment(finalElement.currXSize);

				//Copy into memory from the device
				CudaError(cudaMemcpy(PdeviceSegment.data(),Ng->data+finalElement.beginP,sizeof(unsigned)*finalElement.currPSize,cudaMemcpyDeviceToHost));
				CudaError(cudaMemcpy(XdeviceSegment.data(),Ng->data+finalElement.beginX,sizeof(unsigned)*finalElement.currXSize,cudaMemcpyDeviceToHost))

				//assert that the values in the Psegment and the values in the Xsegment are the same.
				for(int i=0;i<finalElement.currPSize;i++)
					assert(PdeviceSegment[i]==Psegment[i]);

				for(int i=0;i<finalElement.currXSize;i++)
					assert(XdeviceSegment[i]==Xsegment[i]);


				//delete the arrays.

				Psegment.clear();
				Xsegment.clear();
				PdeviceSegment.clear();
				XdeviceSegment.clear();

			}
			#endif

		}

		void TestMoveToX(BK_GPU::NeighbourGraph *Ng,BK_GPU::GPU_Stack *stack,Graph *host_graph,int pivot)
		{
			#ifdef TEST_ON
			{
				//Check top configuration parameters
				stack->topElement(&currentTopElement);

				//Make two auxillary array to store the results from the device
				Psegment.resize(finalElement.currPSize);
				Xsegment.resize(finalElement.currXSize + 1);

				//Copy from device to host.
				CudaError(cudaMemcpy(Psegment.data(),Ng->data + currentTopElement.beginP,sizeof(unsigned)*currentTopElement.currPSize,cudaMemcpyDeviceToHost));
				CudaError(cudaMemcpy(Xsegment.data(),Ng->data + currentTopElement.beginX,sizeof(unsigned)*currentTopElement.currXSize,cudaMemcpyDeviceToHost));

				//Top of the stack
				Xsegment[currentTopElement.beginX + currentTopElement.currXSize] = currentTopElement.pivot;

				std::sort(Xsegment.begin(),Xsegment.end());

				//Set finalelement values

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
			#endif


		}


		void TestMoveToXEnd(StackElement &topElement,BK_GPU::NeighbourGraph *Ng,BK_GPU::GPU_Stack *stack,Graph *host_graph)
		{
			#ifdef TEST_ON
			{
				finalElement.TestEquality(topElement);

				std::vector<unsigned> PdeviceSegment(finalElement.currPSize);
				std::vector<unsigned> XdeviceSegment(finalElement.currXSize);

				for(int i=0;i<topElement.currPSize;i++)
					assert(PdeviceSegment[i] == Psegment[i]);

				for(int i=0;i<topElement.currXSize;i++)
					assert(XdeviceSegment[i] == Xsegment[i]);

				Psegment.clear();
				Xsegment.clear();
				PdeviceSegment.clear();
				XdeviceSegment.clear();

			}
			#endif
		}

		void TestMoveFromXToP(BK_GPU::NeighbourGraph *Ng,BK_GPU::GPU_Stack *stack,Graph *host_graph)
		{
			#ifdef TEST_ON
			{
				//Load the topElement and Second Element
				stack->topElement(&currentTopElement);
				stack->secondElement(&secondElement);

				//resize the arrays
				Psegment.resize(currentTopElement.currPSize + secondElement.trackerSize - currentTopElement.trackerSize);
				Xsegment.resize(currentTopElement.currXSize);

				//Copy from device to host.
				CudaError(cudaMemcpy(Psegment.data(),Ng->data + currentTopElement.beginP,sizeof(unsigned)*currentTopElement.currPSize,cudaMemcpyDeviceToHost));
				CudaError(cudaMemcpy(Xsegment.data(),Ng->data + currentTopElement.beginX,sizeof(unsigned)*currentTopElement.currXSize,cudaMemcpyDeviceToHost));




				//Set the finalElement values
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
			#endif
		}
};

} /* namespace BK_GPU */
#endif /* BKINSTANCETEST_H_ */
