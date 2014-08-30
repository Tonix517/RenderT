#include "_lightcuts.h"

#include <queue>

//
//	NOTE: ForGpu code is just copied from Cpu. I know it's bad, it is only because 
//		  i'm hurrying for graduation..
//

float sim_metric(DirPointLight *l1, DirPointLight *l2)
{
	float fTotalIllum = vecLen(l1->_diffuseColor) + vecLen(l2->_diffuseColor);

	vect3d vLen;
	points2vec(l1->_pos, l2->_pos, vLen);
	float fDist = vecLen(vLen);

	float constV = 1.f;

	float halfAngle = acosf(dot_product(l1->_dir, l2->_dir) / (vecLen(l1->_dir) * vecLen(l2->_dir))) / 2;

	return fTotalIllum * (fDist * fDist + constV * constV * (1 - powf(cosf(halfAngle), 2)));
}


float sim_metric_gpu(LightCpu &l1, LightCpu &l2)
{
	float fTotalIllum = vecLen(l1._diffuseColor) + vecLen(l2._diffuseColor);

	vect3d vLen;
	points2vec(l1._dirp_pos, l2._dirp_pos, vLen);
	float fDist = vecLen(vLen);

	float constV = 1.f;

	float halfAngle = acosf(dot_product(l1._dirp_dir, l2._dirp_dir) / (vecLen(l1._dirp_dir) * vecLen(l2._dirp_dir))) / 2;

	return fTotalIllum * (fDist * fDist + constV * constV * (1 - powf(cosf(halfAngle), 2)));
}


short getBestPairLightInx(short inx, bool *marks, std::vector<Light*> &lights, unsigned nPointLightCount)
{
	short ret = -1;
	
	//	NOTICE: this is a naive linear algo. I didn't use BSP for VPL. I don't have enough time 
	//			for this, sorry. 
	//
	float fCurrMetric = 0xFFFFFFFF;
	for(int i = 0; i < nPointLightCount; i ++)
	{
		if( (marks[i] == false) && (inx != i))
		{
			float fMetric = sim_metric((DirPointLight *)lights[inx], (DirPointLight *)lights[i]);
			if(fMetric < fCurrMetric)
			{
				fCurrMetric = fMetric;
				ret = i;
			}
		}
	}

	return ret;
}


short getBestPairLightInxForGpu(short inx, bool *marks, std::vector<LightCpu> &lights, unsigned nPointLightCount)
{
	short ret = -1;
	
	//	NOTICE: this is a naive linear algo. I didn't use BSP for VPL. I don't have enough time 
	//			for this, sorry. 
	//
	float fCurrMetric = 0xFFFFFFFF;
	for(int i = 0; i < nPointLightCount; i ++)
	{
		if( (marks[i] == false) && (inx != i))
		{
			float fMetric = sim_metric_gpu(lights[inx], lights[i]);
			if(fMetric < fCurrMetric)
			{
				fCurrMetric = fMetric;
				ret = i;
			}
		}
	}

	return ret;
}

unsigned getBinTreeNodeCount(unsigned nLeafCount)
{
	assert(nLeafCount > 1);

	unsigned ret = 0;

	//	Make the bottom layer EVEN num. of lights
	nLeafCount += ((nLeafCount % 2 == 1) && (nLeafCount > 1)) ? 1 : 0;

	while(nLeafCount >= 2)
	{
		ret += nLeafCount;	// add from the bottom layer
		nLeafCount /= 2;
		nLeafCount += ((nLeafCount % 2 == 1) && (nLeafCount > 1)) ? 1 : 0;
	}

	ret += 1;	// the root node

	return ret;
}

unsigned chooseOneLightByInx(unsigned inx1, unsigned inx2, lt_node *nodes, std::vector<Light*> &lights)
{
	DirPointLight *p1 = dynamic_cast<DirPointLight *>(lights[nodes[inx1].inx_in_light_array]);
	DirPointLight *p2 = dynamic_cast<DirPointLight *>(lights[nodes[inx2].inx_in_light_array]);
		assert(p1 && p2);

	float fIllum1 = vecLen(p1->_diffuseColor);
	float fIllum2 = vecLen(p1->_diffuseColor);

	float fProb1 = fIllum1 / (fIllum1 + fIllum2);
	float fRnm = rand() % 100 / 100.f;

	return (fRnm < fProb1) ? inx1 : inx2;
}


unsigned chooseOneLightByInxForGpu(unsigned inx1, unsigned inx2, lt_node *nodes, std::vector<LightCpu> &lights)
{
	LightCpu &p1 = lights[nodes[inx1].inx_in_light_array];
	LightCpu &p2 = lights[nodes[inx2].inx_in_light_array];

	float fIllum1 = vecLen(p1._diffuseColor);
	float fIllum2 = vecLen(p1._diffuseColor);

	float fProb1 = fIllum1 / (fIllum1 + fIllum2);
	float fRnm = rand() % 100 / 100.f;

	return (fRnm < fProb1) ? inx1 : inx2;
}

std::queue<lt_node *> node_queue;
void outputTree(std::queue<lt_node *> &theQueue, lt_node *pRoot)
{
	size_t nOutputCount = 0;

	while(!theQueue.empty())
	{
		size_t nCurrCount = theQueue.size();

		for(size_t i = 0; i < nCurrCount; i ++)
		{
			if( (nOutputCount % 2 == 1) && (nOutputCount > 0) ) printf("{");
			lt_node *pNode = theQueue.front();	theQueue.pop();
			printf("[%d] ", pNode->inx_in_light_array);
			if( (nOutputCount % 2 == 0) && (nOutputCount > 0) ) printf("\b}");

			nOutputCount ++;

			if(pNode->l_child_inx_in_tree != -1)
			{
				theQueue.push(pRoot + pNode->l_child_inx_in_tree);
				theQueue.push(pRoot + pNode->r_child_inx_in_tree);
			}
		}
		printf("\n");		
	}
}

float estimateRelevantFactor(vect3d &vHitPoint, Light *pLight)
{
	vect3d vl2hit;
	vect3d lPos; pLight->getPos(lPos);
	points2vec(lPos, vHitPoint, vl2hit);

	//	1. Distance
	float fDist = vecLen(vl2hit);

	//	2. Illumination
	float fIllum = vecLen(pLight->_diffuseColor);

	//	3. Oriental cos
	vect3d vldir;	pLight->getDir(vldir);
	float cosTheta = dot_product(vl2hit, vldir) / (fDist * vecLen(vldir));

	return (cosTheta > 0 ? cosTheta : 0) * fIllum / (fDist * fDist);
}

////////////////////////////////////////////////

lt_node *pCPULCTree = NULL;

lt_node* buildLightTree(std::vector<Light*> &pointLights, unsigned &nRetNodeCount)
{
	//	Keep the original lights array. and the tree will 
	//	take int as the node, for the sake of space

	unsigned nPointLightCount = pointLights.size();
	unsigned nTotalNodeInTree = getBinTreeNodeCount(nPointLightCount);

	pCPULCTree = new lt_node[nTotalNodeInTree];
	assert(pCPULCTree);

	//	1. Fill Leaves
	//
	bool *aLeafTaken = new bool[nPointLightCount];
	for(int i = 0; i<nPointLightCount; i ++) aLeafTaken[i] = false;

	unsigned nLeafCount = nPointLightCount + 
						(((nPointLightCount % 2 == 1) && (nPointLightCount > 1)) ? 1 : 0);

	unsigned nTakenCount = 0;

	while(nTakenCount < nPointLightCount)
	{
		//	Get 1st un-taken light
		short nCurrInx = -1;
		for(int i = 0; i < nPointLightCount; i ++)
		{	if(!aLeafTaken[i]){	nCurrInx = i;	break; }	}

		//	Get the best match 
		short nMatchInx = getBestPairLightInx(nCurrInx, aLeafTaken, pointLights, nPointLightCount);

		if(nMatchInx != -1)	// found
		{
			//	put the two into the tree
			pCPULCTree[nTotalNodeInTree - nLeafCount + nTakenCount].setValue(nCurrInx, -1, -1);
			pCPULCTree[nTotalNodeInTree - nLeafCount + nTakenCount + 1].setValue(nMatchInx, -1, -1);

			aLeafTaken[nCurrInx] = aLeafTaken[nMatchInx] = true;
			nTakenCount += 2;

			if(nTakenCount == nPointLightCount)
			{
				assert( (nTotalNodeInTree - nLeafCount + nTakenCount - 1) == (nTotalNodeInTree - 1) );	// Odd number VPL
			}
		}
		else	// it is the last one..
		{
			//	put the same two into the tree
			pCPULCTree[nTotalNodeInTree - nLeafCount + nTakenCount].setValue(nCurrInx, -1, -1);
				assert( (nTotalNodeInTree - nLeafCount + nTakenCount + 1) == (nTotalNodeInTree - 1) );	// Odd number VPL
			pCPULCTree[nTotalNodeInTree - nLeafCount + nTakenCount + 1].setValue(nCurrInx, -1, -1);

			aLeafTaken[nCurrInx] = true;
			nTakenCount += 1;

			assert((nTakenCount + 1) == nLeafCount);

			break;
		}
	}
	
	assert(nTakenCount == nPointLightCount);

	delete [] aLeafTaken;

	
	//	2. Build up Nodes
	//
	unsigned nLastLayerStartInx = nTotalNodeInTree - nLeafCount;
	unsigned nLastLayerCount = nLeafCount;
	unsigned nCurrLayerNodeCount = nLeafCount / 2;	assert(nLeafCount % nCurrLayerNodeCount == 0);
	nCurrLayerNodeCount += ((nCurrLayerNodeCount % 2 == 1) && (nCurrLayerNodeCount > 1)) ? 1 : 0;
	unsigned nCurrLayerStartInx = nLastLayerStartInx - nCurrLayerNodeCount;

#define DBG_OUTPUT 0

#if DBG_OUTPUT
	printf("[Bottom Layer] - Start Inx : %d, Count : %d\n", nLastLayerStartInx, nLastLayerCount);
#endif

	unsigned nLayerCount = 1;

	while(nCurrLayerNodeCount >= 1)
	{
		//
		for(int i = 0; i <  nLastLayerCount / 2; i ++)
		{
			unsigned inx1 = nLastLayerStartInx + i * 2;
			unsigned inx2 = nLastLayerStartInx + i * 2 + 1;
			unsigned iChosen = chooseOneLightByInx(inx1, inx2, pCPULCTree, pointLights);

			pCPULCTree[nCurrLayerStartInx + i].setValue(pCPULCTree[iChosen].inx_in_light_array, inx1, inx2);
#if DBG_OUTPUT
			printf("[%d] -> {%d, %d} by %d\n", nCurrLayerStartInx + i, inx1, inx2, iChosen);
#endif
		}
		if( (nLastLayerCount / 2) < nCurrLayerNodeCount)
		{
			pCPULCTree[nCurrLayerStartInx + nCurrLayerNodeCount - 1].setValue(
					pCPULCTree[nCurrLayerStartInx + nCurrLayerNodeCount - 2].inx_in_light_array, -1, -1 );
#if DBG_OUTPUT
			printf("[%d] -> {%d, %d} by %d\n", nCurrLayerStartInx + nCurrLayerNodeCount - 1, 
				   -1, -1, nCurrLayerStartInx + nCurrLayerNodeCount - 2);
#endif
		}

		//	Move on to upper layer
		//
		nLastLayerStartInx = nLastLayerStartInx - nCurrLayerNodeCount;
		nLastLayerCount = nCurrLayerNodeCount;
		nCurrLayerNodeCount /= 2;
		nCurrLayerNodeCount += ((nCurrLayerNodeCount % 2 == 1) && (nCurrLayerNodeCount > 1)) ? 1 : 0;
		nCurrLayerStartInx = nLastLayerStartInx - nCurrLayerNodeCount;
#if DBG_OUTPUT
		printf("[Layer %d] - Start Inx : %d, Count : %d\n", nLayerCount ++, nLastLayerStartInx, nLastLayerCount);
#endif
	}

	//	Output Tree
#if 0
	node_queue.push(pCPULCTree);
	outputTree(node_queue, pCPULCTree);
#endif

	nRetNodeCount = nTotalNodeInTree;

	return pCPULCTree;
}

lt_node * buildLightTreeForGpu(std::vector<LightCpu> &pointLights, unsigned &nRetNodeCount)
{
		//	Keep the original lights array. and the tree will 
	//	take int as the node, for the sake of space

	unsigned nPointLightCount = pointLights.size();
	unsigned nTotalNodeInTree = getBinTreeNodeCount(nPointLightCount);

	pCPULCTree = new lt_node[nTotalNodeInTree];
	assert(pCPULCTree);

	//	1. Fill Leaves
	//
	bool *aLeafTaken = new bool[nPointLightCount];
	for(int i = 0; i<nPointLightCount; i ++) aLeafTaken[i] = false;

	unsigned nLeafCount = nPointLightCount + 
						(((nPointLightCount % 2 == 1) && (nPointLightCount > 1)) ? 1 : 0);

	unsigned nTakenCount = 0;

	while(nTakenCount < nPointLightCount)
	{
		//	Get 1st un-taken light
		short nCurrInx = -1;
		for(int i = 0; i < nPointLightCount; i ++)
		{	if(!aLeafTaken[i]){	nCurrInx = i;	break; }	}

		//	Get the best match 
		short nMatchInx = getBestPairLightInxForGpu(nCurrInx, aLeafTaken, pointLights, nPointLightCount);

		if(nMatchInx != -1)	// found
		{
			//	put the two into the tree
			pCPULCTree[nTotalNodeInTree - nLeafCount + nTakenCount].setValue(nCurrInx, -1, -1);
			pCPULCTree[nTotalNodeInTree - nLeafCount + nTakenCount + 1].setValue(nMatchInx, -1, -1);

			aLeafTaken[nCurrInx] = aLeafTaken[nMatchInx] = true;
			nTakenCount += 2;

			if(nTakenCount == nPointLightCount)
			{
				assert( (nTotalNodeInTree - nLeafCount + nTakenCount - 1) == (nTotalNodeInTree - 1) );	// Odd number VPL
			}
		}
		else	// it is the last one..
		{
			//	put the same two into the tree
			pCPULCTree[nTotalNodeInTree - nLeafCount + nTakenCount].setValue(nCurrInx, -1, -1);
				assert( (nTotalNodeInTree - nLeafCount + nTakenCount + 1) == (nTotalNodeInTree - 1) );	// Odd number VPL
			pCPULCTree[nTotalNodeInTree - nLeafCount + nTakenCount + 1].setValue(nCurrInx, -1, -1);

			aLeafTaken[nCurrInx] = true;
			nTakenCount += 1;

			assert((nTakenCount + 1) == nLeafCount);

			break;
		}
	}
	
	assert(nTakenCount == nPointLightCount);

	delete [] aLeafTaken;

	
	//	2. Build up Nodes
	//
	unsigned nLastLayerStartInx = nTotalNodeInTree - nLeafCount;
	unsigned nLastLayerCount = nLeafCount;
	unsigned nCurrLayerNodeCount = nLeafCount / 2;	assert(nLeafCount % nCurrLayerNodeCount == 0);
	nCurrLayerNodeCount += ((nCurrLayerNodeCount % 2 == 1) && (nCurrLayerNodeCount > 1)) ? 1 : 0;
	unsigned nCurrLayerStartInx = nLastLayerStartInx - nCurrLayerNodeCount;

#define DBG_OUTPUT 0

#if DBG_OUTPUT
	printf("[Bottom Layer] - Start Inx : %d, Count : %d\n", nLastLayerStartInx, nLastLayerCount);
#endif

	unsigned nLayerCount = 1;

	while(nCurrLayerNodeCount >= 1)
	{
		//
		for(int i = 0; i <  nLastLayerCount / 2; i ++)
		{
			unsigned inx1 = nLastLayerStartInx + i * 2;
			unsigned inx2 = nLastLayerStartInx + i * 2 + 1;
			unsigned iChosen = chooseOneLightByInxForGpu(inx1, inx2, pCPULCTree, pointLights);

			pCPULCTree[nCurrLayerStartInx + i].setValue(pCPULCTree[iChosen].inx_in_light_array, inx1, inx2);
#if DBG_OUTPUT
			printf("[%d] -> {%d, %d} by %d\n", nCurrLayerStartInx + i, inx1, inx2, iChosen);
#endif
		}
		if( (nLastLayerCount / 2) < nCurrLayerNodeCount)
		{
			pCPULCTree[nCurrLayerStartInx + nCurrLayerNodeCount - 1].setValue(
					pCPULCTree[nCurrLayerStartInx + nCurrLayerNodeCount - 2].inx_in_light_array, -1, -1 );
#if DBG_OUTPUT
			printf("[%d] -> {%d, %d} by %d\n", nCurrLayerStartInx + nCurrLayerNodeCount - 1, 
				   -1, -1, nCurrLayerStartInx + nCurrLayerNodeCount - 2);
#endif
		}

		//	Move on to upper layer
		//
		nLastLayerStartInx = nLastLayerStartInx - nCurrLayerNodeCount;
		nLastLayerCount = nCurrLayerNodeCount;
		nCurrLayerNodeCount /= 2;
		nCurrLayerNodeCount += ((nCurrLayerNodeCount % 2 == 1) && (nCurrLayerNodeCount > 1)) ? 1 : 0;
		nCurrLayerStartInx = nLastLayerStartInx - nCurrLayerNodeCount;
#if DBG_OUTPUT
		printf("[Layer %d] - Start Inx : %d, Count : %d\n", nLayerCount ++, nLastLayerStartInx, nLastLayerCount);
#endif
	}

	//	Output Tree
#if 0
	node_queue.push(pCPULCTree);
	outputTree(node_queue, pCPULCTree);
#endif

	nRetNodeCount = nTotalNodeInTree;

	return pCPULCTree;
}

void deleteLightTree()
{
	if(pCPULCTree)
	{
		delete [] pCPULCTree;
		pCPULCTree = NULL;
	}
}