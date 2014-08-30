#include "gpu_util.h"

#include "global.h"
#include "consts.h"
#include "kd-tree.h"
#include "gpu/kd_tree_gpu.h"
#include "gpu/texture_gpu.cu"

#include <vector>
#include <cuda_runtime.h>


////////////

unsigned nCurrObjCount = 0;
PrimGpuObj *gpuObjs = NULL;
PrimGpuObj_host *hostObjs = NULL;

unsigned nMaxGpuKdDepth = 0;
kd_node_gpu *hostKdRoot = NULL;
kd_node_gpu *deviceKdRoot = NULL;
unsigned *hostPrimObjIdList = NULL;
unsigned *devicePrimObjIdList = NULL;
bool *deviceKdRecBuf = NULL;
unsigned nMaxKdNodeCount = 0;

unsigned nCurrLightCount = 0;
LightGpu *gpuLights = NULL;
LightCpu *cpuLights = NULL;

//	Instant Radiosity
LightGpu *gpuDiffLightsPerLine = NULL;
LightCpu *cpuDiffLightsPerLine = NULL;
float *gpuRdm = NULL;
float *cpuRdm = NULL;

__device__ bool bGpuHasDiffLights = false;

////////////

void gpu_destroy()
{
	
	if(gpuObjs)
	{
		cudaFree(gpuObjs);
		gpuObjs = NULL;
	}
	if(hostObjs)
	{
		free(hostObjs);
		hostObjs = NULL;
	}

	if(hostKdRoot)
	{
		free(hostKdRoot);
		hostKdRoot = NULL;
	}

	if(deviceKdRoot)
	{
		cudaFree(deviceKdRoot);
		deviceKdRoot = NULL;
	}

	if(hostPrimObjIdList)
	{
		free(hostPrimObjIdList);
		hostPrimObjIdList = NULL;
	}

	if(devicePrimObjIdList)
	{
		cudaFree(devicePrimObjIdList);
		devicePrimObjIdList = NULL;
	}

	if(deviceKdRecBuf)
	{
		cudaFree(deviceKdRecBuf);
		deviceKdRecBuf = NULL;
	}

	if(gpuLights)
	{
		cudaFree(gpuLights);
		gpuLights = NULL;
	}
	if(cpuLights)
	{
		free(cpuLights);
		cpuLights = NULL;
	}

	//
	if(gpuDiffLightsPerLine)
	{
		cudaFree(gpuDiffLightsPerLine);
		gpuDiffLightsPerLine = NULL;
	}

	if(cpuDiffLightsPerLine)
	{
		free(cpuDiffLightsPerLine);
		cpuDiffLightsPerLine = NULL;
	}

	if(gpuRdm)
	{
		cudaFree(gpuRdm);
		gpuRdm = NULL;
	}

	if(cpuRdm)
	{
		free(cpuRdm);
		cpuRdm = NULL;
	}

	unloadGpuTexture();

}

//////////////////

__constant__ int MaxRayDepth_gpu;
__constant__ float AmbiColor_gpu[3];
__constant__ float epsi_gpu;
  
__constant__ float fVPLPossibility_gpu;
__constant__ float fVPLIllmThreshold_gpu;
__constant__ float fVPLAtten_gpu;

void sendConstants2GPU()
{
	cudaError_t err = cudaMemcpyToSymbol(MaxRayDepth_gpu, &MaxRayDepth, sizeof(int)/*, 0, cudaMemcpyHostToDevice*/);

	vect3d ambiColor; scene.getAmbiColor(ambiColor);
	err = cudaMemcpyToSymbol(AmbiColor_gpu, ambiColor.data, sizeof(float) * 3/*, 0, cudaMemcpyHostToDevice*/);
	err = cudaMemcpyToSymbol(epsi_gpu, &epsi, sizeof(float) /*, 0,	cudaMemcpyHostToDevice*/);

	//

	err = cudaMemcpyToSymbol(fVPLPossibility_gpu, &fVPLPossibility, sizeof(float) /*, 0,	cudaMemcpyHostToDevice*/);
	err = cudaMemcpyToSymbol(fVPLIllmThreshold_gpu, &fVPLIllmThreshold, sizeof(float) /*, 0,	cudaMemcpyHostToDevice*/);
	err = cudaMemcpyToSymbol(fVPLAtten_gpu, &fVPLAtten, sizeof(float) /*, 0,	cudaMemcpyHostToDevice*/);
	
	cudaThreadSynchronize();
}

unsigned getKdTreeDepth(kd_node *pNode, unsigned nCurrLayer = 0)
{
	if(pNode->child0 == NULL && pNode->child1 == NULL)
	{
		return nCurrLayer;
	}

	unsigned d1 = getKdTreeDepth(pNode->child0, nCurrLayer + 1);
	unsigned d2 = getKdTreeDepth(pNode->child1, nCurrLayer + 1);

	return (d1 > d2) ? d1 : d2;
}

unsigned getNodeObjCount(kd_node *pKdNode)
{
	unsigned nCurrCount = pKdNode->objects.size();

	unsigned n0 = 0, n1 = 0;

	if(pKdNode->child0)
	{
		n0 = getNodeObjCount(pKdNode->child0);
	}
	if(pKdNode->child1)
	{
		n1 = getNodeObjCount(pKdNode->child1);
	}

	return (nCurrCount + n0 + n1);
}

unsigned buildKdTreeForGPUObjs(kd_node *pKdNode, std::vector<Object*> &vObjs, unsigned *nIdCount)
{
	for(int i = 0; i < vObjs.size(); i ++)
	{
		pKdNode->addObject(vObjs[i]);
	}
	
	pKdNode->updateBBox();

	{
		//	Combine the two factors this way...
		kd_node::nSceneDepth = (nSceneDepth == -1 || nObjDepth == -1) ? 
							   (8 + 1.3 * log(vObjs.size() * 1.f)) : (nSceneDepth + nObjDepth);
		printf("[Building KD-tree for Primary Object on GPU of Depth %d...\n", kd_node::nSceneDepth);

		pKdNode->split();
	}

	*nIdCount = getNodeObjCount(pKdNode);

	return getKdTreeDepth(pKdNode);
}

static unsigned nCurrInxInx = 0;
void serialize_kd_tree(kd_node *pNode, kd_node_gpu *hostKdRoot, unsigned *hostPrimObjIdList, unsigned nodeId = 0)
{
	if(pNode == NULL)	return;

	kd_node_gpu *pNodeGpu = hostKdRoot + nodeId;

	pNodeGpu->_nDepth = pNode->_nDepth;

	pNodeGpu->_bbox._xmin = pNode->_bbox._xmin;
	pNodeGpu->_bbox._ymin = pNode->_bbox._ymin;
	pNodeGpu->_bbox._zmin = pNode->_bbox._zmin;
	pNodeGpu->_bbox._xmax = pNode->_bbox._xmax;
	pNodeGpu->_bbox._ymax = pNode->_bbox._ymax;
	pNodeGpu->_bbox._zmax = pNode->_bbox._zmax;

	pNodeGpu->eAxis = pNode->eAxis;
	pNodeGpu->_delim = pNode->_delim;

	//	assign objects
	pNodeGpu->nInxStartInx = nCurrInxInx;
	pNodeGpu->nInxCount = pNode->objects.size();
	if(pNode->objects.size() > 0)
	{
		//printf("{(%d) ", nodeId);

		for(int i = 0; i < pNodeGpu->nInxCount; i ++)
		{
			//printf("%d  ", pNode->objects[i]->_id);
			hostPrimObjIdList[nCurrInxInx] = pNode->objects[i]->_id;
			nCurrInxInx ++;
		}
		//printf(" (%d)} ", pNodeGpu->nInxCount);
	}

	//	Recursive
	//
	kd_node *pLeft = pNode->child0;
	if(pLeft)
	{
		unsigned lid = (nodeId + 1) * 2 - 1;
		pNodeGpu->child0Inx = lid;
		serialize_kd_tree(pLeft, hostKdRoot, hostPrimObjIdList, lid);
	}

	kd_node *pRight = pNode->child1;
	if(pRight)
	{
		unsigned rid = (nodeId + 1) * 2;
		pNodeGpu->child1Inx = rid;
		serialize_kd_tree(pRight, hostKdRoot, hostPrimObjIdList, rid);
	}
}

void copySceneGeomotry()
{
	///
	///		NOTE: All kinds of geometry in CPU side. Since CUDA doesn't support
	///			  C++ features as polymorphism, they have to be handled separately.
	///

	std::vector<Object*> v4kdtree;

	//	Count all primary objs
	//
	unsigned nTotalPrimaryCount = 0;
	for(int i = 0; i < scene.getObjectNum(); i ++)
	{
		Object *pObj = scene.getObject(i);
		ObjType eType = pObj->getObjType();

		if(eType != OBJ_CPU)
		{
			if(eType != CUBE_CPU)
			{
				nTotalPrimaryCount ++;
			}
			else
			{
				nTotalPrimaryCount += 6;
			}
		}
		else
		{
			ObjObject *pObjObj = dynamic_cast<ObjObject*>(pObj);	// I know...
			nTotalPrimaryCount += pObjObj->getTriCount();
		}		
	}

	v4kdtree.reserve(nTotalPrimaryCount);

	//	1. Re-Alloc space for Objects
	if(gpuObjs)
	{
		cudaFree(gpuObjs);
	}
	cudaError_t err = cudaMalloc((void**)&gpuObjs, sizeof(PrimGpuObj) * nTotalPrimaryCount);
	if(err != cudaSuccess)
	{
		printf("CUDA error %s, %s, %s \n", __FILE__, __LINE__, cudaGetErrorString(err));
	}
		
	if(hostObjs)
	{
		free(hostObjs);
	}
	hostObjs = (PrimGpuObj_host*)malloc(sizeof(PrimGpuObj_host) * nTotalPrimaryCount);
	if(!hostObjs)
	{
		printf("malloc failed %s, %s \n", __FILE__, __LINE__);
	}

	//	2. Copy Objects
	unsigned nCurrPrimObjInx = 0;
	for(int i = 0; i < scene.getObjectNum(); i ++)
	{
		Object *pObj = scene.getObject(i);
		ObjType eType = pObj->getObjType();

		if(eType != OBJ_CPU)
		{
			PrimGpuObj_host *pCurrPrimGpuObj = &hostObjs[nCurrPrimObjInx];

			//	Copy the common part
			pCurrPrimGpuObj->nId = pObj->_id;

			pCurrPrimGpuObj->_fReflectionRatio = pObj->getReflectionRatio();
			pCurrPrimGpuObj->_fRefractionRatio = pObj->getRefractionRatio();
			pCurrPrimGpuObj->_fRefractionK = pObj->getRefractionK();
			pCurrPrimGpuObj->_fEmitRatio = pObj->getEmissionRatio();
			pCurrPrimGpuObj->_mat.fShininess = pObj->_mat.fShininess;	

			vecCopy(pCurrPrimGpuObj->_mat.specColor, pObj->_mat.specColor);
			vecCopy(pCurrPrimGpuObj->_mat.diffColor, pObj->_mat.diffColor);
			vecCopy(pCurrPrimGpuObj->_mat.ambiColor, pObj->_mat.ambiColor);

			switch(eType)
			{
			case SPH_CPU:
				{
					Sphere *pSph = dynamic_cast<Sphere*>(pObj);
					copySphere(pCurrPrimGpuObj, pSph);
					
					v4kdtree.push_back(pSph);

					nCurrPrimObjInx ++;
				}
				break;

			case SQU_CPU:
				{
					Square *pSqu = dynamic_cast<Square*>(pObj);
					copySquare(pCurrPrimGpuObj, pSqu);

					v4kdtree.push_back(pSqu);

					nCurrPrimObjInx ++;
				}
				break;

			case CUBE_CPU:
				{
					Cube *pCube = dynamic_cast<Cube*>(pObj);
					for(int m = 0; m < 6; m ++)
					{
						pCurrPrimGpuObj = &hostObjs[nCurrPrimObjInx];
						copySquare(pCurrPrimGpuObj, pCube->_vs[m]);	
						
						v4kdtree.push_back(pCube->_vs[m]);

						nCurrPrimObjInx ++;
					}
				}
				break;

			default:
				printf("not supported obj type \n");
				return;
				break;
			}
		}
		else
		{
			ObjObject *pObjObj = dynamic_cast<ObjObject*>(pObj);	// I know...
			unsigned nCurrTriNum = pObjObj->getTriCount();
			for(int j = 0; j < nCurrTriNum; j ++)
			{
				Triangle *pCurrTri = pObjObj->getTriangle(j);
				PrimGpuObj_host *pCurrPrimGpuObj = &hostObjs[nCurrPrimObjInx];

				copyTriangle(pCurrPrimGpuObj, pObjObj, pCurrTri);

				v4kdtree.push_back(pCurrTri);

				nCurrPrimObjInx ++;
			}//	for
		}//	else		
	}//	copy for

	///
	///		Build KD-Tree for GPU only
	///
	kd_node *pNode = new kd_node;
	unsigned nTotalIdCount = 0;
	unsigned nMaxDepth = buildKdTreeForGPUObjs(pNode, v4kdtree, &nTotalIdCount);
	nMaxGpuKdDepth = nMaxDepth + 1;

	//	Get ready for the KD-Tree on GPU
	//
	unsigned nKdTreeNodeCount = pow(2.f, (nMaxDepth + 2) * 1.f) - 1; 
	nMaxKdNodeCount = nKdTreeNodeCount;

	//	Kd-tree recursion record buf.
	if(deviceKdRecBuf)
	{
		cudaFree(deviceKdRecBuf);
	}
	err = cudaMalloc(&deviceKdRecBuf, sizeof(bool) * nKdTreeNodeCount * WinWidth);
	if(err != cudaSuccess)
	{
		printf("CUDA error %s, %s, %s \n", __FILE__, __LINE__, cudaGetErrorString(err));
	}

	//	Re-Alloc Kd-tree sapce
	if(hostKdRoot)
	{
		free(hostKdRoot);
		hostKdRoot = NULL;
	}
	hostKdRoot = (kd_node_gpu *)malloc(sizeof(kd_node_gpu) * nKdTreeNodeCount );
	if(!hostKdRoot)
	{
		printf("malloc failed %s, %s \n", __FILE__, __LINE__);
	}

	memset(hostKdRoot, 0 ,sizeof(kd_node_gpu) * nKdTreeNodeCount);
	for(int i = 0; i < nKdTreeNodeCount; i ++)
	{
		(hostKdRoot + i)->child0Inx = -1; 
		(hostKdRoot + i)->child1Inx = -1; 
		(hostKdRoot + i)->nInxCount =  0; 
	}

	if(deviceKdRoot)
	{
		cudaFree(deviceKdRoot);
		deviceKdRoot = NULL;
	}
	err = cudaMalloc(&deviceKdRoot, sizeof(kd_node_gpu) * nKdTreeNodeCount);
	if(err != cudaSuccess)
	{
		printf("CUDA error %s, %s, %s \n", __FILE__, __LINE__, cudaGetErrorString(err));
	}

	//	Id list buf on GPU
	if(hostPrimObjIdList)
	{
		free(hostPrimObjIdList);
	}
	hostPrimObjIdList = (unsigned *)malloc(sizeof(unsigned) * nTotalIdCount);//	BUG: id could be repeated
	if(!hostPrimObjIdList)
	{
		printf("malloc failed %s, %s \n", __FILE__, __LINE__);
	}

	if(devicePrimObjIdList)
	{
		cudaFree(devicePrimObjIdList);
	}
	err = cudaMalloc(&devicePrimObjIdList, sizeof(unsigned) * nTotalIdCount);
	if(err != cudaSuccess)
	{
		printf("CUDA error %s, %s, %s \n", __FILE__, __LINE__, cudaGetErrorString(err));
	}

	///	Serialize KD-Tree and PrimaryObject Ids
	serialize_kd_tree(pNode, hostKdRoot, hostPrimObjIdList);
	nCurrInxInx = 0;

	//	copy KD-tree data from host to device
	err = cudaMemcpy(deviceKdRoot, hostKdRoot, sizeof(kd_node_gpu) * nKdTreeNodeCount, cudaMemcpyHostToDevice);
	if(err != cudaSuccess)
	{
		printf("CUDA error %s, %s, %s \n", __FILE__, __LINE__, cudaGetErrorString(err));
	}

	err = cudaMemcpy(devicePrimObjIdList, hostPrimObjIdList, sizeof(unsigned) * nTotalIdCount, cudaMemcpyHostToDevice);
	if(err != cudaSuccess)
	{
		printf("CUDA error %s, %s, %s \n", __FILE__, __LINE__, cudaGetErrorString(err));
	}
	
	delete pNode;

	//	cuda copy objs
	//
	nCurrObjCount = nTotalPrimaryCount;
	err = cudaMemcpy(gpuObjs, hostObjs, sizeof(PrimGpuObj) * nTotalPrimaryCount, cudaMemcpyHostToDevice);
	if(err != cudaSuccess)
	{
		printf("CUDA error %s, %s, %s \n", __FILE__, __LINE__, cudaGetErrorString(err));
	}

	cudaThreadSynchronize();
}

void copySceneLights()
{
	///
	///		NOTE: All kinds of lights in CPU side. Since CUDA doesn't support
	///			  C++ features as polymorphism, they have to be handled separately.
	///

	//	1.	Count Lights
	//
	unsigned nLightCount = scene.getLightNum();

	//	re-alloc space
	if(gpuLights)
	{
		cudaFree(gpuLights);
	}
	cudaError_t err = cudaMalloc(&gpuLights, sizeof(LightGpu) * nLightCount);
	if(err != cudaSuccess)
	{
		printf("CUDA error %s, %s, %s \n", __FILE__, __LINE__, cudaGetErrorString(err));
	}

	if(cpuLights)
	{
		free(cpuLights);
	}
	cpuLights = (LightCpu*)malloc(sizeof(LightCpu) * nLightCount);
	if(!cpuLights)
	{
		printf("malloc failed %s, %s \n", __FILE__, __LINE__);
	}

	//	copy to host
	for(int i = 0; i < nLightCount; i ++)
	{
		Light *pLight = scene.getLight(i);
		LightType eType = pLight->getType();

		LightCpu *pCurrCpuLight = cpuLights + i;
		pCurrCpuLight->eType = eType;

		// common
		pCurrCpuLight->_fAttenuate = pLight->_fAttenuate;
		vecCopy(pCurrCpuLight->_ambientColor, pLight->_ambientColor);
		vecCopy(pCurrCpuLight->_diffuseColor, pLight->_diffuseColor);
		vecCopy(pCurrCpuLight->_specularColor, pLight->_specularColor);

		switch(eType)
		{
		case OMNI_P:
			{
				OmniPointLight *pOPl = dynamic_cast<OmniPointLight*>(pLight);
				vecCopy(pCurrCpuLight->_omni_pos, pOPl->_pos);
			}
			break;
			
		case DIR_P:
			{
				DirPointLight *pDPl = dynamic_cast<DirPointLight*>(pLight);
				vecCopy(pCurrCpuLight->_dirp_pos, pDPl->_pos);	
				vecCopy(pCurrCpuLight->_dirp_dir, pDPl->_dir);
			}
			break;
			
		case DIR:
			{
				DirLight *pDl = dynamic_cast<DirLight*>(pLight);
				vecCopy(pCurrCpuLight->_dir_dir, pDl->_dir);
			}
			break;
		}
	}

	//	copy to gpu
	//
	nCurrLightCount = nLightCount;
	cudaMemcpy(gpuLights, cpuLights, sizeof(LightGpu) * nLightCount, cudaMemcpyHostToDevice);

	//	Alloc Diffuse Lights space
	//
	err = cudaMalloc(&gpuDiffLightsPerLine, sizeof(LightCpu) * WinWidth * MAX_RAY_COUNT_PER_TREE);
	if(err != cudaSuccess)
	{
		printf("CUDA error %s, %s, %s \n", __FILE__, __LINE__, cudaGetErrorString(err));
	}

	cpuDiffLightsPerLine = (LightCpu *)malloc(sizeof(LightGpu) * WinWidth * MAX_RAY_COUNT_PER_TREE);
	if(!cpuDiffLightsPerLine)
	{
		printf("malloc failed %s, %s \n", __FILE__, __LINE__);
	}

	//	Random number 
	err = cudaMalloc(&gpuRdm, sizeof(float) * WinWidth * MAX_RAY_COUNT_PER_TREE);
	if(err != cudaSuccess)
	{
		printf("CUDA error %s, %s, %s \n", __FILE__, __LINE__, cudaGetErrorString(err));
	}

	cpuRdm = (float *)malloc(sizeof(float) * WinWidth * MAX_RAY_COUNT_PER_TREE);
	if(!cpuRdm)
	{
		printf("malloc failed %s, %s \n", __FILE__, __LINE__);
	}
	
	for(int i = 0; i < WinWidth ; i ++)
	for(int j = 0; j < MAX_RAY_COUNT_PER_TREE; j ++)
	{
		*(cpuRdm + j + i * MAX_RAY_COUNT_PER_TREE) = (rand() % 1000000) / 1000000.f;
	}
	cudaMemcpy(gpuRdm, cpuRdm, sizeof(float) * WinWidth * MAX_RAY_COUNT_PER_TREE, cudaMemcpyHostToDevice);

	cudaThreadSynchronize();
}

void releaseSceneGeomotry()
{
	nCurrObjCount = 0;

	if(gpuObjs)
	{
		cudaFree(gpuObjs);
		gpuObjs = NULL;
	}

	if(hostObjs)
	{
		free(hostObjs);
		hostObjs = NULL;
	}

	if(deviceKdRecBuf)
	{
		cudaFree(deviceKdRecBuf);
		deviceKdRecBuf = NULL;
	}

	if(hostKdRoot)
	{
		free(hostKdRoot);
		hostKdRoot = NULL;
	}

	if(deviceKdRoot)
	{
		cudaFree(deviceKdRoot);
		deviceKdRoot = NULL;
	}
	
	if(hostPrimObjIdList)
	{
		free(hostPrimObjIdList);
		hostPrimObjIdList = NULL;
	}

	if(devicePrimObjIdList)
	{
		cudaFree(devicePrimObjIdList);
		devicePrimObjIdList = NULL;
	}

	//
	if(gpuDiffLightsPerLine)
	{
		cudaFree(gpuDiffLightsPerLine);
		gpuDiffLightsPerLine = NULL;
	}

	if(cpuDiffLightsPerLine)
	{
		free(cpuDiffLightsPerLine);
		cpuDiffLightsPerLine = NULL;
	}
}

void releaseSceneLights()
{
	nCurrLightCount = 0;

	if(gpuLights)
	{
		cudaFree(gpuLights);
		gpuLights = NULL;
	}

	if(cpuLights)
	{
		free(cpuLights);
		cpuLights = NULL;
	}
}

static void copyTriangle(PrimGpuObj_host *pCurrPrimGpuObj, ObjObject *pObjObj, Triangle *pCurrTri)
{
	pCurrPrimGpuObj->eType = TRI_GPU;
	pCurrPrimGpuObj->nId = pCurrTri->_id;

	pCurrPrimGpuObj->_fReflectionRatio = pObjObj->getReflectionRatio();
	pCurrPrimGpuObj->_fRefractionRatio = pObjObj->getRefractionRatio();
	pCurrPrimGpuObj->_fRefractionK = pObjObj->getRefractionK();
	pCurrPrimGpuObj->_fEmitRatio = pObjObj->getEmissionRatio();
	pCurrPrimGpuObj->_mat.fShininess = pObjObj->_mat.fShininess;	

	vecCopy(pCurrPrimGpuObj->_mat.specColor, pObjObj->_mat.specColor);
	vecCopy(pCurrPrimGpuObj->_mat.diffColor, pObjObj->_mat.diffColor);
	vecCopy(pCurrPrimGpuObj->_mat.ambiColor, pObjObj->_mat.ambiColor);
	
	for(int n = 0; n < 3; n ++)
	{
		vecCopy(pCurrPrimGpuObj->_vertices[n], pCurrTri->_vertices[n]);
		vecCopy(pCurrPrimGpuObj->_vnormal[n], pCurrTri->_vnormal[n]);
	}
	vecCopy(pCurrPrimGpuObj->_normal, pCurrTri->_normal);

	pCurrPrimGpuObj->_bSmooth = pObjObj->_bSmooth;
	pCurrPrimGpuObj->_bHasVNorm = pObjObj->_bHasVNorm;

	//	NOTE: not supported yet
	pCurrPrimGpuObj->pTex = NULL;
}

static void copySquare(PrimGpuObj_host *pCurrPrimGpuObj, Square *pSqu)
{
	pCurrPrimGpuObj->eType = SQU_GPU;
				
	pCurrPrimGpuObj->nId = pSqu->_id;
	vecCopy(pCurrPrimGpuObj->_vNormal, pSqu->_vNormal);
	vecCopy(pCurrPrimGpuObj->_vWidthVec, pSqu->_vWidthVec); 
	vecCopy(pCurrPrimGpuObj->_vCenter, pSqu->_vCenter);

	pCurrPrimGpuObj->_nWidth = pSqu->_nWidth;
	pCurrPrimGpuObj->_nHeight = pSqu->_nHeight;	
	
	vecCopy(pCurrPrimGpuObj->_v2HeightVec, pSqu->_v2HeightVec);
	vecCopy(pCurrPrimGpuObj->_v2WidthVec, pSqu->_v2WidthVec);

	pCurrPrimGpuObj->a = pSqu->a;
	pCurrPrimGpuObj->b = pSqu->b;
	pCurrPrimGpuObj->c = pSqu->c;
	pCurrPrimGpuObj->d = pSqu->d;

	//	Tex
	if(pSqu->_tex)
	{
		int nMipInx = 0;
		int texId = TextureManager::find(pSqu->_tex, &nMipInx);
		if(texId != -1)
		{
			pCurrPrimGpuObj->pTex = pTexGpu + texId * MAX_GPU_MIPMAP_COUNT + nMipInx;
			pCurrPrimGpuObj->eTexType = pSqu->_eTexMapType;
		}
	}
	else
	{
		pCurrPrimGpuObj->pTex = NULL;
	}
}

static void copySphere(PrimGpuObj_host *pCurrPrimGpuObj, Sphere *pSph)
{
	pCurrPrimGpuObj->nId = pSph->_id;
	pCurrPrimGpuObj->eType = SPH_GPU;
	pCurrPrimGpuObj->_fRad = pSph->_fRad;
	vecCopy(pCurrPrimGpuObj->_ctr, pSph->_ctr);

	//	Tex
	int nMipInx = 0;
	int texId = TextureManager::find(pSph->_tex, &nMipInx);
	if(texId != -1)
	{
		pCurrPrimGpuObj->pTex = pTexGpu + texId * MAX_GPU_MIPMAP_COUNT + nMipInx;
		pCurrPrimGpuObj->eTexType = pSph->_eTexMapType;
	}
	else
	{
		pCurrPrimGpuObj->pTex = NULL;
	}
}