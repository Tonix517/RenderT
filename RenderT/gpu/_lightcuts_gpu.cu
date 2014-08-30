#include "_lightcuts_gpu.h"
#include "consts.h"

__device__
float estimateRelevantFactor_gpu(vect3d_gpu &vHitPoint, LightGpu *pLight)
{
	vect3d_gpu vl2hit;
	points2vec_gpu(pLight->_dirp_pos, vHitPoint, vl2hit);

	//	1. Distance
	float fDist = vecLen_gpu(&vl2hit);

	//	2. Illumination
	float fIllum = vecLen_gpu(&pLight->_diffuseColor);

	//	3. Oriental cos
	float cosTheta = dot_product_gpu(vl2hit, pLight->_dirp_dir) / (fDist * vecLen_gpu(&pLight->_dirp_dir));

	return (cosTheta > 0 ? cosTheta : 0) * fIllum / (fDist * fDist);
}

////////

//	1. global Lt Tree
//
		   lt_node_gpu *gpuLtTreePtr = NULL;
__device__ lt_node_gpu *gpuLtTree = NULL;

		   short *gpuQueueSpacePtr = NULL;
__device__ short *gpuQueueSpace = NULL;

		   light_rec *gpuRecSpacePtr = NULL;
__device__ light_rec *gpuRecSpace = NULL;

__device__ unsigned gpuLightCount = 0;
__device__ float gpuRelThreshold = 0;

		   float *gpuLightStatBuf = NULL;
__device__ float *gpuLightStat = NULL;
		   unsigned *gpuLightCountStatBuf = NULL;
__device__ unsigned *gpuLightCountStat = NULL;

////////

__global__
void setupLightcutsParamGpu(lt_node_gpu *pTreePtrGpu, short *pQueueSpacePtr, light_rec *pRecSpace, unsigned nLightCount, 
							float *pGpuLightStatBuf, unsigned *pGpuLightCountStatBuf, float fLCThreshold)
{
	gpuRelThreshold = fLCThreshold;

	gpuLtTree = pTreePtrGpu;
	gpuQueueSpace = pQueueSpacePtr;

	gpuRecSpace = pRecSpace;
	gpuLightCount = nLightCount;
	
	gpuLightStat = pGpuLightStatBuf;
	gpuLightCountStat = pGpuLightCountStatBuf;

	for(int i = 0; i < WIN_WIDTH; i ++)
	{
		light_rec *pCurrPixelRec = gpuRecSpace + i * nLightCount;
		for(int j = 0; j < nLightCount; j ++)
		{
			(pCurrPixelRec + j)->fRelFactor = -1;
			(pCurrPixelRec + j)->bHit = false;
		}
		//
		*(gpuLightStat + i) = 0;
		*(gpuLightCountStat + i) = 0;
	}	
}

//	CPU code

void setupLightcutsParam(lt_node *pRoot, unsigned nTotalNodeCount, unsigned nLightCount, float fLCThreshod)
{
	//	Global Lightcuts Tree
	//
	cudaError_t err = cudaMalloc(&gpuLtTreePtr, sizeof(lt_node_gpu) * nTotalNodeCount);
	if(err != cudaSuccess)
	{
		printf("\nin setupLightcutsParam(): cudaMalloc failed..\n");
	}
	err = cudaMemcpy( gpuLtTreePtr,	pRoot, sizeof(lt_node_gpu) * nTotalNodeCount, cudaMemcpyHostToDevice);
	if(err != cudaSuccess)
	{
		printf("\nin setupLightcutsParam(): cudaMemcpy failed..\n");
	}

	//	Queue space for each Pixel
	//
	unsigned nQueueSize = (nLightCount % 2 == 0) ? nLightCount : nLightCount + 1;
	err = cudaMalloc(&gpuQueueSpacePtr, sizeof(short) * (nQueueSize) * WinWidth);
	if(err != cudaSuccess)
	{
		printf("\nin setupLightcutsParam(): cudaMalloc failed..\n");
	}

	//	Record space for each Pixel
	//
	err = cudaMalloc(&gpuRecSpacePtr, sizeof(light_rec) * nLightCount * WinWidth);
	if(err != cudaSuccess)
	{
		printf("\nin setupLightcutsParam(): cudaMalloc failed..\n");
	}

	err = cudaMalloc(&gpuLightStatBuf, sizeof(float) * WinWidth);
	if(err != cudaSuccess)
	{
		printf("\nin setupLightcutsParam(): cudaMalloc failed..\n");
	}

	err = cudaMalloc(&gpuLightCountStatBuf, sizeof(unsigned) * WinWidth);
	if(err != cudaSuccess)
	{
		printf("\nin setupLightcutsParam(): cudaMalloc failed..\n");
	}

	// Pass to GPU
	setupLightcutsParamGpu<<<1, 1>>>(gpuLtTreePtr, gpuQueueSpacePtr, gpuRecSpacePtr, nLightCount, 
									 gpuLightStatBuf, gpuLightCountStatBuf, fLCThreshod);
}

void releaseLightcutsParam()
{
	if(gpuLightCountStatBuf != NULL)
	{
		cudaFree(gpuLightCountStatBuf);
		gpuLightCountStatBuf = NULL;
	}

	if(gpuLightStatBuf != NULL)
	{
		cudaFree(gpuLightStatBuf);
		gpuLightStatBuf = NULL;
	}    

	if(gpuLtTreePtr != NULL)
	{
		cudaFree(gpuLtTreePtr);
		gpuLtTreePtr = NULL;
	}

	if(gpuQueueSpacePtr != NULL)
	{
		cudaFree(gpuQueueSpacePtr);
		gpuQueueSpacePtr = NULL;
	}

	if(gpuRecSpacePtr != NULL)
	{
		cudaFree(gpuRecSpacePtr);
		gpuRecSpacePtr = NULL;
	}
}

float getAvgCutRatioGpu()
{
	//	just 512 for now, so..
	float buf[WIN_WIDTH] = {0};
	unsigned cbuf[WIN_WIDTH] = {0};

	//	1. Count
	//
	cudaError_t err = cudaMemcpy(	buf, gpuLightStatBuf, 
									sizeof(float) * WinWidth, cudaMemcpyDeviceToHost );
	if(err != cudaSuccess)
	{
		printf("cudaMemcpy error : getAvgCutRatioGpu()\n");
	}

	err = cudaMemcpy(	cbuf, gpuLightCountStatBuf, 
									sizeof(unsigned) * WinWidth, cudaMemcpyDeviceToHost );
	if(err != cudaSuccess)
	{
		printf("cudaMemcpy error : getAvgCutRatioGpu()\n");
	}

	//	2. Calc.
	//
	float ret = 0;
	unsigned count = 0;
	for(int i = 0; i < WinWidth; i ++)
	{
		ret += buf[i];
		count += cbuf[i];
	}

	return ret / count;
}
