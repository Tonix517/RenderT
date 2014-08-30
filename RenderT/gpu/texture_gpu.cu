#ifndef TEXTURE_GPU_CU
#define TEXTURE_GPU_CU

#include "texture.h"

#include <stdio.h>

#include "gpu/vector_gpu.cu"

#define MAX_GPU_TEX_COUNT (15)
#define MAX_GPU_MIPMAP_COUNT (4)

///	NOTE: CUDA Texture API has fatal limitations... I don't like it.

struct texture_s_gpu
{
	unsigned width;
	unsigned height;
	float *data;

	
};

__device__
void getTexAt(texture_s_gpu *pTex, unsigned nWidInx, unsigned nHeiInx, vect3d_gpu &ret);

texture_s_gpu texCpu[MAX_GPU_TEX_COUNT * MAX_GPU_MIPMAP_COUNT];

texture_s_gpu *pTexGpu;

__device__
void getTexAt(texture_s_gpu *pTex, unsigned nWidInx, unsigned nHeiInx, vect3d_gpu &ret)
{
	float *pColor = ((float*)pTex->data) + (nHeiInx * pTex->width + nWidInx) * 3;
	ret.data[0] = pColor[0];
	ret.data[1] = pColor[1];
	ret.data[2] = pColor[2];
}

///
///		LOAD
///
void loadTexture2GPU()
{
	//	Init
	for(unsigned i = 0; i < MAX_GPU_TEX_COUNT; i ++)
	{
		for(int j = 0; j < MAX_GPU_MIPMAP_COUNT; j ++)
		{
			texCpu[MAX_GPU_MIPMAP_COUNT * i + j].width = 0;
			texCpu[MAX_GPU_MIPMAP_COUNT * i + j].height = 0;
			texCpu[MAX_GPU_MIPMAP_COUNT * i + j].data = NULL;			
		}
	}
	cudaError_t err = cudaMalloc( &pTexGpu, sizeof(texture_s_gpu) * MAX_GPU_TEX_COUNT * MAX_GPU_MIPMAP_COUNT);
	if(err != cudaSuccess)
	{
		printf("CUDA error %s, %s, %s \n", __FILE__, __LINE__, cudaGetErrorString(err));
	}

	cudaMemcpy(pTexGpu, texCpu, sizeof(texture_s_gpu) * MAX_GPU_TEX_COUNT * MAX_GPU_MIPMAP_COUNT, cudaMemcpyHostToDevice);

	//	Load
	unsigned nTexCount = TextureManager::getTextureCount();
	if(nTexCount > MAX_GPU_TEX_COUNT)
	{
		printf("Too many textures on CPU....\n");
		return;
	}
	
	//	http://visionexperts.blogspot.com/2009/11/rgb-images-and-cuda.html
	
	for(unsigned i = 0; i < nTexCount; i ++)
	{
		texture_s *pCurrTex = TextureManager::getTexture(i);

		for(int j = 0; j < MAX_GPU_MIPMAP_COUNT; j ++)
		{
			if( !pCurrTex )	break;

			texture_s_gpu &rCurrTex = texCpu[MAX_GPU_MIPMAP_COUNT * i + j];

			// set texture parameters
			rCurrTex.width = pCurrTex->nWidth;
			rCurrTex.height = pCurrTex->nHeight;

			err = cudaMalloc(&rCurrTex.data, sizeof(float) * 3 * rCurrTex.width * rCurrTex.height);
			if(err != cudaSuccess)
			{
				printf("CUDA error %s, %s, %s \n", __FILE__, __LINE__, cudaGetErrorString(err));
			}

			cudaMemcpy(rCurrTex.data, pCurrTex->data, sizeof(float) * 3 * rCurrTex.width * rCurrTex.height, cudaMemcpyHostToDevice);

			pCurrTex = pCurrTex->pNextMip;
		}
	}

	cudaMemcpy(pTexGpu, texCpu, sizeof(texture_s_gpu) * MAX_GPU_TEX_COUNT * MAX_GPU_MIPMAP_COUNT, cudaMemcpyHostToDevice);
	cudaThreadSynchronize();
}

///
///		UNLOAD
///
void unloadGpuTexture()
{
	unsigned nTexCount = TextureManager::getTextureCount();
	for(unsigned i = 0; i < nTexCount; i ++)
	{
		for(int j = 0; j < MAX_GPU_MIPMAP_COUNT; j ++)
		{
			texture_s_gpu &rCurrTex = texCpu[MAX_GPU_MIPMAP_COUNT * i + j];
			if(rCurrTex.data)
			{
				cudaFree(rCurrTex.data);
				rCurrTex.data = NULL;
			}
		}
	}

	if(pTexGpu)
	{
		cudaFree(pTexGpu);
		pTexGpu = NULL;
	}
}

#endif