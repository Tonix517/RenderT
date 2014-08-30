#include "thread.h"
#include "global.h"
#include "gpu_util.h"
#include <process.h>
#include <time.h>

#include "GL/glui.h"

extern GLUI *glui;
extern GLUI *glui2;

void start_thread()
{
	glui->disable();
	glui2->disable();

	//	TODO: cross-platform threading
	if(!bIsRunning && scene.bLoaded)
	{
		//scene.clear();
		_beginthread(thread_startComputing, 0, NULL);
	}
	else
	{
		printf(" Please load a scene file first !\n");
	}
}

void end_thread()
{
	scene.abort();
	while(bIsRunning);
	glui->enable();
	glui2->enable();
}

void start_bloom_thread()
{
	glui->disable();
	glui2->disable();

	//	TODO: cross-platform threading
	if(!bIsRunning && scene.bLoaded)
	{
		//scene.clear();
		_beginthread(thread_bloom, 0, NULL);
	}
	else
	{
		printf(" Please load a scene file first !\n");
	}
}

void thread_bloom(void *)
{
	Film::bloom();

	glui->enable();
	glui2->enable();
}

void start_tone_m_thread()
{
	glui->disable();
	glui2->disable();

	//	TODO: cross-platform threading
	if(!bIsRunning && scene.bLoaded)
	{
		//scene.clear();
		_beginthread(thread_tone_m, 0, NULL);
	}
	else
	{
		printf(" Please load a scene file first !\n");
	}
}

void thread_tone_m(void *)
{
	Film::toneMap();

	glui->enable();
	glui2->enable();
}

extern void loadTexture2GPU();
extern void unloadGpuTexture();

void thread_startComputing(void *)
{
	if(bGPUEnabled)
	{
		unsigned nCurrMaxRayNum = (unsigned)pow(2.f, MaxRayDepth * 1.f);
		if(bAOEnabled)
		{
			unsigned nAORaysPerRay = (fAngleScope/fAngleStep) * PI / 4.f;
			nCurrMaxRayNum *= (1 + nAORaysPerRay);
		}

		if(nCurrMaxRayNum > MAX_RAY_COUNT)
		{
			printf("The current Ray depth is too large for GPU !\n");
			return;
		}

		//
		printf("Sending data to GPU ...");

		//
		//
		if(!_hostRays)
		{
			_hostRays = new Ray[MAX_RAY_COUNT];//(Ray*)malloc(sizeof(Ray) * MAX_RAY_COUNT);			
		}

		if(!_deviceRays)
		{
			cudaError_t err = cudaMalloc((void **)&_deviceRays, sizeof(Ray_gpu) * MAX_RAY_COUNT);			
			if(err != cudaSuccess)
			{
				printf("CUDA error %s, %s, %s \n", __FILE__, __LINE__, cudaGetErrorString(err));
			}

			cudaMemset(_deviceRays, 0, sizeof(Ray_gpu) * MAX_RAY_COUNT);
		}

		//	Copy data to GPU
		sendConstants2GPU();

		loadTexture2GPU();	// this has to lead geometry copy.
		copySceneGeomotry();
		copySceneLights();
		
		printf("Done \n");
	}

	printf("Started...  ");
	bIsRunning = true;
	clock_t nStart = clock();
	scene.compute();
	bIsRunning = false;
	printf("[Cost : %.2f]\n", (clock() - nStart) / 1000.f);
	
	nCurrObj = 0;

	if(bGPUEnabled)
	{
		//	Release the memory here
		//
		if(_hostRays)
		{
			delete [] _hostRays;
			_hostRays = NULL;
		}
		if(_deviceRays)
		{
			cudaFree(_deviceRays);
			_deviceRays = NULL;
		}

		releaseSceneGeomotry();
		releaseSceneLights();
		unloadGpuTexture();
	}

	glui->enable();
	glui2->enable();
}