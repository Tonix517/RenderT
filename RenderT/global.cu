#include "global.h"

#include "consts.h"
#include "film.h"
#include "camera.h"
#include "thread.h"
#include "vector.h"
#include "_lightcuts.h"
#include "obj_object.h"
#include "gpu_util.h"
#include "gpu/geometry_gpu.h"
#include "gpu/_lightcuts_gpu.h"

#include "IL/ilut.h"
ILuint nCurrImg = 1;

#include <time.h>
#include <stdlib.h>

#include <cuda_runtime.h>

///
Ray *_hostRays = NULL;
Ray_gpu *_deviceRays = NULL;

//
//	Engine..
//
Scene scene;

volatile bool bIsRunning = false;

void global_init()
{
	//	DevIL init
	//
	ilInit();
	ilutRenderer(ILUT_OPENGL);
	ilutEnable(ILUT_OPENGL_CONV);

	//ilOriginFunc(IL_ORIGIN_UPPER_LEFT);
	//ilEnable(IL_ORIGIN_SET);

	ilGenImages(1, &nCurrImg);
	ilBindImage(nCurrImg);	

	//
	srand(clock());
}

void global_destroy()
{
	if(bLCEnabled)
	{
		deleteLightTree();

		if(bGPUEnabled)
		{
			releaseLightcutsParam();
		}
	}

	gpu_destroy();

	end_thread();

	//	DevIL finalization
	ilDeleteImages(1, &nCurrImg);
}