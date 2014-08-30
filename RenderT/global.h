#ifndef GLOBAL_H
#define GLOBAL_H

#include "scene.h"
#include "gpu/ray_gpu.cu"
extern Scene scene;

///
///		WARNING: This "volatile" is necessary because we have a while loop
///				 over it in end_thread() , and compiler considers this loop
///				 stupid and will do some harmful optimization - this leads to
///				 the bug in release mode that the value of bIsRunning will
///				 not be updated - this is a common issue in hardware driver
///				 development though.
///
extern volatile bool bIsRunning;

///
///		
///
extern Ray_gpu *_deviceRays;
extern Ray *_hostRays;
//extern Ray_gpu *_deviceRays;

void global_init();
void global_destroy();

#endif