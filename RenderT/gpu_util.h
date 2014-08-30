#ifndef GPU_UTIL_H
#define GPU_UTIL_H

#include "obj_object.h"
#include "gpu/geometry_gpu.h"

void sendConstants2GPU();
void copySceneGeomotry();
void copySceneLights();
void releaseSceneGeomotry();
void releaseSceneLights();

//
static void copyTriangle(PrimGpuObj_host *, ObjObject *pObjObj, Triangle *);
static void copySquare(PrimGpuObj_host *, Square *);
static void copySphere(PrimGpuObj_host *, Sphere *);

void gpu_destroy();

#endif