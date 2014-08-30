#ifndef LIGHTCUTS_GPU_H
#define LIGHTCUTS_GPU_H

#include  "_lightcuts.h"
#include "gpu/vector_gpu.cu"

#include "gpu/geometry_gpu.h"

//
//	WARNING: should be exactly the same as in _lightcuts.h !!
//
struct lt_node_gpu
{
	lt_node_gpu()
	{
		inx_in_light_array = -1;
		l_child_inx_in_tree = -1;
		r_child_inx_in_tree = -1;
	}

	__device__
	void setValue(short lightInx, short l_inx, short r_inx)
	{
		inx_in_light_array = lightInx;
		l_child_inx_in_tree = l_inx;
		r_child_inx_in_tree = r_inx;
	}

	////

	short inx_in_light_array;
	short l_child_inx_in_tree;
	short r_child_inx_in_tree;
};

__device__
float estimateRelevantFactor_gpu(vect3d_gpu &vHitPoint, LightGpu *pLight);

__device__
void evalColorByLightcuts_gpu(vect3d_gpu &vStartPoint, vect3d_gpu &vHitPoint, vect3d_gpu &vNorm, material_gpu *mat, vect3d_gpu &rRetColor);

struct light_rec
{
	float fRelFactor;
	bool  bHit;
};

//

void setupLightcutsParam(lt_node *pRoot, unsigned nTotalNodeCount, unsigned nLightCount, float fLightCutsThreshold);
float getAvgCutRatioGpu();
void releaseLightcutsParam();

#endif