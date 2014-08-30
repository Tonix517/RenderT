#ifndef LIGHTCUTS_H
#define LIGHTCUTS_H

#include <vector>
#include "light.h"

struct lt_node
{
	lt_node()
	{
		inx_in_light_array = -1;
		l_child_inx_in_tree = -1;
		r_child_inx_in_tree = -1;
	}

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

lt_node * buildLightTree(std::vector<Light*> &pointLights, unsigned &nRetNodeCount);
lt_node * buildLightTreeForGpu(std::vector<LightCpu> &pointLights, unsigned &nRetNodeCount);
void deleteLightTree();

float estimateRelevantFactor(vect3d &vHitPoint, Light *pLight);

////////////////////////

#endif