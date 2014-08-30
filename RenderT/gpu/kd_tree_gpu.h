#ifndef KD_TREE_GPU_H
#define KD_TREE_GPU_H

#include "bbox.h"

struct kd_node_gpu
{
	unsigned _nDepth;
	BBox	_bbox;

	AxisType eAxis;
	float _delim;

	int child0Inx; 
	int child1Inx; 

	int nInxStartInx;
	unsigned nInxCount;
};

#endif