#ifndef KD_TREE_H
#define KD_TREE_H

#include <vector>
#include "bbox.h"
#include "object.h"
#include "consts.h"
#include "ray.h"

struct kd_node
{

	kd_node(unsigned nDepth = 0);
	~kd_node();

	//	Building...
	void addObject(Object *);
	void split(bool bScene = true);

	void updateBBox();

	///	Get the point
	Object* isHit(Ray &ray, vect3d &pNormal, float *pt, vect3d *pTexColor = NULL);

///
///		member vars
///

	unsigned _nDepth;
	BBox	_bbox;

	AxisType eAxis;
	float _delim;
	kd_node	*child0; // indicating to node that lies at the smaller value end
	kd_node *child1; // indicating to node that lies at the bigger value end

	static unsigned nSceneDepth;
	static unsigned nObjDepth;
private:

	float splitByAxis(AxisType eType, float *pCost);

	//	-1 for the smaller end
	//	 0 for the both ends
	//	 1 for the bigger end
	int delimitObject(Object *, AxisType eType, float value);

	float getCost(AxisType eType, float value, float min, float max);

public:
//private:
	std::vector<Object *>  objects;	
};

#endif