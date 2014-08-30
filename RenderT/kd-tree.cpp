#include "kd-tree.h"
#include "vector.h"

#include <assert.h>
#include <vector>
#include <set>
#include <algorithm>
#include <functional>

unsigned kd_node::nSceneDepth = 0;
unsigned kd_node::nObjDepth = 0;

kd_node::kd_node(unsigned nDepth)
	: child0(NULL)
	, child1(NULL)
	, _nDepth(nDepth)
	, _delim(0)
{
	
}

kd_node::~kd_node()
{
	if(child0)
	{
		delete child0;
	}

	if(child1)
	{
		delete child1;
	}
}

void kd_node::addObject(Object *pObj)
{
	assert(pObj);
	objects.push_back(pObj);
}

void kd_node::split(bool bScene)
{
	//	to split or not?
	if(objects.size() <= nMaxNodeCount || (_nDepth + 1) > (bScene? nSceneDepth : nObjDepth) )
	{
		printf("\tD : %d, C : %d \n", _nDepth, objects.size());
		return;
	}

	//
	float x_cost = 0, y_cost = 0, z_cost = 0;

	//	1. Calculate each AXIS
	float x_value = splitByAxis(X_AXIS, &x_cost);
	float y_value = splitByAxis(Y_AXIS, &y_cost);
	float z_value = splitByAxis(Z_AXIS, &z_cost);

	//	2. Get the axis to divide
	float theValue = x_value;
	AxisType retType = X_AXIS;
	if(y_cost < x_cost)
	{
		retType = Y_AXIS;
		theValue = y_value;

		if(z_cost < y_cost)
		{
			retType = Z_AXIS;
			theValue = z_value;
		}
	}
	else
	{
		if(z_cost < x_cost)
		{
			retType = Z_AXIS;
			theValue = z_value;
		}
	}
	

	//	3. Setup children kd_node
	child0 = new kd_node(_nDepth + 1);
	child1 = new kd_node(_nDepth + 1);

	float xmin = 0, xmax = 0;	_bbox.getBoundValues(X_AXIS, &xmin, &xmax);
	float ymin = 0, ymax = 0;	_bbox.getBoundValues(Y_AXIS, &ymin, &ymax);
	float zmin = 0, zmax = 0;	_bbox.getBoundValues(Z_AXIS, &zmin, &zmax);

	switch(retType)
	{
	case X_AXIS:

		//assert( theValue <= xmax && theValue >= xmin);
		child0->_bbox.setDim(xmin, theValue, ymin, ymax, zmin, zmax);
		child1->_bbox.setDim(theValue, xmax, ymin, ymax, zmin, zmax);
		
		break;

	case Y_AXIS:
		
		//assert( theValue <= ymax && theValue >= ymin);
		child0->_bbox.setDim(xmin, xmax, ymin, theValue, zmin, zmax);
		child1->_bbox.setDim(xmin, xmax, theValue, ymax, zmin, zmax);

		break;

	case Z_AXIS:
		
		//assert( theValue <= zmax && theValue >= zmin);
		child0->_bbox.setDim(xmin, xmax, ymin, ymax, zmin, theValue);
		child1->_bbox.setDim(xmin, xmax, ymin, ymax, theValue, zmax);

		break;
	}

	//	4. Assign objects
	for(int i = 0; i < objects.size(); i ++)
	{
		int ret = delimitObject(objects[i], retType, theValue);
		switch(ret)
		{
		case -1:
			child0->addObject(objects[i]);
			break;

		case 1:
			child1->addObject(objects[i]);
			break;

		case 0:
			child0->addObject(objects[i]);
			child1->addObject(objects[i]);	
			break;
		}
	}

	//	5. Clear current objects
	eAxis = retType;
	_delim = theValue;
	
	if( child0->objects.size() < objects.size() && 
		child0->objects.size() > 0)	// avoid redundant recursion and splitting empty set
	{
		child0->split(bScene);
	}
	else
	{
		printf("\t[D: %d, C: %d]\n", _nDepth, child0->objects.size());
	}

	if( child1->objects.size() < objects.size() && 
		child1->objects.size() > 0)	// avoid redundant recursion and splitting empty set
	{
		child1->split(bScene);
	}
	else
	{
		printf("\t[D: %d, C: %d]\n", _nDepth, child1->objects.size());
	}

	objects.clear();
}

float kd_node::splitByAxis(AxisType eType, float *pCost)
{
	if(objects.size() == 0)
	{
		return 0;
	}

	float setMin = 999999.f, setMax = -999999.f;
	//	collect values
	std::set<float> valueSet;
	for(int i = 0; i < objects.size(); i ++)
	{
		float min = 0, max = 0;
		objects[i]->getBBox()->getBoundValues(eType, &min, &max);
		if(min < setMin) setMin = min;
		if(max > setMax) setMax = max;
		valueSet.insert(min);
		valueSet.insert(max);
	}

	float retValue = 0, fCost = 999999.f;

	assert(setMin <= setMax);

	//	The bound values will not be considered
	for(std::set<float>::iterator it = valueSet.begin(); it != valueSet.end(); it++)
	{		
		std::set<float>::iterator tmpIt = it;

		float value = *tmpIt;
		if(value != setMin && value != setMax)
		{
			float nextValue = *(++tmpIt);
			value = (value + nextValue) / 2; // pick the middle line so to promise no object will cling to the plane ...

			float fCurrCost = getCost(eType, value, setMin, setMax);
			if(fCurrCost < fCost)
			{
				fCost = fCurrCost;
				retValue = value;
			}
		}
	}

	*pCost = fCost;
	return retValue;
}

//	-1 for the smaller end
//	 0 for the both ends
//	 1 for the bigger end
//
int kd_node::delimitObject(Object *pObj, AxisType eType, float value)
{
	float min = 0, max = 0;	
	pObj->getBBox()->getBoundValues(eType, &min, &max);

	if(max <= value)
	{
		return -1;
	}
	else if(min >= value)
	{
		return  1;
	}
	else
	{
		assert( min <= value && value <= max);
		return 0;
	}

}

float kd_node::getCost(AxisType eType, float value, float min, float max)
{
	unsigned nCount0 = 0, nCount1 = 0;

	for(int i = 0; i < objects.size(); i ++)
	{
		int ret = delimitObject(objects[i], eType, value);
		switch(ret)
		{
		case 1:
			nCount1 ++;
			break;

		case -1:
			nCount0 ++;
			break;

		case 0:
			nCount0 ++;
			nCount1 ++;
			break;
		}
	}

	float fMinRatio = (value - min) / (max - min);;	// a value of 0.5 is ok - to promise the same num. of objs on each side

	//	This is how the cost is computed ...
	return fMinRatio * nCount0 / (nCount0 + nCount1) + (1 - fMinRatio) * nCount1 / (nCount0 + nCount1);
}

void kd_node::updateBBox()
{
	float xmin = 99999999.f, xmax = -99999999.f;
	float ymin = 99999999.f, ymax = -99999999.f;
	float zmin = 99999999.f, zmax = -99999999.f;

	for(int i = 0; i < objects.size(); i ++)
	{
		float cXmin = 0, cXmax = 0;
		objects[i]->getBBox()->getBoundValues(X_AXIS, &cXmin, &cXmax);
		if(cXmin < xmin)	xmin = cXmin;
		if(cXmax > xmax)	xmax = cXmax;

		float cYmin = 0, cYmax = 0;
		objects[i]->getBBox()->getBoundValues(Y_AXIS, &cYmin, &cYmax);
		if(cYmin < ymin)	ymin = cYmin;
		if(cYmax > ymax)	ymax = cYmax;

		float cZmin = 0, cZmax = 0;
		objects[i]->getBBox()->getBoundValues(Z_AXIS, &cZmin, &cZmax);
		if(cZmin < zmin)	zmin = cZmin;
		if(cZmax > zmax)	zmax = cZmax;
	}

	_bbox.setDim(xmin, xmax, ymin, ymax, zmin, zmax);
}

Object* kd_node::isHit(Ray &ray, vect3d &pNormal, float *pt, vect3d *pTexColor)
{
	//
	//	Removing this can improve the performance quite a lot...
	//
	//if( !_bbox.isHit(ray) )
	//{
	//	return false;
	//}

	if(child0 != NULL && child1 != NULL)
	{
		bool bHit0 = child0->_bbox.isHit(ray);
		bool bHit1 = child1->_bbox.isHit(ray);
		
		//	1. None is hit
		if( !bHit0 && !bHit1)
		{
			return NULL;
		}

		//	2. Both are hit
		if(bHit0 && bHit1)
		{
			vect3d texColor, texColor1;
			vect3d norm, norm1;
			float t, t1;

			bool bChild0 = false;

			Object *pObj0 = child1->isHit(ray, norm, &t, &texColor);
			Object *pObj1 = child0->isHit(ray, norm1, &t1, &texColor1);

			if(pObj0 && pObj1)
			{
				if(t < t1)
				{
					bChild0 = true;
				}
				else
				{

					bChild0 = false;
				}
			}
			else if( pObj0 && !pObj1)
			{
				bChild0 = true;
			}
			else if( !pObj0 && pObj1)
			{
				bChild0 = false;
			}
			else if( !pObj0 && !pObj1)
			{
				return NULL;
			}

			if(bChild0)
			{
				vecCopy(pNormal, norm);
				if(pTexColor)
				{
					vecCopy(*pTexColor, texColor);
				}
				*pt = t;
				return pObj0;
			}
			else
			{
				vecCopy(pNormal, norm1);
				if(pTexColor)
				{
					vecCopy(*pTexColor, texColor1);
				}
				*pt = t1;
				return pObj1;	
			}

			assert(false);
		}//	hit both

		//	3. One is hit
		if(bHit0)
		{
			return child0->isHit(ray, pNormal, pt, pTexColor);
		}
		if(bHit1)
		{
			return child1->isHit(ray, pNormal, pt, pTexColor);
		}
	}
	else
	{

		vect3d norm;
		float t = 99999999.0;
		Object *pObj = NULL;
		vect3d texColor;

		for(int i = 0; i < objects.size(); i ++)
		{
			if(objects[i]->_nLastVisitingRay != ray.id)
			{
				objects[i]->_nLastVisitingRay = ray.id;

				vect3d tmpNorm;
				float tmpT = 0;
				vect3d currTexColor;

				if(objects[i]->isHit(ray, tmpNorm, &tmpT, &currTexColor))
				{
					if(tmpT < t && tmpT > 0)
					{
						t = tmpT;
						vecCopy(norm, tmpNorm);					
						pObj = objects[i];
						vecCopy(texColor, currTexColor);
					}
				}
			}
		}
		if( t < 99999999.0 && t > epsi)
		{
			vecCopy(pNormal, norm);
			*pt = t;			
			if(pTexColor)
			{
				vecCopy(*pTexColor, texColor);
			}
			return pObj;
		}		
	}

	return NULL;
}