#include "ray.h"

#include <assert.h>
#include <algorithm>

///
///		WARNING: Parallelism...
///
static IdType nRayId = 0;

Ray::Ray()
	: bIsInObj(false), fDeltaX(0), fDeltaY(0)
{
	id = -1;	// for GPU to recognize empty node in the tree
	fRefl = 0;
	fRefr = 0;
	fEmit = 0;
}

void Ray::reset()
{
	bIsInObj = false;
	id = -1;
	fDeltaX = 0;
	fDeltaY = 0;
	fEmit = 0;
	fRefl = 0;
	fRefr = 0;
	vect3d null;
	vecCopy(start_point, null);
	vecCopy(direction_vec, null);
	vecCopy(color, null);
	vecCopy(_hitPoint, null);
	vecCopy(_hitNorm, null);
}

///

void Ray::copy(Ray &ray)
{
	id = ray.id;
	
	vecCopy(start_point, ray.start_point);
	vecCopy(direction_vec, ray.direction_vec);
	vecCopy(color, ray.color);

	fDeltaX = ray.fDeltaX;
	fDeltaY = ray.fDeltaY;

	bIsInObj = ray.bIsInObj;

	fEmit = ray.fEmit;
	fRefl = ray.fRefl;
	fRefr = ray.fRefr;

	vecCopy(_hitPoint, ray._hitPoint);
	vecCopy(_hitNorm, ray._hitNorm);
}

Ray::Ray(vect3d &pStart, vect3d &pDir, bool pbIsInObj)
	: bIsInObj(pbIsInObj), fDeltaX(0), fDeltaY(0)
{
	vecCopy(start_point, pStart);
	vecCopy(direction_vec, pDir);

	id = nRayId ++;

	fEmit = 0;
	fRefl = 0;
	fRefr = 0;
}

void clampColor(vect3d &pColor)
{
	for(int i = 0; i < 3; i ++)
	{
		if(pColor[i] > 1.f) pColor[i] = 1.f;
	}
}