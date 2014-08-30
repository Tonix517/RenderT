#ifndef VECTOR_H
#define VECTOR_H

#include <memory.h>
#include <math.h>
#include <assert.h>

struct vect3d
{
	
	float data[3];

	vect3d()
	{
		data[0] = 0;
		data[1] = 0;
		data[2] = 0;
	}

	float& operator[](int inx)
	{		
		assert(inx <3 && inx > -1);
		return data[inx];
	}
};

inline void points2vec(vect3d &vStartPoint, vect3d &vEndPoint, vect3d &vVec)
{
	vVec[0] = vEndPoint[0] - vStartPoint[0];
	vVec[1] = vEndPoint[1] - vStartPoint[1];
	vVec[2] = vEndPoint[2] - vStartPoint[2];
}

inline float vecLen(vect3d &vVec)
{
	return sqrt( pow(vVec[0], 2) + 
				 pow(vVec[1], 2) + 
				 pow(vVec[2], 2) );
}

inline void vecScale(vect3d &vOrigVec, float fScale, vect3d &vScaledVec)
{
	vScaledVec[0] = fScale * vOrigVec[0];
	vScaledVec[1] = fScale * vOrigVec[1];
	vScaledVec[2] = fScale * vOrigVec[2];
}

inline void point2point(vect3d &vStartPoint, vect3d &vVec, vect3d &vEndPoint)
{
	vEndPoint[0] = vStartPoint[0] + vVec[0];
	vEndPoint[1] = vStartPoint[1] + vVec[1];
	vEndPoint[2] = vStartPoint[2] + vVec[2];
}

inline void cross_product(vect3d &vec1, vect3d &vec2, vect3d &vecr)
{
	vecr[0] = vec1[1] * vec2[2] - vec1[2] * vec2[1];
	vecr[1] = vec1[2] * vec2[0] - vec1[0] * vec2[2];
	vecr[2] = vec1[0] * vec2[1] - vec1[1] * vec2[0];
}

inline float dot_product(vect3d &vec1, vect3d &vec2)
{
	return	vec1[0] * vec2[0] + 
			vec1[1] * vec2[1] + 
			vec1[2] * vec2[2];
}

inline void normalize(vect3d &vec)
{
	float fLen = vecLen(vec);
	if(fLen == 0) return;

	float v = 1/fLen;
	vec[0] *= v;
	vec[1] *= v;
	vec[2] *= v;
}

inline void vecCopy(vect3d &destVec, float *srcVec)
{
	destVec[0] = srcVec[0];
	destVec[1] = srcVec[1];
	destVec[2] = srcVec[2];
}

inline void vecCopy(vect3d &destVec, vect3d &srcVec)
{
	destVec[0] = srcVec[0];
	destVec[1] = srcVec[1];
	destVec[2] = srcVec[2];
}


//	Ray-Tracing related

void reflectVec(vect3d &vOrigViewVec, vect3d &vNormal, vect3d &vReflectViewVec);

void refractVec(vect3d &vOrigViewVec, vect3d &vNormal, vect3d &vRefractedVec, float refraK);

void projectPoint(vect3d &pEyePos, vect3d &vViewVec, float t, vect3d &vTargetPoint);

float point2line(vect3d & vPoint, vect3d &pEyePos, vect3d &pViewVec, float *pT);

///
///	REF: Real-Time Rendering 2nd, 3.1.7, 3.1.8
///
///	MATRIX: row first

//	for performance consideration, this should be called first
void set_matrix(float sin, float cos, vect3d &r);

//	Rotation matrix for geometries & normals
//		NOTE: this only applies to rotation
//
void mat_rot(vect3d &p, vect3d &ret_p);

#endif