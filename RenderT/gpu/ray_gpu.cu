#ifndef RAY_GPU_H
#define RAY_GPU_H

#include "vector_gpu.cu"

typedef long long IdType;

struct Ray_gpu
{
public:
	//	For CUDA..
	__device__
	Ray_gpu()
		: bIsInObj(false), fDeltaX(0), fDeltaY(0)
	{
		id = -1;	// for GPU to recognize empty node in the tree
		fRefl = 0;
		fRefr = 0;

		vect3d_gpu null;
		vecCopy_gpu(start_point, null);
		vecCopy_gpu(direction_vec, null);
		vecCopy_gpu(color, null);
	}

	__device__
	Ray_gpu(vect3d_gpu &pStart, vect3d_gpu &pDir, bool pbIsInObj = false)
		: bIsInObj(pbIsInObj), fDeltaX(0), fDeltaY(0)
	{
		vecCopy_gpu(start_point, pStart);
		vecCopy_gpu(direction_vec, pDir);

		id = 0;	

		fEmit = 0;
		fRefl = 0;
		fRefr = 0;

		vect3d_gpu null;
		vecCopy_gpu(color, null);
	}


	__device__
	void reset()
	{
		bIsInObj = false;
		id = -1;
		fDeltaX = 0;
		fDeltaY = 0;

		fEmit = 0;
		fRefl = 0;
		fRefr = 0;

		vect3d_gpu null;
		vecCopy_gpu(start_point, null);
		vecCopy_gpu(direction_vec, null);
		vecCopy_gpu(color, null);

		vecCopy_gpu(_hitPoint, null);
		vecCopy_gpu(_hitNorm, null);
	}


	__device__
	void copy(Ray_gpu &ray)
	{
		bIsInObj = ray.bIsInObj;
		id = ray.id;
		fDeltaX = ray.fDeltaX;
		fDeltaY = ray.fDeltaY;
		vecCopy_gpu(start_point, ray.start_point);
		vecCopy_gpu(direction_vec, ray.direction_vec);
		vecCopy_gpu(color, ray.color);
		
		fEmit = ray.fEmit;
		fRefl = ray.fRefl;
		fRefr = ray.fRefr;

		vecCopy_gpu(_hitPoint, ray._hitPoint);
		vecCopy_gpu(_hitNorm, ray._hitNorm);
	}


	///
	///		Has to be exactly the same with CPU Ray
	///		-1 means NULL ray. 0 means valid
	///
	IdType id;

	vect3d_gpu start_point;
	vect3d_gpu direction_vec;
	vect3d_gpu color;

	//	to make integrator easier
	float fDeltaX, fDeltaY;	// within a PixelIntegrator

	//	In refraction, 
	bool bIsInObj;

	float fEmit;
	float fRefl;
	float fRefr;
	
	//	For putting VPL on GPU only
	vect3d_gpu _hitPoint;
	vect3d_gpu _hitNorm;
};

#endif