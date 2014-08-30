#ifndef GEOMETRY_GPU_H
#define GEOMETRY_GPU_H

#include "object.h"
#include "light.h"
#include "gpu/vector_gpu.cu"

enum GpuObjType {TRI_GPU, SQU_GPU, SPH_GPU, NONE_GPU};

///

struct material_gpu
{
	material_gpu()
	{
		for(int i = 0; i < 3; i ++)
		{
			specColor.data[i] = 0;
			diffColor.data[i] = 0;
			ambiColor.data[i] = 0;
		}

		fShininess = 1.0f;
	}

	vect3d_gpu specColor;
	vect3d_gpu diffColor;
	vect3d_gpu ambiColor;

	float fShininess;
};

extern struct texture_s_gpu;
///		
///
struct PrimGpuObj
{
	PrimGpuObj()
	{
		eType = NONE_GPU;
		nId = -1;
		pTex = NULL;
	}

/////////////////////////////////////////

	//	common
	float _fReflectionRatio;
	float _fRefractionRatio;
	float _fRefractionK;
	float _fEmitRatio;
	material_gpu _mat;	

	///
	///		NOTE: I know... I also hate this.. but CUDA forces me to do this
	///
	
	//	for GPU use
	GpuObjType	eType;
	int nId;
	texture_s_gpu *pTex;
	TexMapType eTexType;

	//	Triangle
	//
	vect3d_gpu	_vertices[3];
	vect3d_gpu	_normal;
	vect3d_gpu	_vnormal[3];
	bool _bSmooth;
	bool _bHasVNorm;

	//	Sphere
	//
	float _fRad;
	vect3d_gpu _ctr;

	//	Square
	//
	vect3d_gpu _vNormal;
	vect3d_gpu _vWidthVec;
	vect3d_gpu _vCenter;
	float _nWidth;
	float _nHeight;	
	vect3d_gpu _v2HeightVec;
	vect3d_gpu _v2WidthVec;
	float a, b, c, d; // ax + by + cz + d = 0

};

////
struct PrimGpuObj_host
{
	PrimGpuObj_host()
	{
		eType = NONE_GPU;
		nId = -1;
		pTex = NULL;
	}

/////////////////////////////////////////

	//	common
	float _fReflectionRatio;
	float _fRefractionRatio;
	float _fRefractionK;
	float _fEmitRatio;
	material _mat;	

	///
	///		NOTE: I know... I also hate this.. but CUDA forces me to do this
	///
	
	//	for GPU use
	GpuObjType	eType;
	int nId;
	texture_s_gpu *pTex;
	TexMapType eTexType;

	//	Triangle
	//
	vect3d	_vertices[3];
	vect3d	_normal;
	vect3d	_vnormal[3];
	bool _bSmooth;
	bool _bHasVNorm;

	//	Sphere
	//
	float _fRad;
	vect3d _ctr;

	//	Square
	//
	vect3d _vNormal;
	vect3d _vWidthVec;
	vect3d _vCenter;
	float _nWidth;
	float _nHeight;	
	vect3d _v2HeightVec;
	vect3d _v2WidthVec;
	float a, b, c, d; // ax + by + cz + d = 0

};

////

struct LightGpu
{
	LightType eType;

	// common
	float _fAttenuate;

	vect3d_gpu _ambientColor;
	vect3d_gpu _diffuseColor;
	vect3d_gpu _specularColor;

	//	OmniPoint
	vect3d_gpu _omni_pos;

	//	DirPoint
	vect3d_gpu _dirp_pos;	
	vect3d_gpu _dirp_dir;

	//	Dir
	vect3d_gpu _dir_dir;
};



////


#endif