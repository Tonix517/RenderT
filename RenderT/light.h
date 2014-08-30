#ifndef LIGHT_H
#define LIGHT_H

#include "object.h"
#include "sampler.h"
#include "vector.h"

enum LightType {OMNI_P, DIR_P, DIR, SQU_AREA, NONE};


struct LightCpu
{
	LightType eType;

	// common
	float _fAttenuate;

	vect3d _ambientColor;
	vect3d _diffuseColor;
	vect3d _specularColor;

	//	OmniPoint
	vect3d _omni_pos;

	//	DirPoint
	vect3d _dirp_pos;	
	vect3d _dirp_dir;

	//	Dir
	vect3d _dir_dir;
};

//	Abstract Parent class
//
class Light
{
public:
	Light(float fAtten) : _fAttenuate(fAtten){}

	virtual ~Light(){};

	virtual LightType getType() = 0;
	float getAttenuate() { return _fAttenuate; };

	//\	Colors settors & gettors
	//
	void setColors( vect3d &pAmbColor, 
					vect3d &pDiffColor,
					vect3d &pSpecColor );

	void getColors( vect3d &pAmbColor, 
					vect3d &pDiffColor,
					vect3d &pSpecColor );

	///
	///
	virtual void getPos(vect3d &pPos) = 0;
	virtual void getDir(vect3d &pDir) = 0;

	virtual bool isVisible(vect3d &pPoint, vect3d &vNorm, Light *pLight) = 0;

//protected:
public:
		float _fAttenuate;

		vect3d _ambientColor;
		vect3d _diffuseColor;
		vect3d _specularColor;
};

//	All-direction Point Light
//
class OmniPointLight : public Light
{
public:
	OmniPointLight(vect3d &pCenter, float fAtten);
	
	LightType getType(){ return OMNI_P; }

	void getPos(vect3d &pPos);
	void getDir(vect3d &);

	bool isVisible(vect3d &pPoint, vect3d &vNorm, Light *pLight);

//private:
public:
	vect3d _pos;	
};

//	Directional Point Light
//
class DirPointLight : public Light
{
public:
	DirPointLight(vect3d &pCenter, vect3d &pDir, float fAtten);

	LightType getType(){ return DIR_P; }

	void getPos(vect3d &pPos);
	void getDir(vect3d &pDir);

	bool isVisible(vect3d &pPoint, vect3d &vNorm, Light *pLight);

//private:
public:
	vect3d _pos;	
	vect3d _dir;
};

//	Directional Light
//
class DirLight : public Light
{
public:
	DirLight(vect3d &pDir, float fAtten);

	LightType getType(){ return DIR; }

	void getDir(vect3d &pDir);

	bool isVisible(vect3d &pPoint, vect3d &vNorm, Light *pLight);

private:
	void getPos(vect3d &pPos){}

//private:
public:
	vect3d _dir;
};

///
///		Area Light
///		NOTE: could be textured
///
class SquareAreaLight : public Light, public Square
{
public:
	SquareAreaLight(float fAtten, 
					vect3d &pCenter, vect3d &pNormal, vect3d &pHeadVec, float nWidth, float nHeight);

	~SquareAreaLight();

	//
	SamplingType getSampleType(){ return _eSampleType; }

	LightType getType(){ return SQU_AREA; }

	void setDiscretizeDimSize(float fDim);
	unsigned getSampleCount();

	Light* getNextLightSample(unsigned nInx);

private:
	void getPos(vect3d &){};
	void getDir(vect3d &){};
	//	defaultly, it will be discretized.
	bool isVisible(vect3d &,vect3d &,Light *){return false;}

private:
	SamplingType _eSampleType;
	Sampler	*_pSampler;

	float _fDisDimSize;
	int _nWidCount;
	int _nHeiCount;
};


#endif
