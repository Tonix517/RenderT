#include "light.h"

#include "global.h"

#include <assert.h>

void Light::setColors( vect3d &pAmbColor, 
					   vect3d &pDiffColor,
					   vect3d &pSpecColor)
{
	for(int i = 0; i < 3; i ++)
	{
		_ambientColor[i] = pAmbColor[i];
		_diffuseColor[i] = pDiffColor[i];
		_specularColor[i] = pSpecColor[i];
	}
}

void Light::getColors( vect3d &pAmbColor, 
					   vect3d &pDiffColor,
					   vect3d &pSpecColor)
{
	vecCopy(pAmbColor, _ambientColor);
	vecCopy(pDiffColor, _diffuseColor);
	vecCopy(pSpecColor, _specularColor);
}

///
///
///

OmniPointLight::OmniPointLight(vect3d &pCenter, float fAtten)
	: Light(fAtten)
{
	vecCopy(_pos, pCenter);
}

void OmniPointLight::getPos(vect3d &pPos)
{
	vecCopy(pPos, _pos);
}

void OmniPointLight::getDir(vect3d &)
{
	assert(false);
}

bool OmniPointLight::isVisible(vect3d &pPoint, vect3d &vNorm, Light *pLight)
{
	vect3d dir;
	points2vec(pPoint, _pos, dir);	
	//normalize(dir);

	//	1. Check normal first
	if(dot_product(vNorm , dir) <= 0)
	{
		return false;
	}

	//	2. Check intersection then
	Ray ray(pPoint, dir);	float t; vect3d tmp;
	if(scene.getHitInfo(ray, &t, tmp) == NULL)	// no hit, ok
	{
		return true;
	}
	//	hit, but farther than the light pos?
	return (t > (1.f));
}
///
///
///

DirPointLight::DirPointLight(vect3d &pCenter, vect3d &pDir, float fAtten)
	:Light(fAtten)
{
	_pos[0] = pCenter[0];
	_pos[1] = pCenter[1];
	_pos[2] = pCenter[2];

	_dir[0] = pDir[0];
	_dir[1] = pDir[1];
	_dir[2] = pDir[2];
}

void DirPointLight::getPos(vect3d &pPos)
{
	vecCopy(pPos, _pos);
}

void DirPointLight::getDir(vect3d &pDir)
{
	vecCopy(pDir, _dir);
}

bool DirPointLight::isVisible(vect3d &pPoint, vect3d &vNorm, Light *pLight)
{
	vect3d dir;
	points2vec(pPoint, _pos, dir);	
	
	//	1. Check normal first
	if(dot_product(vNorm , dir) <= 0)
	{
		return false;
	}

	//	
	if(dot_product(dir, _dir) >= 0)
	{
		return false;
	}

	//	2. Check intersection then
	Ray ray(pPoint, dir);	float t; vect3d tmp;
	if(scene.getHitInfo(ray, &t, tmp) == NULL)	// no hit, ok
	{
		return true;
	}
	//	hit, but farther than the light pos?
	return (t > 1.f);
}
///
///
///

DirLight::DirLight(vect3d &pDir, float fAtten)
	:Light(fAtten)
{
	vecCopy(_dir, pDir);
}

void DirLight::getDir(vect3d &pDir)
{
	vecCopy(pDir, _dir);
}

bool DirLight::isVisible(vect3d &pPoint, vect3d &vNorm, Light *pLight)
{

	//	1. Check normal first
	if(dot_product(vNorm , _dir) >= epsi)
	{
		return false;
	}

	//	2. Check intersection then
	vect3d viewDir;	vecScale(_dir, -1, viewDir);
	Ray ray(pPoint, viewDir);
	float t; vect3d tmp;
	if(scene.getHitInfo(ray, &t, tmp) == NULL)	// no hit, ok
	{
		return true;
	}
	return false;
}

//
//
//

SquareAreaLight::SquareAreaLight(	float fAtten, 
									vect3d &pCenter, vect3d &pNormal, vect3d &pHeadVec, float nWidth, float nHeight)
					: Light(fAtten)
					, Square( pCenter, pNormal, pHeadVec, nWidth, nHeight, 0.f, 0.f, 1.f, 1.0)
					, _eSampleType(STRATIFIED)	//	TODO
{
	_fDisDimSize = 1;

	_nWidCount = _nWidth / _fDisDimSize;
	_nHeiCount = _nHeight / _fDisDimSize;

	switch(_eSampleType)
	{
	case STRATIFIED:
		_pSampler = new StratifiedSampler;
		break;

	default:
		printf("Not supported light type...\n");
		break;
	}
}

SquareAreaLight::~SquareAreaLight()
{
	if(_pSampler)
	{
		delete _pSampler;
	}
}

void SquareAreaLight::setDiscretizeDimSize(float fDim)
{
	assert(fDim < _nWidth && fDim < _nHeight);
	_fDisDimSize = fDim;
	_nWidCount = _nWidth / _fDisDimSize;
	_nHeiCount = _nHeight / _fDisDimSize;
}

unsigned SquareAreaLight::getSampleCount()
{
	//	TODO:
	return (_nWidCount) * (_nHeiCount);
}

Light* SquareAreaLight::getNextLightSample(unsigned nInx)
{
	unsigned nSampleCount = getSampleCount();

	assert(nInx < nSampleCount);
	
	float x = 0, y = 0;
	_pSampler->getNextSample(&x, &y);

	int nCurrWidInx = nInx % _nWidCount;
	int nCurrHeiInx = nInx / _nWidCount;

	vect3d ctr;
	//_vCenter + _v2WidthVec * (nCurrHeiInx - nWidCount / 2);
	vecScale(_v2WidthVec, (nCurrHeiInx - _nHeiCount / 2.f) / (_nHeiCount / 2.f), ctr);
	point2point(ctr, _vCenter, ctr);

	//_vCenter + _v2HeightVec * (nCurrHeiInx - nHeiCount / 2);
	vect3d tmp;
	vecScale(_v2HeightVec, (nCurrWidInx - _nWidCount / 2.f) / (_nWidCount / 2.f), tmp);
	point2point(ctr, tmp, ctr);

	//	This infinitesimal movement is necessary, OR there will be serious alias
	vecCopy(tmp, _vNormal);
	vecScale(tmp, epsi, tmp);
	vect3d pCtr;
	point2point(ctr, tmp, pCtr);
	DirPointLight *pPl = new DirPointLight(pCtr, _vNormal, _fAttenuate);
	//OmniPointLight *pPl = new OmniPointLight(pCtr, _fAttenuate);	//	OmniP can make the square visible

	vect3d ambi, diff, spec;
	float factor = 1.f / nSampleCount;
	vecScale(_mat.ambiColor, factor, ambi);
	vecScale(_mat.diffColor, factor, diff);
	vecScale(_mat.specColor, factor, spec);
	pPl->setColors(ambi, diff, spec);

	return pPl;
}