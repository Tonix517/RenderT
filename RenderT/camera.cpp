#include <stdlib.h>
#include "camera.h"
#include "vector.h"
#include "consts.h"
#include <assert.h>

Camera::Camera(vect3d &pCtrPos, vect3d &pUpVec, vect3d &pViewVec, float fNtoF, float fPlaneRatio)
	: _fN2F(fNtoF), _fPlaneRatio(fPlaneRatio), _pSampler(NULL)
{
	assert((fNtoF > 0) && (fPlaneRatio > 0));	

	//	viewing direction
	_dir[0] = pViewVec[0];
	_dir[1] = pViewVec[1];
	_dir[2] = pViewVec[2];
	normalize(_dir);

	//	Center point of the view plane
	_ctrPos[0] = pCtrPos[0];
	_ctrPos[1] = pCtrPos[1];
	_ctrPos[2] = pCtrPos[2];

	//	Up vec of the view plane, with vec-len as 1/2 of view plane height
	_upVec[0] = pUpVec[0];
	_upVec[1] = pUpVec[1];
	_upVec[2] = pUpVec[2];
	normalize(_upVec);
	vecScale(_upVec, WinHeight * _fPlaneRatio * 0.5, _upVec);

	//	right vec of the view plane, with vec-len as 1/2 of view plane width
	cross_product(_dir, _upVec, _rightVec);
	normalize(_rightVec);
	vecScale(_rightVec, WinWidth * _fPlaneRatio * 0.5, _rightVec);

	//	default: no multi-sampling
	_nMultiSamplingCount = 1;
}

void Camera::setSampler(SamplingType eType)
{
	if(_pSampler)
	{
		delete _pSampler;
	}
	switch(eType)
	{
	case STRATIFIED:
		_pSampler = new StratifiedSampler;
		break;

	case LOW_DISC:
		_pSampler = new LowDiscrepancySampler;
		break;

	case BEST_CANDID:
		_pSampler = new BestCandidateSampler;
		break;
	}
}


///
///
///

OrthoCamera::OrthoCamera(vect3d &pCtrPos, vect3d &pUpVec, vect3d &pViewVec, float fNtoF, float fPlaneRatio)
	: Camera(pCtrPos, pUpVec, pViewVec, fNtoF, fPlaneRatio)
{ }

void OrthoCamera::genViewRaysByRow(unsigned nRowCount, PixelIntegrator *pBuf)
{
	// TODO: the following code piece is redundant... 
	//

	assert(pBuf && nRowCount < WinHeight && _pSampler);
	
	for(unsigned i = 0; i < WinWidth; i ++)	//	loop for an entire row
	{
		//	to fine the current primary ray starting point
		//
		vect3d nCurrCtr;
		
		//	right vec first
		vect3d rightVec;
		point2point(rightVec, _rightVec, rightVec);
		vecScale(rightVec, (i - WinWidth/2.f) * ViewPlaneRatio / vecLen(_rightVec), rightVec);

		//	up vec second
		vect3d upVec;
		point2point(upVec, _upVec, upVec);
		vecScale(upVec, (nRowCount - WinHeight/2.f) * ViewPlaneRatio/ vecLen(_upVec), upVec);
		
		point2point(_ctrPos, rightVec, nCurrCtr);
		point2point(nCurrCtr, upVec, nCurrCtr);
		
		//	since we reuse the vector, we have to clear it before next use
		(pBuf + i)->getRays().clear();

		for(unsigned j = 0; j < _nMultiSamplingCount; j ++)
		{
			vect3d startPoint;

			//	Get current sampling start point
			float fdx = 0, fdy = 0;
			_pSampler->getNextSample(&fdx, &fdy);
			assert( (fdx >= -1 && fdx <= 1) && 
					(fdy >= -1 && fdy <= 1) );

			vect3d vDeltaXVec;
			vecScale(_rightVec, ViewPlaneRatio * fSamplingDeltaFactor * fdx, vDeltaXVec);

			vect3d vDeltaYVec;
			vecScale(_upVec, ViewPlaneRatio * fSamplingDeltaFactor * fdy, vDeltaYVec);

			point2point(nCurrCtr, vDeltaXVec, startPoint);			
			point2point(startPoint, vDeltaYVec, startPoint);			

			//	put into PixelIntegrator
			Ray ray(startPoint, _dir);
			ray.fDeltaX = fdx * fSamplingDeltaFactor;
			ray.fDeltaY = fdy * fSamplingDeltaFactor;
			(pBuf + i)->addRay(ray);		
		}
	}
}

///
///
///

PerpCamera::PerpCamera(float fEye2Near, vect3d &pCtrPos, vect3d &pUpVec, vect3d &pViewVec, float fNtoF, float fPlaneRatio)
	: Camera(pCtrPos, pUpVec, pViewVec, fNtoF, fPlaneRatio)
{
	assert(fEye2Near > 0);
	
	//
	vect3d vInverseVec;
	vecScale(_dir, -fEye2Near, vInverseVec);
	point2point(_ctrPos, vInverseVec, _eyePos);
}

void PerpCamera::genViewRaysByRow(unsigned nRowCount, PixelIntegrator *pBuf)
{
	// TODO: the following code piece is redundant... 
	//

	assert(pBuf && nRowCount < WinHeight && _pSampler);
	
	for(unsigned i = 0; i < WinWidth; i ++)	//	loop for an entire row
	{
		//	to fine the current primary ray starting point
		//
		vect3d nCurrCtr;
		
		//	right vec first
		vect3d rightVec;
		point2point(rightVec, _rightVec, rightVec);
		vecScale(rightVec, (i - WinWidth/2.f) * ViewPlaneRatio / vecLen(_rightVec), rightVec);

		//	up vec second
		vect3d upVec;
		point2point(upVec, _upVec, upVec);
		vecScale(upVec, (nRowCount - WinHeight/2.f) * ViewPlaneRatio/ vecLen(_upVec), upVec);
		
		point2point(_ctrPos, rightVec, nCurrCtr);
		point2point(nCurrCtr, upVec, nCurrCtr);
		
		//	since we reuse the vector, we have to clear it before next use
		(pBuf + i)->getRays().clear();

		for(unsigned j = 0; j < _nMultiSamplingCount; j ++)
		{
			vect3d startPoint;

			//	Get current sampling start point
			float fdx = 0, fdy = 0;
			_pSampler->getNextSample(&fdx, &fdy);
			assert( (fdx >= -1 && fdx <= 1) && 
					(fdy >= -1 && fdy <= 1) );

			vect3d vDeltaXVec;
			vecScale(_rightVec, ViewPlaneRatio * fSamplingDeltaFactor * fdx, vDeltaXVec);

			vect3d vDeltaYVec;
			vecScale(_upVec, ViewPlaneRatio * fSamplingDeltaFactor * fdy, vDeltaYVec);

			point2point(nCurrCtr, vDeltaXVec, startPoint);			
			point2point(startPoint, vDeltaYVec, startPoint);			

			//	put into PixelIntegrator
			vect3d viewDir;
			points2vec(_eyePos, startPoint, viewDir);
			normalize(viewDir);
			Ray ray(startPoint, viewDir);
			ray.fDeltaX = fdx * fSamplingDeltaFactor;
			ray.fDeltaY = fdy * fSamplingDeltaFactor;
			(pBuf + i)->addRay(ray);		
		}
	}
}