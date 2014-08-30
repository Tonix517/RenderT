#include "integrator.h"
#include "vector.h"
#include "consts.h"
#include "ray.h"
#include <assert.h>
#include <math.h>

KernType PixelIntegrator::_eKernType = BOX;
Kernel	*PixelIntegrator::_pKern = NULL;

void PixelIntegrator::setKernelType(KernType eType)
{
	if(_pKern)
	{
		delete _pKern;
		_pKern = NULL;
	}

	switch(eType)
	{
	case BOX:
		_pKern = new BoxKernel;
		break;

	case TRI:
		_pKern = new TriangleKernel;
		break;

	case GAU:
		_pKern = new GaussianKernel(2);
		break;

	case MIT:
		_pKern = new MitchellKernel(1.f/3.f, 1.f/3.f);
		break;

	default:
		printf("~ PixelIntegrator ctor: KernType not valid.\n");
		break;
	}
}

void PixelIntegrator::addRay(Ray &ray)
{
	_rays.push_back(ray);
};

void PixelIntegrator::getColor(vect3d &pColor)
{

	//	TODO: integrate by kernel
	unsigned nCount = _rays.size();
	float fTotalWeight = 0;
	for(unsigned i = 0; i < nCount; i ++)
	{
		float fWeight = _pKern->evaluate(_rays[i].fDeltaX, _rays[i].fDeltaY);
		fTotalWeight += fWeight;

		pColor[0] += _rays[i].color[0] * fWeight;
		pColor[1] += _rays[i].color[1] * fWeight;
		pColor[2] += _rays[i].color[2] * fWeight;
	}
	vecScale(pColor, 1.f / fTotalWeight, pColor);

	clampColor(pColor);
}