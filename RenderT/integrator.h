#ifndef INTEGRATOR_H
#define INTEGRATOR_H

#include "ray.h"
#include "kernel.h"
#include "consts.h"
#include <vector>

//	one for per pixel
class PixelIntegrator
{
public:

	//
	static void setKernelType(KernType eType);
	static KernType getKernelType(){	return _eKernType;	}
	static void clean()
	{
		if(_pKern)	
		{
			delete _pKern;
			_pKern = NULL;
		}
	}
	//

	void addRay(Ray &ray);
	void getColor(vect3d &pColor);

	//	TODO: yeah.. i need a better design
	std::vector<Ray>& getRays()
	{
		return _rays;
	}

private:
	
	static KernType _eKernType;
	static Kernel	*_pKern;

	std::vector<Ray> _rays;

};


#endif