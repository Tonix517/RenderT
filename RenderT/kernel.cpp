#include "kernel.h"

#include <assert.h>
#include <math.h>
#define max(x,y) ( ((x) > (y)) ? (x) : (y))

float TriangleKernel::evaluate(float x, float y)
{
	assert(fabs(x) <= 1.f && fabs(y) <= 1.f);
	return max(0.f, 1.f - fabs(x)) * max(0.f, 1.f - fabs(y));
}

GaussianKernel::GaussianKernel(float a)
{
	alpha = a;
	expX = expf(-alpha * 1);
	expY = expf(-alpha * 1);
}

float GaussianKernel::gaussian(float d, float expv)
{
	return max(0.f, float(expf(-alpha * d * d) - expv));
}

float GaussianKernel::evaluate(float x, float y)
{
	return gaussian(x, expX) * gaussian(y, expY);
}

float MitchellKernel::Mitchell1D(float x)
{
	x = fabsf(2.f * x);
	if (x > 1.f)
		return ((-B - 6*C) * x*x*x + (6*B + 30*C) * x*x +
			(-12*B - 48*C) * x + (8*B + 24*C)) * (1.f/6.f);
	else
		return ((12 - 9*B - 6*C) * x*x*x +
			(-18 + 12*B + 6*C) * x*x +
			(6 - 2*B)) * (1.f/6.f);
}

float MitchellKernel::evaluate(float x, float y)
{
	return	Mitchell1D(x) *	Mitchell1D(y);
}