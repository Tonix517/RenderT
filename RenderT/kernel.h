#ifndef KERNEL_H
#define KERNEL_H

enum KernType {BOX, TRI, GAU, MIT};

class Kernel
{
public:
	//
	//	param x y are all in a [-1, 1]*[-1, 1] square
	//	the returned value is the weight
	//
	virtual float evaluate(float x, float y) = 0;
};

class BoxKernel : public Kernel
{
public:
	float evaluate(float x, float y)
	{
		return 1.f;
	}
};

class TriangleKernel : public Kernel
{
public:
	float evaluate(float x, float y);
};

class GaussianKernel : public Kernel
{
public:
	GaussianKernel(float a);

	float evaluate(float d, float expv);

private:

	float gaussian(float x, float expX);

private:
	float alpha;
	float expX;
	float expY;
};

class MitchellKernel : public Kernel
{
public:
	MitchellKernel(float b, float c)
		: B(b), C(c)
	{ }

	float evaluate(float x, float y);

private:
	float Mitchell1D(float x);

private:
	float B, C;
};

#endif