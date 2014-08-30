#ifndef BXDF_H
#define BXDF_H

#include "object.h"
#include "light.h"

enum BRDF_TYPE {OGL};
typedef void (*func_brdf_eval)(vect3d &pEyePos, vect3d &pPoint, vect3d &pNormal, material *pMat, Light *pLight, vect3d &pRetColor);

//
//		class BRDF
//
class BRDF
{
public:
	
	static void evaluate(float *pEyePos, float *pPoint, float *pNormal, material *pMat, Light *pLight, float *pRetColor);
};

//
//		class OGL_BRDF : public BRDF
//
class OGL_BRDF : public BRDF
{
public:

	static void evaluate(vect3d &pEyePos, vect3d &pPoint, vect3d &pNormal, material *pMat, Light *pLight, vect3d &pRetColor);

private:

	static void color_multiply(vect3d &color1, vect3d &color2, vect3d &rColor);

};

#endif