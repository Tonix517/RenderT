#include "BxDF.h"
#include "vector.h"

#include <stdio.h>
#include <memory.h>
#include <assert.h>
#include <math.h>

//
//
//

void OGL_BRDF::color_multiply(vect3d &color1, vect3d &color2, vect3d &rColor)
{
	rColor[0] = color1[0] * color2[0];
	rColor[1] = color1[1] * color2[1];
	rColor[2] = color1[2] * color2[2];
}

void OGL_BRDF::evaluate(vect3d &pEyePos, vect3d &pPoint, vect3d &pNormal, material *pMat, Light *pLight, vect3d &pRetColor)
{
	assert(pMat && pLight);

	//	Light params
	//
	float fAttenuation = 1;
	fAttenuation = pLight->getAttenuate();

	//	Get Eye2Point view vec
	vect3d dir;
	vect3d v2Eye;
	vect3d v2EyeNormalized;
	points2vec(pPoint, pEyePos , v2Eye);			
	vecCopy(v2EyeNormalized, v2Eye);
	normalize(v2EyeNormalized);	

	//	Get Point2Light vec
	vect3d v2Light;
	vect3d vLightPos;
	vect3d v2LightNormalized;

	switch(pLight->getType())
	{
	case OMNI_P:
	case DIR_P:	//	visibility is checked before calling this			
		pLight->getPos(vLightPos);				
		points2vec(pPoint, vLightPos, v2Light);	
		vecCopy(v2LightNormalized, v2Light);
		normalize(v2LightNormalized);	// vec. L
		break;

	case DIR:		
		pLight->getDir(dir);
		vecScale(dir, -1, dir);
		normalize(dir);
		break;

	default:
		printf("[ERROR] OGL_BRDF : not supported light type !\n");
		return;
	}

	vect3d vLightAmbi, vLightDiff, vLightSpec;
	pLight->getColors(vLightAmbi, vLightDiff, vLightSpec);

	vect3d tmp0;	// ambient
	vect3d tmp1;	// diffuse
	vect3d tmp2;	// specular

	//	ambient part
	color_multiply(vLightAmbi, pMat->ambiColor, tmp0);	

	//	diffuse part		
	float v1 = dot_product(v2LightNormalized, pNormal);
	float c1 = (v1 > 0) ? v1 : 0;
	color_multiply(vLightDiff, pMat->diffColor, tmp1);
	vecScale(tmp1, c1, tmp1);	

	// specular part
	vect3d vS;
	point2point(v2Light, v2Eye, vS);	normalize(vS);
	float v2 = dot_product(vS, pNormal);
	float c2 = (v2 > 0) ? v2 : 0;
	c2 = pow(c2, pMat->fShininess);
	color_multiply(vLightSpec, pMat->specColor, tmp2);
	vecScale(tmp2, c2, tmp2);	

	//	add to light sum
	vect3d tmp;
	point2point(tmp, tmp0, tmp);	//	adding ambient color
	point2point(tmp, tmp1, tmp);			//	adding diffuse color
	point2point(tmp, tmp2, tmp);			//	adding specular color
	vecScale(tmp, fAttenuation, tmp);		//	calc. attenuation

	vecCopy(pRetColor, tmp);

	//	DirPointLight cosine factor
	if(pLight->getType() == DIR_P)
	{
		vect3d vl2hit;
		vect3d vDPdir;
		pLight->getDir(vDPdir);

		vecScale(v2LightNormalized, -1, vl2hit);
		float fCos = dot_product(vl2hit, vDPdir) / vecLen(vDPdir);
		vecScale(pRetColor, (fCos > 0 ? fCos : 0), pRetColor);
	}


	clampColor(pRetColor);
}