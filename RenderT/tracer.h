#ifndef TRACER_H
#define TRACER_H

#include "integrator.h"
#include "BxDF.h"
#include "_lightcuts.h"
#include "photon_map.h"

class Tracer
{
public:

	static void setBRDF(BRDF_TYPE eType);

	//
	//	This method will not integrate rays into one single color
	//	by some kernel function. PixelIntegrator::getColor() will.
	//
	static void computePixels(PixelIntegrator *pInts, unsigned nCount, unsigned nPassNum);

	static void computePixels_GPU(PixelIntegrator *pInts, unsigned nCount, unsigned nPassNum, std::vector<LightCpu> *vDiffLightVec = NULL);

	static void shootAllPhotons();

	//	Lightcuts..
	//
	static void enableLightcuts(lt_node *pLcTree, unsigned nNodeCount);
	static void disableLightcuts();

private:

	static void shootPhotons(Light *, Object *);
	static void shootPhotonRay(PhotonRay &, unsigned nDepth = 0, Object *pObj = NULL);
	static void computePixel(PixelIntegrator *pPixInt, unsigned nPassNum);
	
	static void evalColorByLightcuts(vect3d &vStartPoint, vect3d &vHitPoint, vect3d &vNorm, material *mat, vect3d &rRetColor);

private:

	//
	//	The key recursive procedure
	//
	static void shootRay(Ray &ray, unsigned nDepth, unsigned nPassNum);

	static func_brdf_eval pEval;


	//
	//	AO sinf cache
	//
	static const unsigned MaxAOSinfSize = 50;
	static unsigned nCurrAOSinfSize;
	static float aoSinf[MaxAOSinfSize];

	//
	//	Lightcuts Params
	//
	static bool bLCEnabled;
	static unsigned nNodeCount;
	static lt_node *pLcTree;
};
#endif