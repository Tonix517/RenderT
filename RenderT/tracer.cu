#include "tracer.h"
#include "ray.h"
#include "global.h"
#include "vector.h"
#include "consts.h"
#include "texture.h"

#include "gpu/ray_gpu.cu"
#include "gpu/vector_gpu.cu" 
#include "gpu/kd_tree_gpu.h" 
    
#include <cuda_runtime.h>  
#include <vector>
#include <set>
#include <queue>
#include <hash_set>
#include <hash_map>
#include <assert.h> 
using namespace std; 
 
func_brdf_eval Tracer::pEval = NULL;
unsigned Tracer::nCurrAOSinfSize = 0;
float Tracer::aoSinf[MaxAOSinfSize];

bool Tracer::bLCEnabled = false;
lt_node * Tracer::pLcTree = NULL;
unsigned Tracer::nNodeCount = 0;

void Tracer::enableLightcuts(lt_node *pLcRoot, unsigned nCount)
{
	assert(pLcRoot);

	bLCEnabled = true;
	nNodeCount = nCount;
	pLcTree = pLcRoot;
}

void Tracer::disableLightcuts()
{
	bLCEnabled = false;
	nNodeCount = 0;
	pLcTree = NULL;
}

void Tracer::setBRDF(BRDF_TYPE eType)
{
	switch(eType)
	{
	case OGL:
		pEval = OGL_BRDF::evaluate;
		break;

	default: 
		printf("Tracer::setBRDF() : not supported BRDF type!\n");
		break;
	};
}
      
void Tracer::computePixels(PixelIntegrator *pInts, unsigned nCount, unsigned nPassNum)
{
	assert(pEval);

	if(bAOEnabled && nPassNum == 1)
	{
		nCurrAOSinfSize = 2 * fAngleScope / fAngleStep + 1;
		if(nCurrAOSinfSize > MaxAOSinfSize)
		{
			printf("Dude your AO angle step is too small!\n");
			return;
		}

		for(int i = 0; i < nCurrAOSinfSize; i ++)
		{
			aoSinf[i] = sinf( (-fAngleScope + i * fAngleStep) * -PIon180);
		}
	}

	for(unsigned i = 0; i < nCount; i ++)
	{
		if(!scene.bExeSwitch) return;

		computePixel(pInts + i, nPassNum);
	}
}
 
/// 
///		GPU version computePixels();
///

#include "gpu_util.cu"
#include "gpu/geometry_gpu.cu"
#include "gpu/tracer_util.cu"

void Tracer::computePixels_GPU(PixelIntegrator *pInts, unsigned nCount, unsigned nPassNum, std::vector<LightCpu> *vDiffLightVec)
{

	//	Pick one ray from the PixelIntegrator per loop
	//	WinWidth rays per loop
	//
	for(int n = 0; n < nMultiSampleCount; n ++)
	{
		//	Copy rays to Host mem
		for(int m = 0; m < WinWidth; m ++)
		{
			(_hostRays + m * MAX_RAY_COUNT_PER_TREE)->copy(pInts[m].getRays().at(n));
		}
		
		//	Copy rays to Device Mem
		cudaMemcpy(_deviceRays, _hostRays, sizeof(Ray_gpu) * MAX_RAY_COUNT, cudaMemcpyHostToDevice);
		cudaThreadSynchronize();

		//	GPU!
		assert(WinWidth % 256 == 0);

		unsigned c1 = pow(2.f, (nMaxGpuKdDepth + 1) * 1.f);
		nMaxKdNodeCount = c1 - 1;	// in kd-node, nMaxKdNodeCount is actually the max layer inx.

		for(int i = 0; i < MaxRayDepth; i ++)
		{
			_computePixels_GPU<<<WinWidth / 256, 256>>>(gpuObjs, nCurrObjCount, deviceKdRoot, devicePrimObjIdList, nMaxGpuKdDepth, 
														deviceKdRecBuf, nMaxKdNodeCount,
														gpuLights, nCurrLightCount,
														_deviceRays, WinWidth, MAX_RAY_COUNT, i, nPassNum, bLCEnabled); 
			cudaThreadSynchronize();
		}

		//	TODO: AO calculation goes here using _deviceRays
		//		  the results will be used in following synthesizeRays()

		//	recursively calculating ray color, to the root ray
		synthesizeRays<<<WinWidth / 256, 256>>>(_deviceRays, WinWidth, MAX_RAY_COUNT, gpuDiffLightsPerLine, MAX_RAY_COUNT_PER_TREE, gpuRdm, nPassNum, n);
		cudaThreadSynchronize();

		//	Get VPL
		//
		if(nPassNum == 1 && vDiffLightVec)
		{
			cudaMemcpy(cpuDiffLightsPerLine, gpuDiffLightsPerLine, sizeof(LightCpu) * WinWidth * MAX_RAY_COUNT_PER_TREE, cudaMemcpyDeviceToHost);
			cudaThreadSynchronize();

			//	TODO: put it in the vector
			for(int i = 0; i < WinWidth * MAX_RAY_COUNT_PER_TREE; i ++)
			{
				LightCpu *pCurrLight = (cpuDiffLightsPerLine + i);
				if(pCurrLight->eType != NONE)
				{
					vDiffLightVec->push_back(*pCurrLight);
				}
			}
			printf("LightCount %d\t", vDiffLightVec->size());

			//	Update Rdm
			for(int i = 0; i < WinWidth ; i ++)
			for(int j = 0; j < MAX_RAY_COUNT_PER_TREE; j ++)
			{
				*(cpuRdm + j + i * MAX_RAY_COUNT_PER_TREE) = (rand() % 1000000) / 1000000.f;
			}
			cudaMemcpy(gpuRdm, cpuRdm, sizeof(float) * WinWidth * MAX_RAY_COUNT_PER_TREE, cudaMemcpyHostToDevice);
		}

		//	Copy results from Device
		cudaMemcpy(_hostRays, _deviceRays, sizeof(Ray) * MAX_RAY_COUNT, cudaMemcpyDeviceToHost);
		cudaThreadSynchronize();

		//	Copy results to PixelIntegrators
		for(int m = 0; m < WinWidth; m ++)
		{
			vecCopy(pInts[m].getRays().at(n).color, (_hostRays + m * MAX_RAY_COUNT_PER_TREE)->color);
		}

		//	Reset Host & Device rays
		for(int m = 0; m < MAX_RAY_COUNT; m ++)
		{
			(_hostRays + m)->reset();
		}
	}

}

///
///
///

void Tracer::computePixel(PixelIntegrator *pPixInt, unsigned nPassNum)
{
	vector<Ray> &rays = pPixInt->getRays();

	for(unsigned i = 0; i < rays.size(); i ++)
	{
		shootRay(rays[i], 0, nPassNum);
	}
}

void Tracer::shootRay(Ray &ray, unsigned nDepth, unsigned nPassNum)
{
	
	//	1. get hit point
	float t = 0;
	vect3d vNorm;

	vect3d texColor;
	Object *pObj = scene.getHitInfo(ray, &t, vNorm, &texColor);

	//	sit. 1 - hit some object
	if(t > 0 && pObj != NULL)
	{
		vect3d vFinalColor;

		//	Get hit point position
		vect3d vHitPoint;
		vecScale(ray.direction_vec, t, vHitPoint);
		point2point(vHitPoint, ray.start_point, vHitPoint);

		//	Photon-mapping Rendering
		//
		if( bPMEnabled && nPassNum == 2)
		{
			if(photon_map::isHit(ray))
			{
				vect3d photonColor;
				if(photon_map::getPhotonColor(vHitPoint, photonColor))
				{
					vecScale(photonColor, fPMFactor, photonColor);
					point2point(vFinalColor, photonColor, vFinalColor);
				}
			}		
		}
		
		//	Ambient-Occlusion computation
		//
		float fRayAORatio = 1;
		if( bAOEnabled && nPassNum == 1)
		{
			vect3d normalizedNormal;
			vecCopy(normalizedNormal, vNorm);
			normalize(normalizedNormal);	assert( vecLen(normalizedNormal) > 0);

			//	perpenVec1
			vect3d perpenVec;
			if(normalizedNormal[1] != 0)
			{
				//	omit Z axis
				perpenVec[0] = 1;
				perpenVec[1] = - normalizedNormal[0] / normalizedNormal[1];
			}
			else if(normalizedNormal[2] != 0)
			{
				//	omit X axis
				perpenVec[1] = 1;
				perpenVec[2] = - normalizedNormal[1] / normalizedNormal[2];
			}
			else if(normalizedNormal[0] != 0)
			{
				//	omit Y axis
				perpenVec[2] = 1;
				perpenVec[0] = - normalizedNormal[2] / normalizedNormal[0];
			}
			normalize(perpenVec);

			//	perpenVec2
			vect3d perpenVec2;
			cross_product(normalizedNormal, perpenVec, perpenVec2);

			//	Marching ...
			//
			float nHitCount = 0, nTotal = 0;
			for(int i = 0; i < nCurrAOSinfSize; i ++)
			for(int j = 0; j < nCurrAOSinfSize; j ++)
			{
				vect3d currVec;	
				
				currVec[0] = aoSinf[i];
				currVec[1] = aoSinf[j];

				float rem = 1 - currVec[0] * currVec[0] - currVec[1] * currVec[1];
                if(rem >= 0)	
                {
					currVec[2] = sqrt(rem);
                }
                else
                {
                    continue;    
                }

				vect3d tmp0, tmp1, tmp2;
				vecScale(perpenVec , currVec[0], tmp0);
				vecScale(perpenVec2, currVec[1], tmp1);
				vecScale(normalizedNormal, currVec[2], tmp2);
				point2point(tmp0, tmp1, tmp0);
				point2point(tmp0, tmp2, tmp0);

				//	3. Go
				float t; vect3d cNorm;

				//
				//	BUG: in this for loop, the address of currRay will not change.. 
				//		 so the lastVisitingRay of object doesn't change.
				//
				Ray currRay(vHitPoint, tmp0);
				if(scene.getHitInfo(currRay, &t, tmp1, NULL))
				{
					vect3d dist; vecScale(currRay.direction_vec, t, dist);
					float fDist = vecLen(dist);
					if( fDist <= fEffectiveDist)
					{
						nHitCount += (fEffectiveDist - fDist) / fEffectiveDist;
					}
				}				
				nTotal ++;
			}

			fRayAORatio = (nTotal - nHitCount) / nTotal;
		}

		//	Go Tracing
		//
		float fRefle = pObj->getReflectionRatio();
		float fRefra = pObj->getRefractionRatio();

		//	Object color		
		//
		float fEmitR = pObj->getEmissionRatio();
		if(fEmitR > 0&& nDepth < MaxRayDepth)
		{		
			unsigned nHitLightNum = 0;
			vect3d allLightColor;
			
			if( bLCEnabled && (nPassNum == 2))
			{
				evalColorByLightcuts(ray.start_point, vHitPoint, vNorm, &pObj->getMaterial(), allLightColor);
			}
			else
			{
				for(unsigned i = 0; i < scene.getLightNum(); i ++)
				{
					Light *pLight = scene.getLight(i);
					if(pLight->isVisible(vHitPoint, vNorm, pLight))
					{
						vect3d tmpColor;
						pEval(ray.start_point, vHitPoint, vNorm, &pObj->getMaterial(), pLight, tmpColor);
						point2point(allLightColor, tmpColor, allLightColor);
						nHitLightNum ++;
					}
				}
			}
			vecScale(allLightColor, fEmitR, allLightColor);

			// texture color
			if(nHitLightNum > 0 && nPassNum == 1)
			{
				point2point(allLightColor, texColor, allLightColor);				
			}
			point2point(vFinalColor, allLightColor, vFinalColor);
		
		}

		//	Reflection
		//
		if(fRefle > 0 && nDepth < MaxRayDepth - 1)
		{
			vect3d vReflectedVec;
			reflectVec(ray.direction_vec, vNorm, vReflectedVec);

			Ray refleRay(vHitPoint, vReflectedVec);
			shootRay(refleRay, nDepth + 1, nPassNum);

			vecScale(refleRay.color, fRefle, refleRay.color);
			point2point(vFinalColor, refleRay.color, vFinalColor);
		}
		
		//	Refraction 
		//		
		if(fRefra > 0 && nDepth < MaxRayDepth - 1)
		{
			vect3d vRefractVec;
			refractVec(ray.direction_vec, vNorm, vRefractVec, pObj->getRefractionK());
			normalize(vRefractVec);
			Ray refraRay(vHitPoint, vRefractVec, dot_product(ray.direction_vec, vNorm) < 0);
			shootRay(refraRay, nDepth + 1, nPassNum);

			vecScale(refraRay.color, fRefra, refraRay.color);
			point2point(vFinalColor, refraRay.color, vFinalColor);
		}

		//	clamp
		clampColor(vFinalColor);

		//	Ambient Occlusion apply
		//
		if( bAOEnabled && nPassNum == 1 && fRayAORatio < 1.f)
		{
			vecScale(vFinalColor, fRayAORatio * fAOFactor, vFinalColor);	
		}
		
		//	Put VPL
		//
		if(nPassNum == 1)
		{
			//
			//	Yeah baby, the diffuse DIR_P lights are put here....
			//
			float fPosi = (rand() % 1000000) / 1000000.f;
			unsigned nMultiSamplCount = scene.getCamera()->getMultiSamplingCount();
				assert(nMultiSamplCount > 0);
			if( fPosi <= (fVPLPossibility / nMultiSamplCount) && 
				(vFinalColor[0] + vFinalColor[1] + vFinalColor[2]) >= fVPLIllmThreshold)
			{
				vect3d blank;
				
				normalize(vNorm);
				vect3d epsiVec; vecScale(vNorm, epsi, epsiVec);

				vect3d vVPLPos;
				point2point(vHitPoint, epsiVec, vVPLPos);

				DirPointLight *pDiffLight = new DirPointLight(vVPLPos, vNorm, fVPLAtten);
				pDiffLight->setColors(blank, vFinalColor, blank); // TODO
				scene.addDiffuseLight(pDiffLight);
			}
		}

		vecCopy(ray.color, vFinalColor);

		return;
	}

	//	sit. 2 - No hit at all
	if(nPassNum == 1)
	{
		scene.getAmbiColor(ray.color);
	}
}

void Tracer::shootAllPhotons()
{
	unsigned nLightCount = scene.getLightNum();
	unsigned nObjectCount = scene.getObjectNum();

	for(unsigned i = 0; i < nLightCount; i ++)
	{
		Light *pLight = scene.getLight(i);

		for(unsigned j = 0; j < nObjectCount; j ++)
		{
			Object *pObj = scene.getObject(j);
			if( pObj->getReflectionRatio() >= ReflectionRatioThreshold ||
				pObj->getRefractionRatio() >= RefractionRatioThreshold )
			{
				shootPhotons(pLight, pObj);	
			}
		}	
	}
}

void Tracer::shootPhotons(Light *pLight, Object *pObj)
{
	LightType eLightType = pLight->getType();
	
	//	Get bounding sphere from BBox
	BBox *pBBox = pObj->getBBox();
	float fBoundingSphereRad = 0;	vect3d vBoundingSphereCtr;
	pBBox->genBoundingSphereParam(fBoundingSphereRad, vBoundingSphereCtr);
	assert(fBoundingSphereRad > 0);

	std::vector<PhotonRay> rays;

	//	Gen rays
	//
	switch(eLightType)
	{
	case DIR_P:
		{
			vect3d lDir;
			pLight->getDir(lDir);

			vect3d lPos;
			pLight->getPos(lPos);

			vect3d v2ctrVec;
			points2vec(lPos, vBoundingSphereCtr, v2ctrVec);
			normalize(v2ctrVec);

			if(dot_product(lDir, v2ctrVec) < 0)
			{
				return;
			}
		}

	case OMNI_P:
		{
			vect3d lPos;
			pLight->getPos(lPos);

			vect3d v2ctrVec;
			points2vec(lPos, vBoundingSphereCtr, v2ctrVec);
			assert(vecLen(v2ctrVec) > 0);
			float fLenV2ctrVec = vecLen(v2ctrVec);
						
			vect3d v2ctrVecN;	// normalized one
			vecCopy(v2ctrVecN, v2ctrVec);
			normalize(v2ctrVecN);


			//	get the center of the dist
			vect3d distCtr;
			float cosVecLen = sqrt(fLenV2ctrVec * fLenV2ctrVec - fBoundingSphereRad * fBoundingSphereRad);
			float fDistRad = fBoundingSphereRad * cosVecLen / fLenV2ctrVec;
			float light2distCtrVecLen = sqrt(cosVecLen * cosVecLen - fDistRad * fDistRad);
			point2point(lPos, v2ctrVec, distCtr);

			//	get the perpendicular vec on the round disk in the sphere...
			vect3d radVec; 
			if(v2ctrVec[0] != 0)
			{
				//	omit Z-axis value
				radVec[1] = 1;
				radVec[0] = - v2ctrVec[1] / v2ctrVec[0];
			}
			else if(v2ctrVec[1] != 0)
			{
				//	omit X-axis value
				radVec[2] = 1;
				radVec[1] = - v2ctrVec[2] / v2ctrVec[1];
			}
			else if(v2ctrVec[2] != 0)
			{
				//	omit Y-axis value
				radVec[0] = 1;
				radVec[2] = - v2ctrVec[0] / v2ctrVec[2];
			}
			else
			{
				printf("In Tracer::shootPhotons() : there must be something wrong... \n");
				assert(false);
			}
			normalize(radVec);	vecScale(radVec, fDistRad, radVec);

			//	get 3rd vec..
			vect3d perpenRadVec;
			cross_product(v2ctrVecN, radVec, perpenRadVec); 
			normalize(perpenRadVec);	vecScale(perpenRadVec, fDistRad, perpenRadVec);
			
			//
			//	start to sampling
			//
			float nHalfDimCount = fDistRad / PhotonStep;
			for(float i = -nHalfDimCount; i <= nHalfDimCount; i += PhotonStep)
			for(float j = -nHalfDimCount; j <= nHalfDimCount; j += PhotonStep)
			{
				vect3d samplePoint;

				float ri = i * 1.f / nHalfDimCount;
				float rj = j * 1.f / nHalfDimCount;

				vect3d	tmpi; vecScale(radVec, ri, tmpi);
				vect3d	tmpj; vecScale(perpenRadVec, rj, tmpj);

				point2point(distCtr, tmpi, samplePoint);
				point2point(samplePoint, tmpj, samplePoint);

				//	within the disk?
				vect3d ctr2pVec;
				points2vec(distCtr, samplePoint, ctr2pVec);
				if( vecLen(ctr2pVec) > fDistRad )
				{
					continue;
				}

				vect3d toSamplePointVec;
				points2vec(lPos, samplePoint, toSamplePointVec);

				PhotonRay photonRay(lPos, toSamplePointVec);
				
				//	TODO
				vect3d diffColor, tmp1, tmp2;
				pLight->getColors(tmp1, diffColor, tmp2);
				vecCopy(photonRay.thePhoton.color, diffColor);

				rays.push_back(photonRay);
			}
			
		}
		break;
			
	case DIR:
		assert(false);
		break;
	}

	///	Shoot the photon rays!
	for(int i = 0; i < rays.size(); i ++)
	{
		shootPhotonRay(rays[i], 0, pObj);
	}
}

void Tracer::shootPhotonRay(PhotonRay &ray, unsigned nDepth, Object *pObj)
{
	vect3d vNorm; float t;
	Object *pHitOne = scene.getHitInfo(ray, &t, vNorm);
	if(pHitOne != pObj)	
	{
		return;
	}

	//	Get hit point position
	vect3d vHitPoint;
	point2point(vHitPoint, ray.direction_vec, vHitPoint);
	vecScale(vHitPoint, t, vHitPoint);
	point2point(vHitPoint, ray.start_point, vHitPoint);

	//	Put Photon
	if( (	pHitOne->getMaterial().diffColor[0] + 
			pHitOne->getMaterial().diffColor[1] + 
			pHitOne->getMaterial().diffColor[2]) >= DiffuseThreshold &&
			nDepth > 0 && !ray.bIsInObj )
	{
		vecCopy(ray.thePhoton.dir, vNorm);
		vecCopy(ray.thePhoton.pos3d, vHitPoint);
		photon_map::addPhoton(ray.thePhoton);	

		return;
	}

	//
	float fRefle = pHitOne->getReflectionRatio();
	float fRefra = pHitOne->getRefractionRatio();

	//	Reflection
	//
	if(fRefle > 0 && nDepth <= MaxRayDepth)
	{
		vect3d vReflectedVec;
		reflectVec(ray.direction_vec, vNorm, vReflectedVec);

		PhotonRay refleRay(vHitPoint, vReflectedVec);

		const float refleClrFactor = 0.5;
		vect3d diffColorFactor;
		vecScale(pHitOne->getMaterial().diffColor, refleClrFactor, diffColorFactor);
		
		vecScale(ray.thePhoton.color, (1 - refleClrFactor), refleRay.thePhoton.color);
		point2point(refleRay.thePhoton.color, diffColorFactor, refleRay.thePhoton.color);

		shootPhotonRay(refleRay, nDepth + 1);
	}
	
	//	Refraction 
	//		
	if(fRefra > 0 && nDepth <= MaxRayDepth)
	{
		vect3d vRefractVec;
		refractVec(ray.direction_vec, vNorm, vRefractVec, pHitOne->getRefractionK());
		normalize(vRefractVec);

		PhotonRay refraRay(vHitPoint, vRefractVec, dot_product(ray.direction_vec, vNorm) < 0);
		
		if( !refraRay.bIsInObj && ray.bIsInObj)
		{
			//	consider refraction - object internal color
#if 1
			const float refraClrFactor = 0.5;
			vect3d diffColorFactor;
			vecScale(pHitOne->getMaterial().diffColor, refraClrFactor, diffColorFactor);
			
			vecScale(ray.thePhoton.color, (1 - refraClrFactor), refraRay.thePhoton.color);
			point2point(refraRay.thePhoton.color, diffColorFactor, refraRay.thePhoton.color);
			
#else
			vecCopy(refraRay.thePhoton.color, pHitOne->getMaterial().diffColor);
#endif
			vecScale(refraRay.thePhoton.color, fRefra, refraRay.thePhoton.color);
	
		}
		else
		{
			vecCopy(refraRay.thePhoton.color, ray.thePhoton.color);
		}
		
		shootPhotonRay(refraRay, nDepth + 1);
	}
	
}

void Tracer::evalColorByLightcuts(vect3d &vStartPoint, vect3d &vHitPoint, vect3d &vNorm, material *mat, vect3d &rRetColor)
{

	//stdext::hash_set<Light *> retSet;
	std::set<Light *> retSet;
	stdext::hash_map<short, float> estValmap; //for pruning

	//	Choose the CUT !
	//
	std::queue<lt_node *> nodeQueue;
	nodeQueue.push(pLcTree);

#define DBG_LT 1

#if DBG_LT
	clock_t nStart = clock();
#endif

	size_t nCurrCount = 0;
	while((nCurrCount = nodeQueue.size()) > 0)
	{
		for(size_t i = 0; i < nCurrCount; i ++)
		{
			lt_node *pNode = nodeQueue.front();	nodeQueue.pop();

			clock_t nS1 = clock();
			Light *pLight = scene.getLight(pNode->inx_in_light_array);
			float fRel = 0;

			stdext::hash_map<short, float>::iterator iterRet = estValmap.find(pNode->inx_in_light_array);
			if(iterRet != estValmap.end())
			{
				//	already computed
				fRel = iterRet->second;
			}
			else
			{
				fRel = estimateRelevantFactor(vHitPoint, pLight);
			}
			fSubTime1 += clock() - nS1;

			if(fRel >= fLCRelevantFactorThreshold)
			{
				retSet.insert(pLight);
				estValmap.insert( std::pair<short, float>(pNode->inx_in_light_array, fRel));
			}
			if( pNode->l_child_inx_in_tree != -1 )
			{
				nodeQueue.push(pLcTree + pNode->r_child_inx_in_tree);
				nodeQueue.push(pLcTree + pNode->l_child_inx_in_tree);
			}
		}
	}

#if DBG_LT
	clock_t nMid = clock();
#endif

	unsigned nSelectedCount = retSet.size();
	
	//	Update Cut-Ratio Info
	fCurRatio += nSelectedCount * 1.f / scene.getLightNum();
	nInvolvedCount ++;

	if(nSelectedCount > 0)
	{
		//	Evaluate Color
		//
		//stdext::hash_set<Light *>::iterator iter= retSet.begin();
		std::set<Light *>::iterator iter= retSet.begin();
		for(; iter != retSet.end(); iter ++)
		{
			Light *pLight = *iter;
			if(pLight->isVisible(vHitPoint, vNorm, pLight))
			{
				vect3d tmpColor;
				pEval(vStartPoint, vHitPoint, vNorm, mat, pLight, tmpColor);
				point2point(rRetColor, tmpColor, rRetColor);
			}
		}
	}
	
	//	current problem: GPU selected VPL is half the CPU's
	//printf("{%d}", nSelectedCount);

#if DBG_LT
	clock_t nEnd = clock();
	fTraversalTime += nMid - nStart;
	fLightEvalTime += nEnd - nMid;
#endif
	
}