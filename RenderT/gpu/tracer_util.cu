
   
__device__
unsigned getThreadInx( unsigned *pxi, unsigned *pyi)
{
	unsigned nAbsTid = blockIdx.x * blockDim.x + threadIdx.x;

	//*pzi = nAbsTid % Z_Dim;
	//*pxi = (nAbsTid - *pzi) / (Y_Dim * Z_Dim);
	//*pyi = (nAbsTid - *pzi) / Z_Dim - *pxi * Y_Dim;
	return nAbsTid;
}

__device__
void fetchVPL(Ray_gpu *pRay, unsigned nAbsOffset, LightGpu *gpuDiffLightsPerLine, float *gpuRdm)
{

	float *pCurrRdm = gpuRdm + nAbsOffset;
	LightGpu *pCurrDiffLights = gpuDiffLightsPerLine + nAbsOffset;

	if(pRay->id != 1)
	{
		if( (*pCurrRdm) <= fVPLPossibility_gpu && 
			(pRay->color.data[0] + pRay->color.data[1] + pRay->color.data[2]) >= fVPLIllmThreshold_gpu &&
			!(pRay->_hitNorm.data[0] == 0 && pRay->_hitNorm.data[1] == 0 && pRay->_hitNorm.data[2] == 0))
		{
			vect3d_gpu blank;

			pCurrDiffLights->eType = DIR_P;
			pCurrDiffLights->_fAttenuate = fVPLAtten_gpu;

			vecCopy_gpu(pCurrDiffLights->_ambientColor, blank);
			vecCopy_gpu(pCurrDiffLights->_diffuseColor, pRay->color);
			vecCopy_gpu(pCurrDiffLights->_specularColor, blank);

			//	DirPoint
			normalize_gpu(pRay->_hitNorm);
			vect3d_gpu epsiVec; vecScale_gpu(pRay->_hitNorm, epsi_gpu, epsiVec);

			vect3d_gpu vVPLPos;
			point2point_gpu(pRay->_hitPoint, epsiVec, vVPLPos);

			vecCopy_gpu(pCurrDiffLights->_dirp_pos, vVPLPos);
			vecCopy_gpu(pCurrDiffLights->_dirp_dir, pRay->_hitNorm);
		}
		else
		{
			pCurrDiffLights->eType = NONE;
		}
	}else
	{
		pCurrDiffLights->eType = NONE;
	}
}

__global__
void synthesizeRays(Ray_gpu *pRays, unsigned nPixelCount, unsigned nTotalRayCount, 
					LightGpu *gpuDiffLightsPerLine, unsigned nMaxRayCountPerTree, float *gpuRdm, unsigned nPassNum, unsigned nPixIntInx)
{
	unsigned x, y;
	unsigned tid = getThreadInx(&x, &y);

	//	GO
	if(tid < nPixelCount)
	{
		unsigned nStartRayInx = tid * nTotalRayCount / nPixelCount;
		Ray_gpu *pRootRay = pRays + nStartRayInx;

		//	for bottom layer rays... VPL
		if(nPassNum == 1 && nPixIntInx == 0)
		{
			unsigned nRayCountInCurrLayer = pow(2.f, ( MaxRayDepth_gpu - 1) * 1.f);
			Ray_gpu *pStartRayInLayer = pRootRay + nRayCountInCurrLayer - 1;

			for(int j = 0; j < nRayCountInCurrLayer; j ++)
			{
				Ray_gpu *pCurrRay = pStartRayInLayer + j;
				fetchVPL(pCurrRay, tid * nMaxRayCountPerTree + nRayCountInCurrLayer - 1 + j, gpuDiffLightsPerLine, gpuRdm);
			}
		}

		for(int i = MaxRayDepth_gpu - 2; i >= 0; i --)
		{

			unsigned nRayCountInCurrLayer = pow(2.f, i * 1.f);
			Ray_gpu *pStartRayInLayer = pRootRay + nRayCountInCurrLayer - 1;

			for(int j = 0; j < nRayCountInCurrLayer; j ++)
			{
				Ray_gpu *pCurrRay = pStartRayInLayer + j;
				if(pCurrRay->id != -1)
				{
					//	refl
					float fRefl = pCurrRay->fRefl;
					if(fRefl > 0)	
					{
						Ray_gpu *pLeftChild = (pRootRay + (nRayCountInCurrLayer - 1 + j + 1) * 2 - 1);
						if(pLeftChild->id != -1)
						{
							vect3d_gpu reflClr;
							vecScale_gpu( pLeftChild->color, fRefl, reflClr);
							point2point_gpu(pCurrRay->color, reflClr, pCurrRay->color);
						}
					}

					//	refr
					float fRefr = pCurrRay->fRefr;
					if(fRefr > 0)	
					{
						Ray_gpu *pRightChild = (pRootRay + (nRayCountInCurrLayer - 1 + j + 1) * 2);
						if(pRightChild->id != -1)
						{
							vect3d_gpu refrClr;
							vecScale_gpu( pRightChild->color, fRefr, refrClr);
							point2point_gpu(pCurrRay->color, refrClr, pCurrRay->color);
						}
					}

					clampColor_gpu(pCurrRay->color);
					
				}//	id check

				//	Put VPL
				//
				if(nPassNum == 1 && nPixIntInx == 0)
				{
					fetchVPL(pCurrRay, tid * nMaxRayCountPerTree + nRayCountInCurrLayer - 1 + j, gpuDiffLightsPerLine, gpuRdm);					
				}//	if for VPL

			}// for
		}//	for
	}
}

__global__
void _computePixels_GPU(PrimGpuObj *gpuObjs, unsigned nCurrObjCount, kd_node_gpu *deviceKdRoot, unsigned *devicePrimObjIdList, unsigned nMaxKdDepth, 
						bool *deviceKdRecBuf, unsigned nMaxKdNodeCount,
						LightGpu *gpuLights, unsigned nCurrLightCount, 
						Ray_gpu *pRays, unsigned nPixelCount, unsigned nTotalRayCount, unsigned nCurrDepth, unsigned nPassNum,
						bool bLtEnabled)
{

	unsigned x, y;
	unsigned tid = getThreadInx(&x, &y);

	if(tid < nPixelCount)
	{
		unsigned nStartRayInx = tid * nTotalRayCount / nPixelCount;
		 
		unsigned nRayCountInCurrLayer = pow(2.f, nCurrDepth * 1.f);
		unsigned nStartRayInxInTree = nRayCountInCurrLayer - 1;

		for(int i = 0; i < nRayCountInCurrLayer; i ++)
		{

 			Ray_gpu *pRay = (pRays + nStartRayInx + nStartRayInxInTree + i);
			if(pRay->id != -1)
			{
				vect3d_gpu	vNorm, texColor;
				float t = 0, fEmit = 0, fReflR = 0, fRefrR = 0, fRefrRK = 0;
	 
				PrimGpuObj *pObj = isHit_gpu(	gpuObjs, nCurrObjCount, pRay, 
												deviceKdRoot, devicePrimObjIdList, nMaxKdDepth, deviceKdRecBuf + tid * nMaxKdNodeCount,
												&t, vNorm, &fEmit, &fReflR, &fRefrR, &fRefrRK, texColor);
				if(pObj && t > 0)  
				{ 
					//	Get hit point position
					vect3d_gpu vHitPoint;
					vecScale_gpu(pRay->direction_vec, t, vHitPoint);
					point2point_gpu(vHitPoint, pRay->start_point, vHitPoint);
 
					//	Recording the geo info for putting VPL 
					//
					vecCopy_gpu(pRay->_hitPoint, vHitPoint);
					vecCopy_gpu(pRay->_hitNorm, vNorm);

 					//	Evaluate Lights illumin.
					// 
					vect3d_gpu retColor;

					if( !bLtEnabled )
					{
 						evaluateIlluminFromLights(deviceKdRoot, devicePrimObjIdList, nMaxKdDepth, deviceKdRecBuf + tid * nMaxKdNodeCount, nMaxKdNodeCount, 
												  gpuObjs, nCurrObjCount, gpuLights, nCurrLightCount, *pRay, pObj, vHitPoint, vNorm, retColor);
					}
					else
					{
						evaluateIlluminByLightcuts(deviceKdRoot, devicePrimObjIdList, nMaxKdDepth, deviceKdRecBuf + tid * nMaxKdNodeCount, nMaxKdNodeCount, 
												  gpuObjs, nCurrObjCount, gpuLights, nCurrLightCount, *pRay, pObj, vHitPoint, vNorm, retColor);
					}
					vecScale_gpu(retColor, fEmit, pRay->color);

					//	Texture 
					if(nPassNum == 1 )
						//!(retColor.data[0] == 0 && retColor.data[1] == 0 && retColor.data[2] == 0) )

					{
						point2point_gpu(pRay->color, texColor, pRay->color);
					}

					pRay->fEmit = fEmit;
					pRay->fRefl = fReflR;
					pRay->fRefr = fRefrR;
					if(nCurrDepth < (MaxRayDepth_gpu - 1))
					{
						//	Reflect - Left child
						if(fReflR > 0)
						{  
							vect3d_gpu vReflectedVec;
							reflectVec_gpu(pRay->direction_vec, vNorm, vReflectedVec);

							Ray_gpu refleRay(vHitPoint, vReflectedVec);
							(pRays + nStartRayInx + (nStartRayInxInTree + i + 1) * 2 - 1)->copy(refleRay);	// TODO: not copy
						}

						//	Refract - Right child
						if(fRefrR > 0)
						{
							vect3d_gpu vRefractVec;
							refractVec_gpu(pRay->direction_vec, vNorm, vRefractVec, fRefrRK);
							normalize_gpu(vRefractVec);

							Ray_gpu refraRay(vHitPoint, vRefractVec, dot_product_gpu(pRay->direction_vec, vNorm) < 0);
							(pRays + nStartRayInx + (nStartRayInxInTree + i + 1) * 2)->copy(refraRay);
						}

					}//	if
				}//	hit or not
				else
				{
					if(nPassNum == 1)
					{
						vecCopy_gpu(pRay->color, AmbiColor_gpu);
					}
				}

			}//	NULL ray?
		}
	}
}
