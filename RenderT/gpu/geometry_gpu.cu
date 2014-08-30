#include "geometry_gpu.h"


//////////////////////////////////////////////////////////////////////////
__device__
bool isTriangleHit_gpu(	vect3d_gpu vertices[3], Ray_gpu &ray, 
						float *pt, float *pu, float *pv)
{
	//
	//	Real-Time Rendering 2nd, 13.7.2
	//
	vect3d_gpu e1; points2vec_gpu(vertices[0], vertices[1], e1);
	vect3d_gpu e2; points2vec_gpu(vertices[0], vertices[2], e2);

	vect3d_gpu p;  cross_product_gpu(ray.direction_vec, e2, p);
	float a  = dot_product_gpu(e1, p);
	if(a > -epsi_gpu && a < epsi_gpu)
	{
		return false;
	}

	float f  = 1.f / a;
	vect3d_gpu s; points2vec_gpu(vertices[0], ray.start_point, s);
	float u = f * dot_product_gpu(s, p);
	if(u < 0.f || u > 1.f)
	{
		return false;
	}

	vect3d_gpu q;	cross_product_gpu(s, e1, q);
	float v = f * dot_product_gpu(ray.direction_vec, q);
	if(v < 0.f || (u + v) > 1.f)
	{
		return false;
	}

	float t = f * dot_product_gpu(e2, q);
	if(t <= epsi_gpu)
	{
		return false;
	}

	*pt = t;
	if(pu)
	{
		*pu = u;
	}
	if(pv)
	{
		*pv = v;
	}
	
	return true;
}

__device__
bool isTriHit_gpu(PrimGpuObj *pObj, Ray_gpu *ray, vect3d_gpu &tmpNorm, float *tmpT)
{
	
	float u = 0, v = 0;
	if(isTriangleHit_gpu(pObj->_vertices, *ray, tmpT, &u, &v))
	{
		if(pObj->_bSmooth && pObj->_bHasVNorm)
		{
			vect3d_gpu vSmoothNorm;
			point2point_gpu(pObj->_vnormal[1], vSmoothNorm, vSmoothNorm);
			vecScale_gpu(vSmoothNorm, u, vSmoothNorm);

			vect3d_gpu vnorm2;
			point2point_gpu(pObj->_vnormal[2], vnorm2, vnorm2);
			vecScale_gpu(vnorm2, v, vnorm2);

			vect3d_gpu vnorm3;
			point2point_gpu(pObj->_vnormal[0], vnorm3, vnorm3);
			vecScale_gpu(vnorm3, (1 - u - v), vnorm3);

			point2point_gpu(vSmoothNorm, vnorm2, vSmoothNorm);
			point2point_gpu(vSmoothNorm, vnorm3, tmpNorm);

			normalize_gpu(tmpNorm);
		}
		else
		{
			vecCopy_gpu(tmpNorm, pObj->_normal);
		}
		return true;
	}
	return false;
}

__device__
float len2Line(PrimGpuObj *pObj, Ray_gpu &ray, float *pT )
{
	return point2line_gpu( pObj->_ctr, ray.start_point, ray.direction_vec, pT);
}

__device__
bool isSphereHit_gpu(PrimGpuObj *pObj, Ray_gpu *ray, vect3d_gpu &tmpNorm, float *tmpT, vect3d_gpu &texColor)
{
	float t, len;
	len = len2Line(pObj, *ray, &t);	
	if(len > pObj->_fRad)
	{
		return false;
	}

	if( len <= pObj->_fRad)	
	{	
		float d = sqrt(pObj->_fRad * pObj->_fRad - len * len);		
		float r = d / vecLen_gpu(&ray->direction_vec);

		if(ray->bIsInObj)
		{
			t += r;
		}
		else
		{
			t -= r;
		}
	}

	*tmpT = t;

	//	calc normal
	vect3d_gpu pPoint, vec;
	point2point_gpu(pPoint, ray->start_point, pPoint);
	point2point_gpu(vec, ray->direction_vec, vec);	vecScale_gpu(vec, t, vec);
	point2point_gpu(pPoint, vec, pPoint);

	points2vec_gpu(pObj->_ctr, pPoint, tmpNorm);
	normalize_gpu(tmpNorm);

	//	Texture
	if(pObj->pTex != NULL)
	{
		float x = pPoint.data[0] - pObj->_ctr.data[0];
		float y = pPoint.data[1] - pObj->_ctr.data[1];
		float z = pPoint.data[2] - pObj->_ctr.data[2];

		float a = atan2f( x, z );
		float b = acosf( y / pObj->_fRad );

		float u = a / (2 * PI)	;
		float v = b / (1 * PI);
		//
		getTexAt( pObj->pTex, (u + 0.5) * pObj->pTex->width, (v) * pObj->pTex->height, texColor);
	}

	return true;

}

__device__
bool isSquareHit_gpu(PrimGpuObj *pObj, Ray_gpu *ray, vect3d_gpu &tmpNorm, float *tmpT, vect3d_gpu &texColor)
{
	//	The hit point on the plane
	vect3d_gpu op;
	points2vec_gpu(pObj->_vCenter, ray->start_point, op);

	float dn = dot_product_gpu(ray->direction_vec, pObj->_vNormal);
	if(dn == 0.f)
	{
		return false;
	}

	float t = - dot_product_gpu(op, pObj->_vNormal) / dn;
	//	NOTE: since it is a 0-thickness plane, we need this.
	if(t <= epsi_gpu)
	{
		return false;
	}

	//	Get the hit point
	vect3d_gpu vHitPoint;
	vect3d_gpu pView; vecScale_gpu(ray->direction_vec, t, pView);
	point2point_gpu(ray->start_point, pView, vHitPoint);

	vect3d_gpu vHitVec;
	points2vec_gpu(vHitPoint, pObj->_vCenter, vHitVec);

	float dx = dot_product_gpu(vHitVec, pObj->_v2HeightVec) / pow( pObj->_nWidth /2 , 2);
	float dy = dot_product_gpu(vHitVec, pObj->_v2WidthVec) / pow( pObj->_nHeight /2 , 2);
	
	if( fabs(dy) <= 1.f && fabs(dx) <= 1.0f )
	{
		*tmpT = t;
		vecCopy_gpu(tmpNorm, pObj->_vNormal);

		if(pObj->pTex != NULL)
		{
			float fWidInx = (-dx + 1.f) / 2.f * pObj->_nWidth;
			float fHeiInx = (dy + 1.f) / 2.f * pObj->_nHeight;

			switch(pObj->eTexType)
			{
			case STRETCH:
				getTexAt(pObj->pTex, fWidInx * 1.f / pObj->_nWidth * pObj->pTex->width, 
								fHeiInx * 1.f/ pObj->_nHeight * pObj->pTex->height, texColor);
				break;

			case REPEAT:
				getTexAt(pObj->pTex, (unsigned)fWidInx % pObj->pTex->width, 
								(unsigned)fHeiInx % pObj->pTex->height, texColor);
				break;

			case STRAIGHT:
				
				if(fWidInx < pObj->pTex->width && fHeiInx < pObj->pTex->height)
				{
					getTexAt(pObj->pTex, (unsigned)fWidInx, (unsigned)fHeiInx, texColor);
				}
				break;

			};
		}
		return true;
	}
	
	return false;
}

__device__
bool isObjHit_gpu(PrimGpuObj *pObj, Ray_gpu *ray, vect3d_gpu &tmpNorm, float *tmpT, vect3d_gpu &texColor)
{
	switch(pObj->eType)
	{
	case TRI_GPU:
		return isTriHit_gpu(pObj, ray, tmpNorm, tmpT);
		break;
		
	case SQU_GPU:
		return isSquareHit_gpu(pObj, ray, tmpNorm, tmpT, texColor);
		break;
		
	case SPH_GPU:
		return isSphereHit_gpu(pObj, ray, tmpNorm, tmpT, texColor);
		break;

	default:
		return false;
		break;
	}

	return false;
}

//////		BBox
//////

__device__
bool isHitOnPlane(Ray_gpu &ray, BBox &bbox, AxisType eType)
{
	float min = 0, max = 0;
	float start = 0, dir = 0;

	switch(eType)	
	{
	case X_AXIS:
		min = bbox._xmin;
		max = bbox._xmax;
		start = ray.start_point.data[0];
		dir = ray.direction_vec.data[0];
		break;

	case Y_AXIS:
		min = bbox._ymin;
		max = bbox._ymax;
		start = ray.start_point.data[1];
		dir = ray.direction_vec.data[1];
		break;

	case Z_AXIS:
		min = bbox._zmin;
		max = bbox._zmax;
		start = ray.start_point.data[2];
		dir = ray.direction_vec.data[2];
		break;
	}

	//	just between the slabs? yes
	if(start <= max && start > min)
	{
		return true;
	}
	
	//	no marching in this direction?
	if(dir == 0)
	{
		return false;
	}

	float toMinT = (min - start)/dir;
	float toMaxT = (max - start)/dir;

	if( start <= min)
	{
		return toMinT <= toMaxT;
	}
	if( start >= max)
	{
		return toMaxT <= toMinT;
	}

	return false;
}

__device__
bool isBBoxHit_gpu(Ray_gpu &ray, BBox &bbox)
{
	return ( isHitOnPlane(ray, bbox, X_AXIS) && 
			 isHitOnPlane(ray, bbox, Y_AXIS) && 
			 isHitOnPlane(ray, bbox, Z_AXIS) );
}

//////

__device__
PrimGpuObj* isNodeHit_gpu(PrimGpuObj *gpuObjs, Ray_gpu *ray, 
						  kd_node_gpu *pKdRoot, kd_node_gpu *pKdNode, unsigned *devicePrimObjIdList, unsigned nCurrKdDepth, unsigned nMaxKdDepth, 
					      float *pt, vect3d_gpu &vNorm, float *pfEmit, float *pfReflR, float *pfRefrR, float *pfRefrRK,
						  bool *bLeftHit, bool *bRightHit, vect3d_gpu &texColor)
{

	if(	pKdNode->nInxCount > 0 && 
		pKdNode->child0Inx == -1 && pKdNode->child1Inx == -1)	//	Leaf Node containing objects
	{
		
		 *bLeftHit = false;
		 *bRightHit = false;

		vect3d_gpu norm;
		float t = 99999999.0;
		float fEmit = 0, fReflR = 0, fRefrR = 0, fRefrRK = 0;
		PrimGpuObj *pRetObj = NULL;

		vect3d_gpu finalTexColor;

		//
		//	Linear Search
		//
		for(unsigned i = 0; i < pKdNode->nInxCount; i ++)
		{
			int objInx = * (devicePrimObjIdList + pKdNode->nInxStartInx + i);

			vect3d_gpu tmpNorm;
			float tmpT = 0;
			vect3d_gpu currTexColor;

			PrimGpuObj *pObj = gpuObjs + objInx;

			if(isObjHit_gpu(pObj, ray, tmpNorm, &tmpT, currTexColor))
			{
				if(tmpT < t && tmpT > 0)
				{
					t = tmpT;
					vecCopy_gpu(norm, tmpNorm);					
					vecCopy_gpu(finalTexColor, currTexColor);
					fEmit  = pObj->_fEmitRatio; 
					fReflR = pObj->_fReflectionRatio;
					fRefrR = pObj->_fRefractionRatio;
					fRefrRK= pObj->_fRefractionK;

					pRetObj = pObj;
				}//	if
			}//	if
		}//	for

		if( t < 99999999.0 && t > epsi_gpu)
		{
			vecCopy_gpu(vNorm, norm);
			*pt = t;		
			*pfEmit  = fEmit;
			*pfReflR = fReflR; 
			*pfRefrR = fRefrR; 
			*pfRefrRK= fRefrRK;

			vecCopy_gpu(texColor, finalTexColor);
			return pRetObj;
		}	

		return NULL;
	}
	else if( pKdNode->nInxCount == 0 && 
			 !(pKdNode->child0Inx == -1 && pKdNode->child1Inx == -1) )	//	Internode containing no objects
	{

		//assert(nCurrKdDepth < nMaxKdDepth);

		//	Left
		*bLeftHit = false;
		if( pKdNode->child0Inx != -1 )
		{
			kd_node_gpu *pLeft  = pKdRoot + pKdNode->child0Inx;
			
			if(pLeft)
			{
				*bLeftHit =  isBBoxHit_gpu(*ray, pLeft->_bbox);
			}
		}

		//	Right
		*bRightHit = false;
		if( pKdNode->child1Inx != -1 )
		{
			kd_node_gpu *pRight = pKdRoot + pKdNode->child1Inx;
			if(pRight)
			{
				*bRightHit =  isBBoxHit_gpu(*ray, pRight->_bbox);
			}
		}
		return NULL;
	}
	else
	{
		//	sth. wrong with the kd-tree
		//assert(false);
	}
	
	return NULL;
}

__device__
PrimGpuObj* isHit_gpu(PrimGpuObj *gpuObjs, unsigned nCurrObjCount, Ray_gpu *ray, 
					  kd_node_gpu *pKdRoot, unsigned *devicePrimObjIdList, unsigned nMaxKdDepth, bool *pKdRecBuf,
					  float *pt, vect3d_gpu &vNorm, float *pfEmit, float *pfReflR, float *pfRefrR, float *pfRefrRK, vect3d_gpu &texColor)
{
	vect3d_gpu norm;
	float t = 99999999.0;
	float fEmit = 0, fReflR = 0, fRefrR = 0, fRefrRK = 0;
	PrimGpuObj *pRetObj = NULL;
	//vect3d texColor;

	bool bNextLayerHit = true;

	//	NOTE: according to kd_tree.cpp, nMaxKdDepth is actually the index
	*pKdRecBuf = true;
	//printf("{ ");
	for(int i = 0; i <= nMaxKdDepth; i ++)
	{
		if(!bNextLayerHit)
		{
			break;
		}

		//printf("\tL(%d) [ ", i);

		unsigned nCurrNodeCount = pow(2.f, i * 1.f);

		unsigned nStartNodeInKdTree = nCurrNodeCount - 1;
		kd_node_gpu *pStartNode = pKdRoot + nStartNodeInKdTree;

		bNextLayerHit = false;

		int nHitCount = 0;
		for(int j = 0; j < nCurrNodeCount; j ++)
		{
			kd_node_gpu *pCurrNode = (pStartNode + j);

			if(*(pKdRecBuf + nStartNodeInKdTree + j))
			{
				vect3d_gpu tmpNorm;
				float tmpT = 0;
				float fEmit0, fReflR0, fRefrR0, fRefrRK0;
				bool bLeft = false, bRight = false;

				vect3d_gpu tmpTexColor;

				PrimGpuObj *pObj = isNodeHit_gpu(	gpuObjs, ray, 
													pKdRoot, pCurrNode, devicePrimObjIdList, j, nMaxKdDepth, 
													&tmpT, tmpNorm, &fEmit0, &fReflR0, &fRefrR0, &fRefrRK0,
													&bLeft, &bRight, tmpTexColor);
				//printf(" node(%d)", nStartNodeInKdTree + j);
				if(i < nMaxKdDepth)
				{
					*(pKdRecBuf + (nStartNodeInKdTree + j + 1) * 2 - 1) = bLeft;
					*(pKdRecBuf + (nStartNodeInKdTree + j + 1) * 2) = bRight;

					if(bLeft) nHitCount ++;
					if(bRight) nHitCount ++;

					//printf("[%d = %d, %d = %d] ", (nStartNodeInKdTree + j + 1) * 2 - 1, bLeft, (nStartNodeInKdTree + j + 1) * 2, bRight);
				}

				if(pObj && tmpT < t && tmpT > 0)
				{
					//printf(" hit! ");
					t = tmpT;
					vecCopy_gpu(norm, tmpNorm);					
					vecCopy_gpu(texColor, tmpTexColor);
					fEmit  = pObj->_fEmitRatio; 
					fReflR = pObj->_fReflectionRatio;
					fRefrR = pObj->_fRefractionRatio;
					fRefrRK= pObj->_fRefractionK;

					pRetObj = pObj;
				}
			}
			else
			{
				*(pKdRecBuf + (nStartNodeInKdTree + j + 1) * 2 - 1) = false;
				*(pKdRecBuf + (nStartNodeInKdTree + j + 1) * 2) = false;
			}
			

		}//	for		
		//printf("]\n");

		bNextLayerHit = (nHitCount > 0);
	}//	for

	//printf("} \n");

	if( t < 99999999.0 && t > epsi_gpu)
	{
		vecCopy_gpu(vNorm, norm);
		*pt = t;		
		*pfEmit  = fEmit;
		*pfReflR = fReflR; 
		*pfRefrR = fRefrR; 
		*pfRefrRK= fRefrRK;

		//if(pTexColor)
		//{
		//	vecCopy_gpu(*pTexColor, texColor);
		//}
		return pRetObj;
	}	

	return NULL;
}

__device__
bool isLightVisible(kd_node_gpu *deviceKdRoot, unsigned *devicePrimObjIdList, unsigned nMaxKdDepth, bool *deviceKdRecBuf, unsigned nKdRecNodeCount,
					PrimGpuObj *gpuObjs, unsigned nCurrObjCount, vect3d_gpu &vHitPoint, vect3d_gpu &vNorm, LightGpu *pLight)
{
	switch(pLight->eType)
	{
	case OMNI_P:
		{
			vect3d_gpu dir;
			points2vec_gpu(vHitPoint, pLight->_omni_pos, dir);	
			//normalize(dir);

			//	1. Check normal first
			if(dot_product_gpu(vNorm , dir) <= 0)
			{
				return false;
			}

			//	2. Check intersection then
			Ray_gpu ray(vHitPoint, dir);	float t; vect3d_gpu tmp, texColor;
			float t0, t1, t2, t3;
			if(isHit_gpu(gpuObjs, nCurrObjCount, &ray, deviceKdRoot, devicePrimObjIdList, nMaxKdDepth, deviceKdRecBuf,
						 &t, tmp, &t0, &t1, &t2, &t3, texColor) == NULL)	// no hit, ok
			{
				return true;
			}
			//	hit, but farther than the light pos?
			return (t > (1.f));
		}
		break;

	case DIR_P:
		{
			vect3d_gpu dir;
			points2vec_gpu(vHitPoint, pLight->_dirp_pos, dir);	
			
			//	1. Check normal first
			if(dot_product_gpu(vNorm , dir) <= 0)
			{
				return false;
			}

			//	
			if(dot_product_gpu(dir, pLight->_dirp_dir) >= 0)
			{
				return false;
			}

			//	2. Check intersection then
			Ray_gpu ray(vHitPoint, dir);	float t; vect3d_gpu tmp, texColor;
			float t0, t1, t2, t3;
			if(isHit_gpu(gpuObjs, nCurrObjCount, &ray, deviceKdRoot, devicePrimObjIdList, nMaxKdDepth, deviceKdRecBuf,
							&t, tmp, &t0, &t1, &t2, &t3, texColor) == NULL)	// no hit, ok
			{
				return true;
			}
			//	hit, but farther than the light pos?
			return (t > 1.f);
		}
		break;

	case DIR:
		{
			//	1. Check normal first
			if(dot_product_gpu(vNorm , pLight->_dir_dir) >= epsi_gpu)
			{
				return false;
			}

			//	2. Check intersection then
			vect3d_gpu viewDir;	vecScale_gpu(pLight->_dir_dir, -1, viewDir);
			Ray_gpu ray(vHitPoint, viewDir);
			float t; vect3d_gpu tmp, texColor;
			float t0, t1, t2, t3;
			if(isHit_gpu(gpuObjs, nCurrObjCount, &ray, deviceKdRoot, devicePrimObjIdList, nMaxKdDepth, deviceKdRecBuf,
							&t, tmp, &t0, &t1, &t2, &t3, texColor) == NULL)	// no hit, ok
			{
				return true;
			}
			return false;

		}
		break;
	}

	return false;
}


__device__
void clampColor_gpu(vect3d_gpu &pColor)
{
	for(int i = 0; i < 3; i ++)
	{
		if(pColor.data[i] > 1.f) pColor.data[i] = 1.f;
	}
}

__device__
void color_multiply_gpu(vect3d_gpu &color1, vect3d_gpu &color2, vect3d_gpu &rColor)
{
	rColor.data[0] = color1.data[0] * color2.data[0];
	rColor.data[1] = color1.data[1] * color2.data[1];
	rColor.data[2] = color1.data[2] * color2.data[2];
}

__device__
void evalPhong(vect3d_gpu &start_point, vect3d_gpu &vHitPoint, vect3d_gpu &vNorm, PrimGpuObj *pObj, LightGpu *pLight, vect3d_gpu &retColor)
{
	//	Light params
	//
	float fAttenuation = pLight->_fAttenuate;

	//	Get Eye2Point view vec
	vect3d_gpu dir;
	vect3d_gpu v2Eye;
	vect3d_gpu v2EyeNormalized;
	points2vec_gpu(vHitPoint, start_point , v2Eye);			
	vecCopy_gpu(v2EyeNormalized, v2Eye);
	normalize_gpu(v2EyeNormalized);	

	//	Get Point2Light vec
	vect3d_gpu v2Light;
	vect3d_gpu vLightPos;
	vect3d_gpu v2LightNormalized;

	vect3d_gpu pos;
	switch(pLight->eType)
	{
	case OMNI_P:
	case DIR_P:
		if(pLight->eType == OMNI_P)
		{
			vecCopy_gpu(pos, pLight->_omni_pos);
		}
		else if(pLight->eType == DIR_P)
		{
			vecCopy_gpu(pos, pLight->_dirp_pos);
		}
	
		points2vec_gpu(vHitPoint, pos, v2Light);	
		vecCopy_gpu(v2LightNormalized, v2Light);
		normalize_gpu(v2LightNormalized);	// vec. L
		break;

	case DIR:		
		vecScale_gpu(pLight->_dir_dir, -1, pLight->_dir_dir);
		normalize_gpu(pLight->_dir_dir);
		break;

	default:		
		return;
	}

	vect3d_gpu tmp0;	// ambient
	vect3d_gpu tmp1;	// diffuse
	vect3d_gpu tmp2;	// specular

	//	ambient part
	color_multiply_gpu(pLight->_ambientColor, pObj->_mat.ambiColor, tmp0);	

	//	diffuse part		
	float v1 = dot_product_gpu(v2LightNormalized, vNorm);
	float c1 = (v1 > 0) ? v1 : 0;
	color_multiply_gpu(pLight->_diffuseColor, pObj->_mat.diffColor, tmp1);
	vecScale_gpu(tmp1, c1, tmp1);	

	// specular part
	vect3d_gpu vS;
	point2point_gpu(v2Light, v2Eye, vS);	normalize_gpu(vS);
	float v2 = dot_product_gpu(vS, vNorm);
	float c2 = (v2 > 0) ? v2 : 0;
	c2 = pow(c2, pObj->_mat.fShininess);
	color_multiply_gpu(pLight->_specularColor, pObj->_mat.specColor, tmp2);
	vecScale_gpu(tmp2, c2, tmp2);	

	//	add to light sum
	vect3d_gpu tmp;
	point2point_gpu(tmp, tmp0, tmp);	//	adding ambient color
	point2point_gpu(tmp, tmp1, tmp);			//	adding diffuse color
	point2point_gpu(tmp, tmp2, tmp);			//	adding specular color
	vecScale_gpu(tmp, fAttenuation, retColor);		//	calc. attenuation

	//	DirPointLight cosine factor
	if(pLight->eType == DIR_P)
	{
		vect3d_gpu vl2hit;
		vecScale_gpu(v2LightNormalized, -1, vl2hit);
		float fCos = dot_product_gpu(vl2hit, pLight->_dirp_dir) / vecLen_gpu(&pLight->_dirp_dir);
		vecScale_gpu(retColor, (fCos > 0 ? fCos : 0), retColor);
	}

	clampColor_gpu(retColor);
}	

///
///		Lightcuts
///
#include "gpu/_queue_gpu.cu"
#include "gpu/_lightcuts_gpu.cu"

__device__
void evaluateIlluminByLightcuts(kd_node_gpu *deviceKdRoot, unsigned *devicePrimObjIdList, unsigned nMaxKdDepth, bool *deviceKdRecBuf, unsigned nKdRecNodeCount,
							    PrimGpuObj *gpuObjs, unsigned nCurrObjCount, LightGpu *gpuLights, unsigned nCurrLightCount, 
							    Ray_gpu &ray, PrimGpuObj *pObj, vect3d_gpu &vHitPoint, vect3d_gpu &vNorm, vect3d_gpu &retColor)
{
	unsigned nAbsTid = blockIdx.x * blockDim.x + threadIdx.x;

	unsigned nBottomLayerCount = (gpuLightCount % 2 == 0) ? gpuLightCount : (gpuLightCount + 1);
	light_rec *currRecs = gpuRecSpace + nAbsTid * gpuLightCount;
	for(int i = 0; i < WIN_WIDTH; i ++)
	{
		light_rec *pCurrPixelRec = gpuRecSpace + i * gpuLightCount;
		for(int j = 0; j < gpuLightCount; j ++)
		{
			(pCurrPixelRec + j)->fRelFactor = -1;
			(pCurrPixelRec + j)->bHit = false;
		}
	}

	//	Choose the CUT !
	//	
	range_queue_gpu nodeQueue(gpuQueueSpace + (nBottomLayerCount) * nAbsTid, nBottomLayerCount);
	short rootInx = 0;	nodeQueue.push(rootInx);

	size_t nCurrCount = 0;
	while((nCurrCount = nodeQueue.size()) > 0)
	{
		for(size_t i = 0; i < nCurrCount; i ++)
		{
			lt_node_gpu *pNode = gpuLtTree + nodeQueue.front();	nodeQueue.pop();

			LightGpu *pLight = gpuLights + pNode->inx_in_light_array;
			light_rec *pRec = currRecs + pNode->inx_in_light_array;

			float fRel = 0;
			if(pRec->fRelFactor != -1)
			{
				fRel = pRec->fRelFactor;
			}
			else
			{
				fRel = estimateRelevantFactor_gpu(vHitPoint, pLight);
			}

			if(fRel >= gpuRelThreshold)
			{
				pRec->bHit = true;
				pRec->fRelFactor = fRel;
			}
			if( pNode->l_child_inx_in_tree != -1 )
			{
				nodeQueue.push(pNode->r_child_inx_in_tree);
				nodeQueue.push(pNode->l_child_inx_in_tree);
			}
		}
	}

	unsigned nCount = 0;
	for(int i = 0; i < gpuLightCount; i ++)
	{
		if( (currRecs + i)->bHit)
		{
			LightGpu *pLight = gpuLights + i;
			if(isLightVisible(deviceKdRoot, devicePrimObjIdList, nMaxKdDepth, deviceKdRecBuf, nKdRecNodeCount,
								gpuObjs, nCurrObjCount, vHitPoint, vNorm, pLight))
			{
				vect3d_gpu tmpColor;
				evalPhong(ray.start_point, vHitPoint, vNorm, pObj, pLight, tmpColor);
				point2point_gpu(retColor, tmpColor, retColor);			
			}	
			nCount ++;
		}
	}
	
	//printf("{%d}", nCount);

	*(gpuLightStat + nAbsTid) += nCount * 1.f / gpuLightCount;
	*(gpuLightCountStat + nAbsTid) += 1;
}
////////////////////////////

__device__
void evaluateIlluminFromLights(kd_node_gpu *deviceKdRoot, unsigned *devicePrimObjIdList, unsigned nMaxKdDepth, bool *deviceKdRecBuf, unsigned nKdRecNodeCount,
							   PrimGpuObj *gpuObjs, unsigned nCurrObjCount, LightGpu *gpuLights, unsigned nCurrLightCount, 
							   Ray_gpu &ray, PrimGpuObj *pObj, vect3d_gpu &vHitPoint, vect3d_gpu &vNorm, vect3d_gpu &retColor)
{
	for(unsigned i = 0; i < nCurrLightCount; i ++)
	{
		LightGpu *pLight = gpuLights + i;
		if(isLightVisible(deviceKdRoot, devicePrimObjIdList, nMaxKdDepth, deviceKdRecBuf, nKdRecNodeCount,
							gpuObjs, nCurrObjCount, vHitPoint, vNorm, pLight))
		{
			vect3d_gpu tmpColor;
			evalPhong(ray.start_point, vHitPoint, vNorm, pObj, pLight, tmpColor);
			point2point_gpu(retColor, tmpColor, retColor);
		}
	}
}

////////////
