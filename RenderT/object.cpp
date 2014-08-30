#include "object.h"
#include "vector.h"
#include "consts.h"
#include <assert.h>
#include <math.h>
#include <memory.h>
#include <stdio.h>
///
///		class Object
///

Object::Object(float fReflR, float fRefrR, float fRefrK, float fEmitR)
	: _fRefractionK(fRefrK)
	, _fRefractionRatio(fRefrR)
	, _fReflectionRatio(fReflR)
	, _fEmitRatio(fEmitR)
	, _id(nCurrObj++)
	, _nLastVisitingRay(-1)
	, _tex(NULL)
	, _eTexMapType(STRETCH)
{
}

void Object::setMaterial(vect3d &specColor, vect3d &diffColor, vect3d &ambiColor, float fShininess)
{
	vecCopy(_mat.specColor, specColor);
	vecCopy(_mat.diffColor, diffColor);
	vecCopy(_mat.ambiColor, ambiColor);
	_mat.fShininess = fShininess;
}

void Object::setTexture(int texId, TexMapType eType)
{
	assert(texId != -1);

	_eTexMapType = eType;
	_tex = TextureManager::getTexture(texId);
}


///
///		class Triangle
///
Triangle::Triangle(	float vVertices[3][3], float *facetNormal, 
					float fReflR, float fRefrR, float fRefrK, float fEmitR)
	: PrimaryObject(fReflR, fRefrR, fRefrK, fEmitR)
	, _bSmooth(false)
	, _bHasVNorm(false)
{
	assert(vVertices && facetNormal);

	for(int i = 0; i < 3; i ++)
	{
		_vertices[i][0] = vVertices[i][0];
		_vertices[i][1] = vVertices[i][1];
		_vertices[i][2] = vVertices[i][2];		
		_normal[i] = facetNormal[i];

		_vnormal[i][0] = 0;
		_vnormal[i][1] = 0;
		_vnormal[i][2] = 0;
	}

	updateBBox();
}

bool Triangle::isHit(Ray &ray, vect3d &pNormal, float *pt, vect3d *pTexColor)
{
	//	This will slow down the performance 
	//
	//if(!_bbox.isHit(ray))
	//{
	//	return false;
	//}

	float u = 0, v = 0;
	if(isTriangleHit(_vertices, ray, pt, &u, &v))
	{
		if(_bSmooth && _bHasVNorm)
		{
			vect3d vSmoothNorm;
			point2point(_vnormal[1], vSmoothNorm, vSmoothNorm);
			vecScale(vSmoothNorm, u, vSmoothNorm);

			vect3d vnorm2;
			point2point(_vnormal[2], vnorm2, vnorm2);
			vecScale(vnorm2, v, vnorm2);

			vect3d vnorm3;
			point2point(_vnormal[0], vnorm3, vnorm3);
			vecScale(vnorm3, (1 - u - v), vnorm3);

			point2point(vSmoothNorm, vnorm2, vSmoothNorm);
			point2point(vSmoothNorm, vnorm3, pNormal);

			normalize(pNormal);
		}
		else
		{
			vecCopy(pNormal, _normal);
		}
		return true;
	}
	return false;
}

void Triangle::setVNorm(float vnorm[3][3])
{
	_bHasVNorm = true;

	for(int i = 0; i < 3; i ++)
	{
		_vnormal[i][0] = vnorm[i][0];
		_vnormal[i][1] = vnorm[i][1];
		_vnormal[i][2] = vnorm[i][2];
	}
}

void Triangle::updateBBox()
{
	float xmin = 99999999.f, xmax = -99999999.f;
	float ymin = 99999999.f, ymax = -99999999.f;
	float zmin = 99999999.f, zmax = -99999999.f;

	for(int i = 0; i < 3; i ++)
	{
		if(_vertices[i][0] < xmin)	xmin = _vertices[i][0];
		if(_vertices[i][0] > xmax)	xmax = _vertices[i][0];

		if(_vertices[i][1] < ymin)	ymin = _vertices[i][1];
		if(_vertices[i][1] > ymax)	ymax = _vertices[i][1];

		if(_vertices[i][2] < zmin)	zmin = _vertices[i][2];
		if(_vertices[i][2] > zmax)	zmax = _vertices[i][2];
	}

	_bbox.setDim(xmin, xmax, ymin, ymax, zmin, zmax);
}

///
///		class Sphere
///

Sphere::Sphere(float fRadius, vect3d &pCenterPos, float fReflR, float fRefrR, float fRefrK, float fEmitR)
	:PrimaryObject(fReflR, fRefrR, fRefrK, fEmitR)
	, _fRad(fRadius)
{
	_ctr[0] = pCenterPos[0];
	_ctr[1] = pCenterPos[1];
	_ctr[2] = pCenterPos[2];

	updateBBox();
}

bool Sphere::isHit(	Ray &ray, vect3d &pNormal, float *pt, vect3d *pTexColor)
{
	if(!_bbox.isHit(ray))
	{
		return false;
	}

	float t, len;
	len = len2Line(ray, &t);	
	if(len > _fRad)
	{
		return false;
	}

	if( len <= _fRad)	
	{	
		float d = sqrt(_fRad * _fRad - len * len);		
		float r = d / vecLen(ray.direction_vec);

		if(ray.bIsInObj)
		{
			t += r;
		}
		else
		{
			t -= r;
		}
	}

	*pt = t;

	//	calc normal
	vect3d pPoint, vec;
	point2point(pPoint, ray.start_point, pPoint);
	point2point(vec, ray.direction_vec, vec);	vecScale(vec, t, vec);
	point2point(pPoint, vec, pPoint);

	points2vec(_ctr, pPoint, pNormal);
	normalize(pNormal);

	//	Texture
	if(_tex != NULL && pTexColor != NULL)
	{
		float x = pPoint[0] - _ctr[0];
		float y = pPoint[1] - _ctr[1];
		float z = pPoint[2] - _ctr[2];

		float a = atan2f( x, z );
		float b = acosf( y / _fRad );

		float u = a / (2 * PI)	;
		float v = b / (1 * PI);
		//
		_tex->getTexAt( (u + 0.5) * _tex->nWidth, (v) * _tex->nHeight, *pTexColor);
	}

	return true;

}

void Sphere::updateBBox()
{
	_bbox.setDim( _ctr[0] - _fRad, _ctr[0] + _fRad, 
				  _ctr[1] - _fRad, _ctr[1] + _fRad, 
				  _ctr[2] - _fRad, _ctr[2] + _fRad);
}

void Sphere::setTexture(int texId, TexMapType eType)
{
	assert(texId != -1);

	_eTexMapType = eType;

	if(eType == STRETCH)
	{
		texture_s *pTex = TextureManager::getTexture(texId);
		assert(pTex);

		//	Calc. the expected Width & Height
		unsigned nSphMapHeight = _fRad * PI;
		unsigned nSphMapWidth = 2 * nSphMapHeight;

		//	choose the minimum LARGER mipmap.. 
		_tex = pTex;
		while( (pTex->pNextMip != NULL) && 
			   (pTex->nWidth > nSphMapWidth && pTex->nHeight > nSphMapHeight) )
		{
			_tex = pTex;	
			pTex = pTex->pNextMip;		
		}
	}
	else
	{
		_tex = TextureManager::getTexture(texId);
	}

}

float Sphere::len2Line(	Ray &ray, float *pT )
{
	return point2line( _ctr, ray.start_point, ray.direction_vec, pT);
}

///
///
///

//////// Square Impl. //////////

Square::Square(	vect3d &pCenter, 
			    vect3d &pNormal,
				vect3d &pHeadVec, 
				float nWidth, 
				float nHeight,
				float fReflR, float fRefrR, float fRefrK, float fEmitR)
	:PrimaryObject(fReflR, fRefrR, fRefrK, fEmitR)
{
		
	vecCopy(_vCenter, pCenter);
	vecCopy(_vNormal, pNormal);	
	vecCopy(_vWidthVec, pHeadVec);	

	_nWidth = nWidth;
	_nHeight = nHeight;

	vecScale(_vWidthVec, nHeight/2, _v2WidthVec);

	vect3d v2HeightVec;
	cross_product(_vWidthVec, _vNormal, v2HeightVec);
	normalize(v2HeightVec);
	vecScale(v2HeightVec, nWidth/2, _v2HeightVec);

	a = _vNormal[0]; b = _vNormal[1]; c = _vNormal[2];
	d = (-1) * (a * _vCenter[0] + b * _vCenter[1] + c * _vCenter[2]);

	updateBBox();
}
		
bool Square::isHit(Ray &ray, vect3d &pNormal, float *pt, vect3d *pTexColor)
{
	if(!_bbox.isHit(ray))
	{
		return false;
	}

	//	The hit point on the plane
	vect3d op;
	points2vec(_vCenter, ray.start_point, op);

	float dn = dot_product(ray.direction_vec, _vNormal);
	if(dn == 0.f)
	{
		return false;
	}

	float t = - dot_product(op, _vNormal) / dn;
	//	NOTE: since it is a 0-thickness plane, we need this.
	if(t <= epsi)
	{
		return false;
	}

	//	Get the hit point
	vect3d vHitPoint;
	vect3d pView; vecScale(ray.direction_vec, t, pView);
	point2point(ray.start_point, pView, vHitPoint);

	vect3d vHitVec;
	points2vec(vHitPoint, _vCenter, vHitVec);

	float dx = dot_product(vHitVec, _v2HeightVec) / pow( _nWidth /2 , 2);
	float dy = dot_product(vHitVec, _v2WidthVec) / pow( _nHeight /2 , 2);
	
	if( fabs(dy) <= 1.f && fabs(dx) <= 1.0f )
	{
		*pt = t;
		vecCopy(pNormal, _vNormal);

		if(_tex != NULL && pTexColor != NULL)
		{
			float fWidInx = (-dx + 1.f) / 2.f * _nWidth;
			float fHeiInx = (dy + 1.f) / 2.f * _nHeight;

			switch(_eTexMapType)
			{
			case STRETCH:
				_tex->getTexAt( fWidInx / _nWidth * _tex->nWidth, 
								fHeiInx / _nHeight * _tex->nHeight, *pTexColor);
				break;

			case REPEAT:
				_tex->getTexAt( (unsigned)fWidInx % _tex->nWidth, 
								(unsigned)fHeiInx % _tex->nHeight, *pTexColor);
				break;

			case STRAIGHT:
				
				if(fWidInx < _tex->nWidth && fHeiInx < _tex->nHeight)
				{
					_tex->getTexAt( (unsigned)fWidInx, (unsigned)fHeiInx, *pTexColor);
				}
				break;

			};
		}
		return true;
	}
	
	return false;
}

void Square::setTexture(int texId, TexMapType eType)
{
	_eTexMapType = eType;

	if(eType == STRETCH)
	{
		//	find the right mipmap
		texture_s *pTex = TextureManager::getTexture(texId);
		assert(pTex);

		//	choose the minimum LARGER mipmap.. 
		_tex = pTex;
		while( (pTex->pNextMip != NULL) && 
			   (pTex->nWidth > _nWidth && pTex->nHeight > _nHeight) )
		{
			_tex = pTex;
			pTex = pTex->pNextMip;		
		}
	}
	else
	{
		_tex = TextureManager::getTexture(texId);
	}
}

void Square::updateBBox()
{
	//	TODO: I believe there exists a better impl.

	//	Get 4 points

	float vertices[4][3] = {0};
	for(int i = 0; i < 3; i ++)
	{
		vertices[0][i] = _vCenter[i] + _v2HeightVec[i] + _v2WidthVec[i];
		vertices[1][i] = _vCenter[i] + _v2HeightVec[i] - _v2WidthVec[i];
		vertices[2][i] = _vCenter[i] - _v2HeightVec[i] + _v2WidthVec[i];
		vertices[3][i] = _vCenter[i] - _v2HeightVec[i] - _v2WidthVec[i];
	}

	//

	float xmin = 99999999.f, xmax = -99999999.f;
	float ymin = 99999999.f, ymax = -99999999.f;
	float zmin = 99999999.f, zmax = -99999999.f;

	for(int i = 0; i < 4; i ++)
	{
		if(vertices[i][0] < xmin)	xmin = vertices[i][0];
		if(vertices[i][0] > xmax)	xmax = vertices[i][0];

		if(vertices[i][1] < ymin)	ymin = vertices[i][1];
		if(vertices[i][1] > ymax)	ymax = vertices[i][1];

		if(vertices[i][2] < zmin)	zmin = vertices[i][2];
		if(vertices[i][2] > zmax)	zmax = vertices[i][2];
	}

	_bbox.setDim(xmin, xmax, ymin, ymax, zmin, zmax);
}

//////// Cube Impl. //////////

Cube::Cube(		float fLen, float fWidth, float fHeight,	 
				vect3d &pCenterPos,
				vect3d &pVerticalVec,
				vect3d &pHorizonVec,				
				float fReflR, float fRefrR, float fRefrK, float fEmitR)
	:PrimaryObject(fReflR, fRefrR, fRefrK, fEmitR)
{
	assert( (fLen > 0) && (fWidth > 0) && (fHeight > 0) );
	
	///
	///	Since there's no ObjObject in GPU, and I want to make primaryObj id the same as the 
	///	GPU primaryObj array index, I do this. This should not impact the current functional 
	///	code.
	///
	nCurrObj --;

	_fLength = fLen;	
	_fWidth = fWidth;	
	_fHeight = fHeight;	

	vecCopy(_vCenterPos, pCenterPos);
	vecCopy(_verticalVec, pVerticalVec);
	vecCopy(_horizonVec, pHorizonVec);

	vect3d tmpVec, tmpPoint;

	//	Top square
	vecScale(_verticalVec, _fHeight / 2.0, tmpVec);
	point2point(_vCenterPos, tmpVec, tmpPoint);
	_vs[0] = new Square(tmpPoint, _verticalVec, _horizonVec, _fLength, _fWidth);
	//_vs[0]->setColor(c1,c1,c1, c1, 0.5);

	vecScale(_verticalVec, (-1)*_fHeight / 2.0, tmpVec);
	point2point(_vCenterPos, tmpVec, tmpPoint);
	vecScale(_verticalVec, -1, tmpVec);
	_vs[1] = new Square(tmpPoint, tmpVec, _horizonVec, _fLength, _fWidth);
	//_vs[1]->setColor(c1,c1,c1, c1,0.5);

	//	Left square
	vect3d vLeftNormalVec;
	cross_product(_horizonVec, _verticalVec, vLeftNormalVec);
	normalize(vLeftNormalVec);

	vecScale(vLeftNormalVec, _fLength / 2.0, tmpVec);
	point2point(_vCenterPos, tmpVec, tmpPoint);
	_vs[2] = new Square(tmpPoint, vLeftNormalVec, _horizonVec, _fHeight, _fWidth);
	//_vs[2]->setColor(c2,c2,c2, c2,0.5);

	vecScale(vLeftNormalVec, (-1)*_fLength / 2.0, tmpVec);
	point2point(_vCenterPos, tmpVec, tmpPoint);
	vecScale(vLeftNormalVec, -1, tmpVec);
	_vs[3] = new Square(tmpPoint, tmpVec, _horizonVec, _fHeight, _fWidth);
	//_vs[3]->setColor(c2,c2,c2, c2,0.5);

	//	Right square
	vecScale(_horizonVec, _fWidth / 2.0, tmpVec);
	point2point(_vCenterPos, tmpVec, tmpPoint);
	_vs[4] = new Square(tmpPoint, _horizonVec, _verticalVec, _fLength, _fHeight);
	//_vs[4]->setColor(c3,c3,c3, c3,0.5);

	vecScale(_horizonVec, (-1)*_fWidth / 2.0, tmpVec);
	point2point(_vCenterPos, tmpVec, tmpPoint);
	vecScale(_horizonVec, -1, tmpVec);
	_vs[5] = new Square(tmpPoint, tmpVec, _verticalVec, _fLength, _fHeight);
	//_vs[5]->setColor(c3,c3,c3, c3,0.5);

	updateBBox();
}

Cube::~Cube()
{
	for(int i = 0; i<6; i++)
	{
		delete _vs[i];
	}
}

bool Cube::isHit(	Ray &ray,
					vect3d &pNormal, float *t, vect3d *pTexColor)
{
	if(!_bbox.isHit(ray))
	{
		return false;
	}

	float fCurrT = 0xFFFFFFFF;
	float t0 = 0xFFFFFFFF;
	vect3d vCurrNormal;
	for(int i = 0; i < 6; i ++)
	{
		t0 = 0xFFFFFFFF;
		vect3d tmpv;
		bool bHit = _vs[i]->isHit(ray, tmpv, &t0, pTexColor);
		if( bHit && (t0 < fCurrT) )
		{
			fCurrT = t0;			
			vecCopy(vCurrNormal, tmpv);
		}	
	}

	if(fCurrT == 0xFFFFFFFF)
	{
		return false;
	}

	*t = fCurrT;
	vecCopy(pNormal, vCurrNormal);
	return true;
}

void Cube::updateBBox()
{

	//	

	float x[6][2] = {0};
	float y[6][2] = {0};
	float z[6][2] = {0};

	for(int i = 0; i < 6; i ++)
	{
		_vs[i]->getBBox()->getBoundValues(X_AXIS, x[i] + 0, x[i] + 1);
		_vs[i]->getBBox()->getBoundValues(Y_AXIS, y[i] + 0, y[i] + 1);
		_vs[i]->getBBox()->getBoundValues(Z_AXIS, z[i] + 0, z[i] + 1);
	}

	//

	float xmin = 99999999.f, xmax = -99999999.f;
	float ymin = 99999999.f, ymax = -99999999.f;
	float zmin = 99999999.f, zmax = -99999999.f;

	for(int i = 0; i < 6; i ++)
	{
		if(x[i][0] < xmin)	xmin = x[i][0];
		if(x[i][1] > xmax)	xmax = x[i][1];

		if(y[i][0] < ymin)	ymin = y[i][0];
		if(y[i][1] > ymax)	ymax = y[i][1];

		if(z[i][0] < zmin)	zmin = z[i][0];
		if(z[i][1] > zmax)	zmax = z[i][1];
	}

	_bbox.setDim(xmin, xmax, ymin, ymax, zmin, zmax);
}

void Cube::setTexture(int texId, TexMapType eType)
{
	for(int i = 0; i < 6; i++)
	{
		_vs[i]->setTexture(texId, eType);
	}
}


///
///
///
bool isTriangleHit(	vect3d vertices[3], Ray &ray, 
					float *pt, float *pu, float *pv)
{
	//
	//	Real-Time Rendering 2nd, 13.7.2
	//
	vect3d e1; points2vec(vertices[0], vertices[1], e1);
	vect3d e2; points2vec(vertices[0], vertices[2], e2);

	vect3d p;  cross_product(ray.direction_vec, e2, p);
	float a  = dot_product(e1, p);
	if(a > -epsi && a < epsi)
	{
		return false;
	}

	float f  = 1.f / a;
	vect3d s; points2vec(vertices[0], ray.start_point, s);
	float u = f * dot_product(s, p);
	if(u < 0.f || u > 1.f)
	{
		return false;
	}

	vect3d q;	cross_product(s, e1, q);
	float v = f * dot_product(ray.direction_vec, q);
	if(v < 0.f || (u + v) > 1.f)
	{
		return false;
	}

	float t = f * dot_product(e2, q);
	if(t <= epsi)
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