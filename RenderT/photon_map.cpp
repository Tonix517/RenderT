#include "photon_map.h"
#include "consts.h"
using namespace std;

float photon_map::_fRadius = 5;
PhotonMapBBox	photon_map::_photonVolumeBBox;	
std::vector<photon> photon_map::_photonMap;
Grid *photon_map::_grids = NULL;
unsigned photon_map::nGridXCount = 0;
unsigned photon_map::nGridYCount = 0;
unsigned photon_map::nGridZCount = 0;
//
//PhotonRay::PhotonRay(PhotonRay &ptnRay)
//	: Ray(ptnRay)
//{
//	vecCopy(thePhoton.color, ptnRay.thePhoton.color);
//	vecCopy(thePhoton.dir, ptnRay.thePhoton.dir);
//	vecCopy(thePhoton.pos3d, ptnRay.thePhoton.pos3d);
//}

void photon_map::setRadius(float fRad)
{
	_fRadius = fRad;

	//	Yeah I know..
	_photonVolumeBBox.setAxisMin(X_AXIS, 999999.f);
	_photonVolumeBBox.setAxisMin(Y_AXIS, 999999.f);
	_photonVolumeBBox.setAxisMin(Z_AXIS, 999999.f);

	_photonVolumeBBox.setAxisMax(X_AXIS, -999999.f);
	_photonVolumeBBox.setAxisMax(Y_AXIS, -999999.f);
	_photonVolumeBBox.setAxisMax(Z_AXIS, -999999.f);
}

void photon_map::addPhoton(photon &rPtn)
{
	//	 I would rather to do some copying 
	//	 instead of tons of new & delete.
	float fCurrXMin = rPtn.pos3d[0] - _fRadius, fCurrXMax = rPtn.pos3d[0] + _fRadius;
	float fCurrYMin = rPtn.pos3d[1] - _fRadius, fCurrYMax = rPtn.pos3d[1] + _fRadius;
	float fCurrZMin = rPtn.pos3d[2] - _fRadius, fCurrZMax = rPtn.pos3d[2] + _fRadius;

	if( fCurrXMin < _photonVolumeBBox.getAxisMin(X_AXIS) )	_photonVolumeBBox.setAxisMin(X_AXIS, fCurrXMin);
	if( fCurrXMax > _photonVolumeBBox.getAxisMax(X_AXIS) )	_photonVolumeBBox.setAxisMax(X_AXIS, fCurrXMax);

	if( fCurrYMin < _photonVolumeBBox.getAxisMin(Y_AXIS) )	_photonVolumeBBox.setAxisMin(Y_AXIS, fCurrYMin);
	if( fCurrYMax > _photonVolumeBBox.getAxisMax(Y_AXIS) )	_photonVolumeBBox.setAxisMax(Y_AXIS, fCurrYMax);

	if( fCurrZMin < _photonVolumeBBox.getAxisMin(Z_AXIS) )	_photonVolumeBBox.setAxisMin(Z_AXIS, fCurrZMin);
	if( fCurrZMax > _photonVolumeBBox.getAxisMax(Z_AXIS) )	_photonVolumeBBox.setAxisMax(Z_AXIS, fCurrZMax);

	_photonMap.push_back(rPtn);
}

bool photon_map::getPhotonColor(vect3d &ctr, vect3d &retColor)
{
	if(nGridXCount * nGridYCount * nGridZCount == 0)
	{
		return false;
	}

	float diam = (_fRadius * 2);
	
	vector<Grid*> refGrids;

	//	Get relative grids
	int currX = (ctr[0] - _photonVolumeBBox.getAxisMin(X_AXIS) + diam) / diam;
	int currY = (ctr[1] - _photonVolumeBBox.getAxisMin(Y_AXIS) + diam) / diam;
	int currZ = (ctr[2] - _photonVolumeBBox.getAxisMin(Z_AXIS) + diam) / diam;

	vect3d currGridCtr;
	currGridCtr[0] = _photonVolumeBBox.getAxisMin(X_AXIS) + (currX + 0.5) * diam;
	currGridCtr[1] = _photonVolumeBBox.getAxisMin(Y_AXIS) + (currY + 0.5) * diam;
	currGridCtr[2] = _photonVolumeBBox.getAxisMin(Z_AXIS) + (currZ + 0.5) * diam;

	for(int ix = -1; ix <= 1; ix ++)
	{
		if( (ix == -1 && ctr[0] > (currGridCtr[0] + diam)) ||
			(ix ==  1 && ctr[0] < (currGridCtr[0] - diam)) )
		{
			continue;
		}
		for(int iy = -1; iy <= 1; iy ++)
		{
			if( (iy == -1 && ctr[1] > (currGridCtr[1] + diam)) ||
				(iy ==  1 && ctr[1] < (currGridCtr[1] - diam)) )
			{
				continue;
			}
			for(int iz = -1; iz <= 1; iz ++)
			{
				if( (iz == -1 && ctr[2] > (currGridCtr[2] + diam)) ||
					(iz ==  1 && ctr[2] < (currGridCtr[2] - diam)) )
				{
					continue;
				}
				
				int x = currX + ix;
				int y = currY + iy;
				int z = currZ + iz;
				if( x >=0 && x < nGridXCount &&
					y >=0 && y < nGridYCount &&
					z >=0 && z < nGridZCount )
				{
					refGrids.push_back(&_grids[x + nGridXCount * y + nGridXCount * nGridYCount * z]);
				}
			}// iz
		}// iy
	}// ix

	if( !refGrids.empty() )
	{
		unsigned nCount = 0;
		for(int i = 0; i < refGrids.size(); i ++)
		{
			for(int j = 0; j < refGrids[i]->_photons.size(); j ++)
			{
				photon &ptn =  refGrids[i]->_photons[j];

				//	within the sphere
				vect3d dist;
				points2vec(ctr, ptn.pos3d, dist);
				float fdist = vecLen(dist);
				if( fdist <= _fRadius)
				{
					point2point(retColor, ptn.color, retColor);
					nCount ++;
				}
			}
		}

		if(nCount > nMinPhotonNum)
		{
			//	this sharpness seems better..
			vecScale(retColor, 1.f/pow((_fRadius * 2/PhotonStep), 2) / _fRadius, retColor);
			clampColor(retColor);
		}
		else
		{
			return false;
		}

		return true;
	}

	return false;
}

unsigned photon_map::organize()
{
	if(_photonMap.empty())
	{
		return 0 ;
	}

	if(_grids)
	{
		delete [] _grids;
		_grids = NULL;
	}

	float diam = (_fRadius * 2);

	nGridXCount = ( _photonVolumeBBox.getAxisMax(X_AXIS) -  _photonVolumeBBox.getAxisMin(X_AXIS) + diam) / diam;
	nGridYCount = ( _photonVolumeBBox.getAxisMax(Y_AXIS) -  _photonVolumeBBox.getAxisMin(Y_AXIS) + diam) / diam;
	nGridZCount = ( _photonVolumeBBox.getAxisMax(Z_AXIS) -  _photonVolumeBBox.getAxisMin(Z_AXIS) + diam) / diam;

	//	TODO: maybe a sparse matrix is needed for some scenes.
	_grids = new Grid[ nGridXCount * nGridYCount * nGridZCount * sizeof(Grid)];

	for(int i = 0; i < _photonMap.size(); i ++)
	{
		photon &rPtn = _photonMap[i];

		unsigned nX = (rPtn.pos3d[0] - _photonVolumeBBox.getAxisMin(X_AXIS) + diam) / diam;
		unsigned nY = (rPtn.pos3d[1] - _photonVolumeBBox.getAxisMin(Y_AXIS) + diam) / diam;
		unsigned nZ = (rPtn.pos3d[2] - _photonVolumeBBox.getAxisMin(Z_AXIS) + diam) / diam;

		Grid *pGrid = &_grids[nX + nGridXCount * nY + nGridXCount * nGridYCount * nZ];
		pGrid->_photons.push_back(rPtn);
	}

	unsigned nCount = _photonMap.size();
	_photonMap.clear();

	return nCount;
}

void photon_map::clear()
{
	if(_grids)
	{
		delete [] _grids;
		_grids = NULL;
	}

	_photonMap.clear();

		//	Yeah I know..
	_photonVolumeBBox.setAxisMin(X_AXIS, 999999.f);
	_photonVolumeBBox.setAxisMin(Y_AXIS, 999999.f);
	_photonVolumeBBox.setAxisMin(Z_AXIS, 999999.f);

	_photonVolumeBBox.setAxisMax(X_AXIS, -999999.f);
	_photonVolumeBBox.setAxisMax(Y_AXIS, -999999.f);
	_photonVolumeBBox.setAxisMax(Z_AXIS, -999999.f);
}

///
///
///

void PhotonMapBBox::setAxisMin(AxisType eAxis, float fValue)
{
	switch(eAxis)
	{
	case X_AXIS:
		_xmin = fValue;
		break;

	case Y_AXIS:
		_ymin = fValue;
		break;

	case Z_AXIS:
		_zmin = fValue;
		break;
	}
}

void PhotonMapBBox::setAxisMax(AxisType eAxis, float fValue)
{
	switch(eAxis)
	{
	case X_AXIS:
		_xmax = fValue;
		break;

	case Y_AXIS:
		_ymax = fValue;
		break;

	case Z_AXIS:
		_zmax = fValue;
		break;
	}
}

bool photon_map::isHit(Ray &ray)
{
	return _photonVolumeBBox.isHit(ray);
}

bool PhotonMapBBox::isHit(Ray &ray)
{
	return BBox::isHit(ray);
}
