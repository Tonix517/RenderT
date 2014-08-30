#ifndef PHOTON_MAP_H
#define PHOTON_MAP_H

#include <vector>
#include "vector.h"
#include "bbox.h"
#include "ray.h"

///
///
///
struct photon
{
	vect3d	color;
	vect3d	pos3d;
	vect3d	dir;	// the same as hit surfaces
};

///
///
///
struct PhotonRay : public Ray
{
	//PhotonRay(PhotonRay &);

	PhotonRay(vect3d &pStart, vect3d &pDir, bool pbIsInObj = false)
		: Ray(pStart, pDir, pbIsInObj)
	{
	}

	photon thePhoton;
};

///
///
class PhotonMapBBox : protected BBox
{
public:

	void setAxisMin(AxisType , float);
	void setAxisMax(AxisType , float);

	inline float getAxisMin(AxisType eAxis)
	{
		switch(eAxis)
		{
		case X_AXIS:
			return _xmin;
			break;

		case Y_AXIS:
			return _ymin;
			break;

		case Z_AXIS:
			return _zmin;
			break;
		}

		return 0;
	}

	inline float getAxisMax(AxisType eAxis)
	{
		switch(eAxis)
		{
		case X_AXIS:
			return _xmax;
			break;

		case Y_AXIS:
			return _ymax;
			break;

		case Z_AXIS:
			return _zmax;
			break;
		}

		return 0;
	}

	bool isHit(Ray &);
};

///
///		Grid Acceleration
///
struct Grid
{
	//	yes, just copy the photons. for the performance use
	std::vector<photon>	_photons;
};

///
///
///
struct photon_map
{
	static void setRadius(float fRad);

	static void addPhoton(photon &);

	static unsigned organize();

	static bool isHit(Ray&);
	static bool getPhotonColor(vect3d &ctr, vect3d &retColor);

	static void clear();

private:	

	static float _fRadius;

	//
	//	Grids
	//
	static unsigned nGridXCount, nGridYCount, nGridZCount;
	static Grid *_grids;

	//	All the photons will be in a volume, with fRad 
	//	as the margin in all 3 dims - so to avoid unnecessary computing
	//
	static PhotonMapBBox	_photonVolumeBBox;	

	static std::vector<photon> _photonMap;
};

#endif