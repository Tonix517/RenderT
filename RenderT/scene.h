#ifndef SCENE_H
#define SCENE_H

#include "film.h"
#include "camera.h"
#include "object.h"
#include "light.h"
#include "kd-tree.h"

#include <vector>

#define USE_KD_TREE

class Scene
{
public:

	Scene();
	~Scene();

	void init();
	void clear();

	void load(char *pPath);

	//
	//	Camera Gettor/Settor
	//
	void setCamera(Camera *pCamera);
	Camera * getCamera()
	{
		return _pCamera;
	}

	//
	//	Ambient Light
	//
	void setAmbiColor(vect3d &pColor);
	void getAmbiColor(vect3d &pColor);

	void addObject(Object *pObj)
	{
		_vObjects.push_back(pObj);
	}

	void addLight(Light *pLight)
	{
		_vLights.push_back(pLight);
	}

	unsigned getLightNum() { return _vLights.size(); }
	Light *getLight(unsigned index) { return _vLights[index]; }

	unsigned getObjectNum() { return _vObjects.size(); }
	Object *getObject(unsigned index) { return _vObjects[index]; }

	//
	//
	Object* getHitInfo(Ray &, float *t, vect3d &vNorm, vect3d *pTexColor = NULL);

	//
	//	It is paralleled with render()
	//
	void compute();
	void abort();	//	To terminate compute() thread elegantly
	
	//
	//	Render the scene
	//
	void render();	

	bool bLoaded;

	void addDiffuseLight(Light*);

	bool bExeSwitch;
private:

	std::vector<Light*> _vDiffuseLights;

private:						

#ifdef USE_KD_TREE
	kd_node *_pKdNode;
#endif

	std::vector<Object *>	_vObjects;
	std::vector<Light *>	_vLights;

	Film	*_pFilm;
	Camera	*_pCamera;

	vect3d _ambientColor;

	PixelIntegrator *_pInt;

};

#endif