#include "consts.h"
#include "scene.h"
#include "film.h"
#include "_lightcuts.h"
#include "integrator.h"
#include "scene_descr.h"
#include "obj_object.h"
#include "gpu/geometry_gpu.h"
#include "gpu/_lightcuts_gpu.h"
#include "tracer.h"
#include "vector.h"
#include <algorithm>
#include <assert.h>
#include <time.h>
#include <math.h>
#include <queue>

Scene::Scene()
	: _pCamera(NULL)
	, bExeSwitch(false)
	, bLoaded(false)
	, _pInt(NULL)
#ifdef USE_KD_TREE
	, _pKdNode(NULL)
#endif
{			
	_ambientColor[0] = 0;			
	_ambientColor[1] = 0;			
	_ambientColor[2] = 0;	
}

Scene::~Scene()
{
	clear();

	if(_pFilm)
	{
		_pFilm->destroy();
		delete _pFilm;
		_pFilm = NULL;
	}
	
	if(_pInt)
	{
		delete[] _pInt;
	}
}

void Scene::init()
{
	_pFilm = new Film;
	_pFilm->init(WinWidth, WinHeight);

	_pInt = new PixelIntegrator[WinWidth];

	Tracer::setBRDF(OGL);
}

void Scene::clear()
{

	_pFilm->clear();

	//

#ifdef USE_KD_TREE
	if(_pKdNode)
	{
		delete _pKdNode;
		_pKdNode = NULL;
	}
#endif

	if(_pCamera)
	{
		delete _pCamera;
		_pCamera = NULL;
	}

	if( !_vObjects.empty())
	{
		std::vector<Object *>::iterator iterObj = _vObjects.begin();
		for(; iterObj != _vObjects.end(); iterObj ++)
		{
			delete *iterObj;
		}
		_vObjects.clear();
	}

	if( !_vLights.empty())
	{
		std::vector<Light *>::iterator iterLight = _vLights.begin();
		for(; iterLight != _vLights.end(); iterLight ++)
		{
			delete *iterLight;
		}
		_vLights.clear();
	}

	if( !_vDiffuseLights.empty())
	{
		std::vector<Light *>::iterator iterLight = _vDiffuseLights.begin();
		for(; iterLight != _vDiffuseLights.end(); iterLight ++)
		{
			delete *iterLight;
		}
		_vDiffuseLights.clear();
	}

	//	texture
	TextureManager::clear();

	photon_map::clear();

	bLoaded = false;
}

void Scene::setCamera(Camera *pCamera)
{
	if(_pCamera)
	{
		delete _pCamera;
	}

	_pCamera = pCamera;
}

void Scene::setAmbiColor(vect3d &pColor)
{
	vecCopy(_ambientColor, pColor);
}

void Scene::getAmbiColor(vect3d &pColor)
{
	vecCopy(pColor, _ambientColor);
}

void Scene::abort()
{
	bExeSwitch = false;
}

void Scene::compute()
{
	bExeSwitch = true;

	//	Set Integrator Kernel Type
	PixelIntegrator::setKernelType((KernType)eKernelType);

	std::vector<LightCpu> vDiffLightVec;

	//	Pass 1 : Specular Color Computation
	//
	clock_t b0 = clock();
	for(unsigned i = 0; i < WinHeight; i ++)
	{
		if(!bExeSwitch) return;

		_pCamera->genViewRaysByRow(i, _pInt);
		
		if( !bGPUEnabled )
		{
			Tracer::computePixels(_pInt, WinWidth, 1);
		}
		else
		{
			Tracer::computePixels_GPU(_pInt, WinWidth, 1, &vDiffLightVec);
		}

		_pFilm->setRowColor(i, _pInt);
	}
	printf("[First Pass] : %.5f\n", (clock() - b0)/1000.f);

	//
	//	Photon Trajectory
	//
	unsigned nPtnCount = 0;
	if(bPMEnabled)
	{
		printf(" - Projecting Photons from Lights... \n");
		photon_map::setRadius(fPMRad);
		Tracer::shootAllPhotons();
		printf(" - Organizing Photons ...\n");
		nPtnCount = photon_map::organize();
		printf(" Photon Number : %d \n", nPtnCount);
	}
	
	//
	//	Get ready for the 2nd pass
	//
	bool bDiffuseAvailable = bGPUEnabled ? !vDiffLightVec.empty() : !_vDiffuseLights.empty();
	if( bDiffuseAvailable || (bPMEnabled && nPtnCount > 0))
	{
		printf("Second Pass is about to begin...\n");


		//	Replace Lights
		//
		if( !bGPUEnabled )
		{
			//	1. clear former lights
			if( !_vLights.empty())
			{
				std::vector<Light *>::iterator iterLight = _vLights.begin();
				for(; iterLight != _vLights.end(); iterLight ++)
				{
					delete *iterLight;
				}
				_vLights.clear();
			}

			if(bDiffuseAvailable)
			{
				//	1. copy diffuse lights
				printf("VPL Number : %d \n", _vDiffuseLights.size());
				std::vector<Light*>::iterator iter = _vDiffuseLights.begin();
				for(; iter != _vDiffuseLights.end(); iter ++)
				{
					_vLights.push_back(*iter);
				}

				//	2. clear diffuse lights vec
				_vDiffuseLights.clear();
			}

			///
			///		CPU Light-cuts Goes here
			///
			if(bLCEnabled)
			{
				unsigned nLCNodeCount = 0;
				lt_node *pLcTree = buildLightTree(_vLights, nLCNodeCount);
				Tracer::enableLightcuts(pLcTree, nLCNodeCount);

				printf("\n======== Lightcuts Enabled ========\n");
			}
		}
		else
		{
			unsigned nLightCount = vDiffLightVec.size();

			extern LightGpu *gpuLights;
			extern LightCpu *cpuLights;

			//	re-alloc space
			if(gpuLights)
			{
				cudaFree(gpuLights);
			}
			cudaError_t err = cudaMalloc(&gpuLights, sizeof(LightGpu) * nLightCount);
			if(err != cudaSuccess)
			{
				printf("CUDA error %s, %s, %s \n", __FILE__, __LINE__, cudaGetErrorString(err));
			}

			if(cpuLights)
			{
				free(cpuLights);
			}
			cpuLights = (LightCpu*)malloc(sizeof(LightCpu) * nLightCount);

			//	copy
			for(int i = 0; i < nLightCount; i ++)
			{
				LightCpu *pCurrCpuLight = cpuLights + i;

				pCurrCpuLight->eType = vDiffLightVec[i].eType;

				pCurrCpuLight->_fAttenuate = vDiffLightVec[i]._fAttenuate;

				vecCopy(pCurrCpuLight->_ambientColor, vDiffLightVec[i]._ambientColor);
				vecCopy(pCurrCpuLight->_diffuseColor, vDiffLightVec[i]._diffuseColor);
				vecCopy(pCurrCpuLight->_specularColor, vDiffLightVec[i]._specularColor);

				//	DirPoint
				vecCopy(pCurrCpuLight->_dirp_pos, vDiffLightVec[i]._dirp_pos);
				vecCopy(pCurrCpuLight->_dirp_dir, vDiffLightVec[i]._dirp_dir);
			}

			cudaMemcpy(gpuLights, cpuLights, sizeof(LightGpu) * nLightCount, cudaMemcpyHostToDevice);
			cudaThreadSynchronize();

			extern unsigned nCurrLightCount;
			nCurrLightCount = nLightCount;
			printf("GPU Lights copied : %d\n", nCurrLightCount);

			///
			///		GPU Light-cuts Goes here
			///
			if(bLCEnabled)
			{
				unsigned nLCNodeCount = 0;
				lt_node *pLcTree = buildLightTreeForGpu(vDiffLightVec, nLCNodeCount);
				Tracer::enableLightcuts(pLcTree, nLCNodeCount);
				
				//extern std::queue<lt_node *> node_queue;
				//extern void outputTree(std::queue<lt_node *> &theQueue, lt_node *pRoot);
				//node_queue.push(pLcTree);
				//outputTree(node_queue, pLcTree);

				setupLightcutsParam(pLcTree, nLCNodeCount, vDiffLightVec.size(), fLCRelevantFactorThreshold);

				printf("\n======== Lightcuts Enabled ========\n");
			}
		}
		//	Pass 2 : Diffuse Color Computation
		//
		clock_t n2ndStart = clock();

		for(unsigned i = 0; i < WinHeight; i ++)
		{
			if(!bExeSwitch) return;

			_pCamera->genViewRaysByRow(i, _pInt);

			if( !bGPUEnabled )
			{
				Tracer::computePixels(_pInt, WinWidth, 2);
			}
			else
			{
				Tracer::computePixels_GPU(_pInt, WinWidth, 2);
			}

			_pFilm->appendRowColor(i, _pInt);

			printf("\b\b\b\b\b\b\b\b%.3f%%", (i + 1) * 1.f/WinHeight * 100);
		}

		if(bLCEnabled)
		{
			if(!bGPUEnabled)
			{
				printf("\nAvg. Cut Ratio (Picked\\Total) : %.6f%%\n", fCurRatio * 100.f/ nInvolvedCount);
				
				printf("{Profile} Traversal : %.4f -- Evaluation : %.4f (sub1 : %.4f)\n", fTraversalTime / 1000.f, 
																			fLightEvalTime / 1000.f, fSubTime1/1000.f);
				fTraversalTime = 0;
				fLightEvalTime = 0;
				fSubTime1 = 0;
				fCurRatio = 0;
				nInvolvedCount = 0;
			}
			else
			{
				float fGpuCutRatio = getAvgCutRatioGpu();
				printf("\n Avg. Cut Ratio on GPU (Picked\\Total) : %.6f%%\n", fGpuCutRatio * 100.f);
			}

			Tracer::disableLightcuts();
		}

		printf("[2nd Pass Cost] : %.4f\n", (clock() - n2ndStart) / 1000.f);
	}

	PixelIntegrator::clean();
	bExeSwitch = false;
}

void Scene::render()
{
	_pFilm->render();
}

Object* Scene::getHitInfo(Ray &ray, float *t, vect3d &vNorm, vect3d *pTexColor)
{
#ifndef USE_KD_TREE
	//
	const float MAGIC_F = 999999.f;
	float finalT = MAGIC_F;
	Object *pFinal = NULL;
	vect3d texColor;

	vect3d normal;

	for(unsigned iObj = 0; iObj < _vObjects.size(); iObj ++)
	{
		float currT = 0.f;
		vect3d currNorm;
		vect3d currTexColor;
		if(_vObjects[iObj]->isHit(ray, currNorm, &currT, &currTexColor))
		{
			if(currT < finalT && currT >= 0)
			{
				finalT = currT;
				vecCopy(normal, currNorm);
				pFinal = _vObjects[iObj];
				vecCopy(texColor, currTexColor);
			}
		}
	}

	if(finalT < MAGIC_F && finalT > epsi)
	{
		*t = finalT;
		vecCopy(vNorm, normal);
		if(pTexColor)
		{
			vecCopy(*pTexColor, texColor);
		}

		return pFinal;
	}

	return NULL;

#else
	
	assert(_pKdNode);
	return _pKdNode->isHit(ray, vNorm, t, pTexColor);

#endif
}

void Scene::load(char *pPath)
{
	///
	SceneLoader::load(pPath, *this);

#ifdef USE_KD_TREE

	clock_t nStart = clock();
	if(_pKdNode)
	{
		delete _pKdNode;		
	}
	_pKdNode = new kd_node;

	for(int i = 0; i < _vObjects.size(); i ++)
	{
		_pKdNode->addObject(_vObjects[i]);
	}
	
	_pKdNode->updateBBox();
	if(nSceneDepth != 0)
	{
		kd_node::nSceneDepth = (nSceneDepth == -1)?(8 + 1.3 * log(_vObjects.size() * 1.f)):nSceneDepth;
		printf("[Building KD-tree for Scene of depth %d...\n", kd_node::nSceneDepth);

		_pKdNode->split();
	}

	printf(" Done %.2f \n", (clock() - nStart) / 1000.f);
#endif

	bLoaded = true;
}

void Scene::addDiffuseLight(Light *pLight)
{
	assert(pLight);
	_vDiffuseLights.push_back(pLight);
}