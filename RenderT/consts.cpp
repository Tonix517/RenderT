#include "consts.h"
#include "kernel.h"

//	512 is to make aligned memory for GPU. coalesce memory
unsigned WinWidth  = WIN_WIDTH;
unsigned WinHeight = WIN_HEIGHT;

unsigned WinLeft   = 200;
unsigned WinTop    = 50;

char * WinTitle = "Tony's Rendering Engine";

float ViewPlaneRatio = 0.01;

int eKernelType = BOX;

int MaxRayDepth = 4;
int nMultiSampleCount = 1;

float epsi = 0.001;

float fSamplingDeltaFactor = 0.3;

unsigned nCurrObj = 0;

const unsigned nMaxNodeCount = 2;

//

int bGPUEnabled = 0;

//
//	This parameter affects performance a lot.
//	I think this parameter depends on the specific scene.
//	Currently I think 2 is a reasonable value
//
//	8 + 1.3 * log(_nTriNum * 1.f)
//
int nObjDepth = 3;
int nSceneDepth = 3; // use the optimized formular

//	Radiosity
float fVPLPossibility = 0.02;
float fVPLIllmThreshold = 3.0;
float fVPLAtten = 0.02;

//	Photon-Mapping
int  bPMEnabled = 1;
float fPMFactor = 1;
unsigned nMinPhotonNum = 10;
float fPMRad = 4;
float ReflectionRatioThreshold = 0.2;
float RefractionRatioThreshold = 0.1;
float DiffuseThreshold = 0.9;
float PhotonStep = 1;

//	Ambient-Occlusion
int bAOEnabled = 1;
float fAOFactor = 1;
float fEffectiveDist = 90;
float fAngleStep = 4;	// 1:12.5 can remove AO alias
float fAngleScope = 50;

//
//	Bloom
//
int nBloomRad = 12;
float fBloomWeight = 0.7;

//	Tone mapping
//
int iTMapType = GLOBAL_TMAP;
float fMaxDisplayY = 683.f;

//	Lightcuts Params
//
int bLCEnabled = 1;
float fLCRelevantFactorThreshold = 0.00001;
float fCurRatio = 0;
unsigned nInvolvedCount = 0;

float fTraversalTime = 0;
float fLightEvalTime = 0;
float fSubTime1 = 0;