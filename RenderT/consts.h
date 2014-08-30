#ifndef CONSTS_H
#define CONSTS_H

#define WIN_WIDTH 512
#define WIN_HEIGHT 512

//	Windows Params
extern unsigned WinWidth;
extern unsigned WinHeight;
extern unsigned WinLeft;
extern unsigned WinTop;

extern char * WinTitle;

extern float ViewPlaneRatio;

//	Zoom params
extern unsigned ZoomStep;
extern unsigned MaxDist;
extern unsigned nCurrDist;

//	
extern int eKernelType;
extern int MaxRayDepth;
extern int nMultiSampleCount;
extern float epsi;

extern float fSamplingDeltaFactor;

extern unsigned nCurrObj;

//
//	KD-tree parameters
//

extern const unsigned nMaxNodeCount;
extern int nObjDepth;
extern int nSceneDepth;

//
//	GPU params
//
extern int bGPUEnabled;

//	All the ray in the ray - tree [ 2^(n + 1) - 1 ]
//	for example, 2048 for MaxRayDepth of 10 (to make it aligned, i use 2^(n + 1))
//	And, WinWidth rays at a time...
#define MAX_RAY_COUNT_PER_TREE ( (2 << (MaxRayDepth - 1)) - 1)
#define MAX_RAY_COUNT (MAX_RAY_COUNT_PER_TREE * WinWidth)
//

//	Radiosity Param
//
extern float fVPLPossibility;
extern float fVPLIllmThreshold;
extern float fVPLAtten;

//
//	Photon-Mapping Param
//
extern int	bPMEnabled;
extern float fPMFactor;
extern float fPMRad;
extern unsigned nMinPhotonNum;
extern float fDensityThreshold;	// how much to render
extern float ReflectionRatioThreshold;
extern float RefractionRatioThreshold;
extern float DiffuseThreshold;
extern float PhotonStep;

//
//	Ambient Occlusion
//
extern int bAOEnabled;
extern float fAOFactor;
extern float fEffectiveDist;
extern float fAngleStep;
extern float fAngleScope;


//
//	Bloom
//
extern int nBloomRad;
extern float fBloomWeight;

//	Tone mapping
//
#define GLOBAL_TMAP 1
#define LOCAL_TMAP 2
extern int iTMapType;
extern float fMaxDisplayY;

//
#define PIon180 (0.017453292222)
#define PI (3.1415926)


//	Lightcuts Params
//
extern int bLCEnabled;
extern float fLCRelevantFactorThreshold;
//	I know.. doing so is only because there's a thesis defense deadline
extern float fCurRatio;
extern unsigned nInvolvedCount;

//	For debug only
extern float fTraversalTime;
extern float fLightEvalTime;
extern float fSubTime1;

#endif