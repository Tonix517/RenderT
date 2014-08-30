#include "scene_descr.h"
#include "global.h"
#include "sampler.h"
#include "camera.h"
#include "consts.h"
#include "vector.h"
#include "light.h"
#include "obj_object.h"

#include <assert.h>
#include <string>
#include <fstream>
using namespace std;

const unsigned MAX_LINE = 1024;

string cutStrPrefix(string &sLine, char cDelim)
{
	streamsize inx = sLine.find_first_of(cDelim);
	if(inx != string::npos)
	{
		string sPre = sLine.substr(0, inx);
		sLine = sLine.substr(inx + 1, sLine.length());

		return sPre;
	}
	return "";
}

void SceneLoader::load(char *pPath, Scene &scene)
{
	assert(pPath);

	///
	///		Yes this is only a tmp implementation.
	///		The ideal situation is to use Lex & Yacc, as PBRT
	///
	ifstream in;
	in.open(pPath, ios_base::in);
	if(in.is_open())
	{
		while(in.good() && !in.eof())
		{
			char zBuf[MAX_LINE] = {0};
			in.getline(zBuf, MAX_LINE);
			
			string sLine(zBuf);

			//	empty line OR comment
			if(sLine.empty())
			{
				continue;
			}
			sLine.assign(zBuf + 2);

			char c0 = zBuf[0];
			switch(c0)
			{
			case '#':	///	Comment
				continue;
				break;
			
			case 's':	
			case 'S':	///	Scene
				loadScene(sLine, scene);
				break;

			case 'c':
			case 'C':	///	Camera	
				loadCamera(sLine, scene);
				break;

			case 'l':
			case 'L':	/// Lights
				loadLights(sLine, scene);
				break;

			case 'o':
			case 'O':	/// ObjObject
				loadObjObject(sLine, scene);
				break;

			case 'p':
			case 'P':	/// PrimaryObject				
				loadPrmObject(sLine, scene);
				break;
			};
		}
		
		in.close();
	}
	else
	{
		printf("SceneLoader:: cannot open scene file : %s\n", pPath);
	}

}

void SceneLoader::loadLights(std::string &sLine, Scene &scene)
{
	//	light type
	LightType eLightType;
	string sPre = cutStrPrefix(sLine, '|');
	if(sPre == "OMNI")
	{
		eLightType = OMNI_P;
		printf("Light Type : OMNI_P \n");
	}
	else if(sPre == "DIRP")
	{
		eLightType = DIR_P;
		printf("Light Type : DIR_P \n");
	}
	else if(sPre == "SQU_A")
	{
		eLightType = SQU_AREA;
		printf("Light Type : SQU_AREA \n");
	}
	else if(sPre == "DIR")
	{
		eLightType = DIR;
		printf("Light Type : DIR \n");
	}
	else
	{
		printf("not supported light type\n");
	}

	//	SQU AREA
	vect3d saCtr, saNorm, saHead; float saWid = 0, saHei = 0, ds = 0;

	//	light pos & dir
	vect3d pos;
	vect3d dir;
	if(eLightType == OMNI_P || eLightType == DIR_P)
	{
		sPre = cutStrPrefix(sLine, '|');
		if(sPre.find_first_of("POS") == 0)
		{
			sscanf(sPre.c_str(), "POS{%f, %f, %f}", &pos[0], &pos[1], &pos[2]);
		}
		sPre = cutStrPrefix(sLine, '|');
		if(sPre.find_first_of("DIR") == 0)
		{
			sscanf(sPre.c_str(), "DIR{%f, %f, %f}", &dir[0], &dir[1], &dir[2]);
		}
	}
	else if(eLightType == SQU_AREA)
	{
		sPre = cutStrPrefix(sLine, '|');
		sscanf(sPre.c_str(), "C{%f, %f, %f}N{%f, %f, %f}H{%f, %f, %f}:W(%f)H(%f):DS(%f)", 
							 &saCtr[0], &saCtr[1], &saCtr[2], 
							 &saNorm[0], &saNorm[1], &saNorm[2], 
							 &saHead[0], &saHead[1], &saHead[2], &saWid, &saHei, &ds);
	}
	else if(eLightType == DIR)
	{
		sPre = cutStrPrefix(sLine, '|');
		if(sPre.find_first_of("DIR") == 0)
		{
			sscanf(sPre.c_str(), "DIR{%f, %f, %f}", &dir[0], &dir[1], &dir[2]);
		}
	}
	else
	{
		printf("not supported light type ... \n");
	}

	//	attenuation
	float fAtten = 1;
	sPre = cutStrPrefix(sLine, '|');
	if(sPre.find_first_of("ATTEN") == 0)
	{
		sscanf(sPre.c_str(), "ATTEN(%f)", &fAtten);
	}
	
	//	colors
	vect3d spec;
	vect3d diff;
	vect3d ambi;
	for(int i = 0; i < 3; i ++)
	{
		sPre = cutStrPrefix(sLine, '|');
		switch(sPre[0])
		{
		case 'S':
			sscanf(sPre.c_str(), "S{%f, %f, %f}", &spec[0], &spec[1], &spec[2]);
			break;

		case 'D':
			sscanf(sPre.c_str(), "D{%f, %f, %f}", &diff[0], &diff[1], &diff[2]);
			break;

		case 'A':
			sscanf(sPre.c_str(), "A{%f, %f, %f}", &ambi[0],&ambi[1], &ambi[2]);
			break;
		}
	}//	for

	Light *l2;
	switch(eLightType)
	{
	case DIR:
		l2 = new DirLight(dir, fAtten);
		l2->setColors(ambi, diff, spec);
		scene.addLight(l2);
		break;

	case OMNI_P:
		l2 = new OmniPointLight(pos, fAtten);
		l2->setColors(ambi, diff, spec);
		scene.addLight(l2);
		break;

	case DIR_P:
		l2 = new DirPointLight(pos, dir, fAtten);
		l2->setColors(ambi, diff, spec);
		scene.addLight(l2);
		break;

	case SQU_AREA:
		{
			normalize(saNorm);
			normalize(saHead);
			SquareAreaLight *pSal = new SquareAreaLight(fAtten, saCtr, saNorm, saHead, saWid, saHei);
			pSal->setMaterial(spec, diff, ambi, fAtten);
			scene.addObject(pSal);
			//	discretize the area light
			pSal->setDiscretizeDimSize(ds);
			unsigned nSize = pSal->getSampleCount();
			for(int i = 0; i < nSize; i ++)
			{
				Light *pLight = pSal->getNextLightSample(i);
				scene.addLight(pLight);
			}
		}
		break;

	default:
		printf("light type not supported yet!\n");
		break;
	}
}
void SceneLoader::loadObjObject(std::string &sLine, Scene &scene)
{
	string sPre = cutStrPrefix(sLine, '|');

	//	FRKE
	float frke[4] = {0};
	if(sPre.find_first_of("FRKE") == 0)
	{
		sscanf(sPre.c_str(), "FRKE{%f, %f, %f, %f}", frke, frke + 1,frke + 2, frke + 3);
	}

	//	Path
	char path[MAX_LINE] = {0};
	sPre = cutStrPrefix(sLine, '|');
	if(sPre.find_first_of("PATH") == 0)
	{
		sscanf(sPre.c_str(), "PATH:%s", path);
	}				

	//	Smooth
	int bSmooth = 0;
	sPre = cutStrPrefix(sLine, '|');
	if(sPre.find_first_of("SMTH") == 0)
	{
		sscanf(sPre.c_str(), "SMTH(%d)", &bSmooth);
	}

	//	Translate
	float tran[3] = {0};
	sPre = cutStrPrefix(sLine, '|');
	if(sPre.find_first_of("TRAN") == 0)
	{
		sscanf(sPre.c_str(), "TRAN{%f, %f, %f}", tran, tran + 1, tran + 2);
	}

	//	Scale
	vect3d scal;
	sPre = cutStrPrefix(sLine, '|');
	if(sPre.find_first_of("SCAL") == 0)
	{
		sscanf(sPre.c_str(), "SCAL{%f, %f, %f}",&scal[0], &scal[1], &scal[2]);
	}

	//	Rotate
	vect3d axis;
	float angle = 0;
	sPre = cutStrPrefix(sLine, '|');
	if(sPre.find_first_of("ROT") == 0)
	{
		sscanf(sPre.c_str(), "ROT{%f, %f, %f}:%f", &axis[0], &axis[1], &axis[2], &angle);
		 normalize(axis);
	}

						
	//	material
	int bCustomMat = 0;
	vect3d spec;
	vect3d diff;
	vect3d ambi;
	float fShininess = 0;
	sPre = cutStrPrefix(sLine, '|');
	if(sPre.find_first_of("MAT") == 0)
	{
		sscanf(sPre.c_str(), "MAT(%d)", &bCustomMat);
		if(bCustomMat)
		{
			sPre = cutStrPrefix(sLine, '|');
			sscanf(sPre.c_str(), "S{%f, %f, %f}D{%f, %f, %f}A{%f, %f, %f}SH(%f)", 
									&spec[0], &spec[1], &spec[2], 
									&diff, &diff[1], &diff[2], 
									&ambi, &ambi[1], &ambi[2], &fShininess);

		}
	}

	ObjObject *pObj = new ObjObject(frke[0], frke[1], frke[2], frke[3]);	
	pObj->load(path);
	pObj->setSmooth(bSmooth == 1);
	
	if(bCustomMat)
	{
		pObj->setMaterial(spec, diff, ambi, fShininess);
	}

	pObj->rotate(angle, axis);
	pObj->scale(scal[0], scal[1], scal[2]);		
	pObj->translate(tran[0], tran[1], tran[2]);

#ifdef USE_KD_TREE_OBJ
	pObj->buildKdTree();
#endif

	scene.addObject(pObj);

	printf("- ObjObject loaded\n");
}

void SceneLoader::loadCamera(std::string &sLine, Scene &scene)
{
	CamType	eCamType;
	SamplingType eSplType;

	//	1.	Sampler Type
	//
	string sPre = cutStrPrefix(sLine, '|');
	if(sPre == "STRAT")
	{
		eSplType = STRATIFIED;
		printf("Sampling: STRATIFIED\n");
	}
	else
	{
		printf("not supported sampling type \n");
	}

	//	2. Camera Type
	float fPerpDist = 1000.f;
	sPre = cutStrPrefix(sLine, '|');							
	if(sPre.find_first_of("PERS") != string::npos)
	{					
		sscanf(sPre.c_str(), "PERS(%f)", &fPerpDist);
		eCamType = PERSP;
		printf("Camera Type : Perspective - %f \n", fPerpDist);
	}
	else if(sPre.find_first_of("ORTHO") != string::npos)
	{
		eCamType = ORTHO;
		printf("Camera Type : Orthogonal\n");
	}
	
	vect3d ctr;
	vect3d view;
	vect3d up;
	for(int i = 0; i < 3; i ++)
	{
		sPre = cutStrPrefix(sLine, '|');	// center pos		
		
		switch(sPre[0])
		{
		case 'C':	//	center pos
			sscanf(sPre.c_str(), "C{%f, %f, %f}", &ctr[0], &ctr[1], &ctr[2]);
			printf("Center Pos : {%.2f, %.2f, %.2f}\n", ctr[0], ctr[1], ctr[2]);
			break;
		case 'V':	//	center pos
			sscanf(sPre.c_str(), "V{%f, %f, %f}",&view[0], &view[1], &view[2]);
			normalize(view);
			printf("View Vec : {%.2f, %.2f, %.2f}\n", view[0], view[1], view[2]);
			break;
		case 'U':	//	center pos
			sscanf(sPre.c_str(), "U{%f, %f, %f}", &up[0], &up[1], &up[2]);
			normalize(up);
			printf("Up Vec : {%.2f, %.2f, %.2f}\n", up[0], up[1], up[2]);
			break;
		}
	}

	//	Set Camera
	Camera *pCam = NULL;
	switch(eCamType)
	{
	case PERSP:
		pCam = new PerpCamera(fPerpDist, ctr, up, view, 10, ViewPlaneRatio);
		break;

	case ORTHO:
		pCam = new OrthoCamera(ctr, up, view, 10, ViewPlaneRatio);
		break;
	}
	pCam->setSampler(eSplType);
	pCam->setMultiSamplingCount(nMultiSampleCount);
	scene.setCamera(pCam);
}

void SceneLoader::loadScene(std::string &sLine, Scene &scene)
{
	string sPre = cutStrPrefix(sLine, '|');
	if(sPre.find_first_of("BK") == 0)
	{
		vect3d bkColor;
		sscanf(sPre.c_str(), "BK{%f, %f, %f}", &bkColor[0], &bkColor[1], &bkColor[2]);
		printf("Scene Bk-Color : {%f, %f, %f} \n" ,bkColor[0], bkColor[1], bkColor[2] );
		scene.setAmbiColor(bkColor);
	}
}

void SceneLoader::loadPrmObject(std::string &sLine, Scene &scene)
{
	string sPre = cutStrPrefix(sLine, '|');

	float fRefl = 0, fRefr = 0, fRefrK = 1, fEmit = 0;
	vect3d spec;
	vect3d diff;
	vect3d ambi;
	float fShininess = 0;
	vect3d ctr;

	if(sPre == "SQU")
	{
		vect3d norm;
		vect3d head;

		float width = 0, height = 0;

		sPre = cutStrPrefix(sLine, '|');
		if(sPre.find_first_of("FRKE") == 0)
		{
			sscanf(sPre.c_str(), "FRKE{%f, %f, %f, %f}", &fRefl, &fRefr, &fRefrK, &fEmit);
		}

		//	texture
		sPre = cutStrPrefix(sLine, '|');
		int texId = -1;
		TexMapType eTexMode = STRETCH;
		if(sPre.find_first_of("TEX") == 0)
		{
			char texPath[1024] = {0};
			char texMode[20] = {0};
			sscanf(sPre.c_str(), "TEX:%s %s", texPath, texMode);

			string sTexMode(texMode);
			if(sTexMode == "STRETCH")
			{
				eTexMode = STRETCH;
			}
			else if(sTexMode == "REPEAT")
			{
				eTexMode = REPEAT;
			}
			else if(sTexMode == "STRAIGHT")
			{
				eTexMode = STRAIGHT;
			}
			else
			{
				printf(" un-recognized Texture mapping mode ! \n");
			}
			texId = TextureManager::loadTexture(texPath);
		}

		if(texId != -1)
		{
			sPre = cutStrPrefix(sLine, '|');
		}
		if(sPre.find_first_of("C") == 0)
		{
			sscanf(sPre.c_str(), "C{%f, %f, %f}N{%f, %f, %f}H{%f, %f, %f}", 
						&ctr[0], &ctr[1], &ctr[2],
						 &norm[0],  &norm[1],  &norm[2],
						 &head[0],  &head[1],  &head[2]);
					
		}

		sPre = cutStrPrefix(sLine, '|');
		if(sPre.find_first_of("W") == 0)
		{
			sscanf(sPre.c_str(), "W%f:H%f", &width, &height);
		}

		sPre = cutStrPrefix(sLine, '|');
		if(sPre.find_first_of("S") == 0)
		{
			sscanf(sPre.c_str(), "S{%f, %f, %f}D{%f, %f, %f}A{%f, %f, %f}SH(%f)", 
								&spec[0], &spec[1], &spec[2], 
								 &diff[0],  &diff[1],  &diff[2], 
								 &ambi[0],  &ambi[1],  &ambi[2], &fShininess );
		}

		//
		Square *squ = new Square(ctr, norm, head, width, height, fRefl, fRefr, fRefrK, fEmit);
		squ->setMaterial(spec, diff, ambi, fShininess);
		if(texId != -1)
		{
			squ->setTexture(texId, eTexMode);
		}
		scene.addObject(squ);

		printf("- Square loaded ... \n");
	}
	else if(sPre == "SPH")
	{
		float rad = 0;

		sPre = cutStrPrefix(sLine, '|');
		if(sPre.find_first_of("FRKE") == 0)
		{
			sscanf(sPre.c_str(), "FRKE{%f, %f, %f, %f}", &fRefl, &fRefr, &fRefrK, &fEmit);
		}

		//	texture
		sPre = cutStrPrefix(sLine, '|');
		int texId = -1;
		TexMapType eTexMode = STRETCH;
		if(sPre.find_first_of("TEX") == 0)
		{
			char texPath[1024] = {0};
			sscanf(sPre.c_str(), "TEX:%s", texPath);
			texId = TextureManager::loadTexture(texPath);
		}

		if(texId != -1)
		{
			sPre = cutStrPrefix(sLine, '|');
		}
		if(sPre.find_first_of("C") == 0)
		{
			sscanf(sPre.c_str(), "C{%f, %f, %f}R(%f)", 
						&ctr[0], &ctr[1], &ctr[2],&rad );
					
		}

		sPre = cutStrPrefix(sLine, '|');
		if(sPre.find_first_of("S") == 0)
		{
			sscanf(sPre.c_str(), "S{%f, %f, %f}D{%f, %f, %f}A{%f, %f, %f}SH(%f)", 
								&spec[0], &spec[1], &spec[2], 
								&diff[0], &diff[1], &diff[2], 
								&ambi[0], &ambi[1], &ambi[2], &fShininess );
		}

		//
		Sphere *sp = new Sphere(rad, ctr, fRefl, fRefr, fRefrK, fEmit);
		sp->setMaterial(spec, diff, ambi, fShininess);
		if(texId != -1)
		{
			sp->setTexture(texId, eTexMode);
		}
		scene.addObject(sp);

		printf("- Sphere loaded ... \n");
	}
	else if(sPre == "CUB")
	{
		vect3d vertVec;
		vect3d horiVec;

		float len = 0, width = 0, height = 0;

		sPre = cutStrPrefix(sLine, '|');
		if(sPre.find_first_of("FRKE") == 0)
		{
			sscanf(sPre.c_str(), "FRKE{%f, %f, %f, %f}", &fRefl, &fRefr, &fRefrK, &fEmit);
		}

		//	texture
		sPre = cutStrPrefix(sLine, '|');
		int texId = -1;
		TexMapType eTexMode = STRETCH;
		if(sPre.find_first_of("TEX") == 0)
		{
			char texPath[1024] = {0};
			char texMode[20] = {0};
			sscanf(sPre.c_str(), "TEX:%s %s", texPath, texMode);

			string sTexMode(texMode);
			if(sTexMode == "STRETCH")
			{
				eTexMode = STRETCH;
			}
			else if(sTexMode == "REPEAT")
			{
				eTexMode = REPEAT;
			}
			else if(sTexMode == "STRAIGHT")
			{
				eTexMode = STRAIGHT;
			}
			else
			{
				printf(" un-recognized Texture mapping mode ! \n");
			}
			texId = TextureManager::loadTexture(texPath);
		}

		if(texId != -1)
		{
			sPre = cutStrPrefix(sLine, '|');
		}
		if(sPre.find_first_of("L") == 0)
		{
			sscanf(sPre.c_str(), "L(%f)W(%f)H(%f)C{%f, %f, %f}V{%f, %f, %f}H{%f, %f, %f}", &len, &width, &height,
						&ctr[0], &ctr[1], &ctr[2],
						&vertVec[0], &vertVec[1], &vertVec[2], 
						&horiVec[0], &horiVec[1], &horiVec[2] );
					
		}

		sPre = cutStrPrefix(sLine, '|');
		if(sPre.find_first_of("S") == 0)
		{
			sscanf(sPre.c_str(), "S{%f, %f, %f}D{%f, %f, %f}A{%f, %f, %f}SH(%f)", 
								&spec[0], &spec[1], &spec[2], 
								&diff[0], &diff[1], &diff[2], 
								&ambi[0], &ambi[1], &ambi[2], &fShininess );
		}
		normalize(vertVec);	normalize(horiVec);
		Cube *pCube0 = new Cube(len, width, height, ctr, vertVec, horiVec, fRefl, fRefr, fRefrK, fEmit);
		if(texId != -1)
		{
			pCube0->setTexture(texId, eTexMode);
		}
		pCube0->setMaterial(spec, diff, ambi, fShininess);
		scene.addObject(pCube0);

		printf("- Cube loaded ... \n");
	}
	else
	{
		printf("Prime object not supported yet\n");
	}
}