#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <assert.h>
#include "windows.h"

#include "consts.h"
#include "scene_descr.h"

#include "global.h"
#include "film.h"
#include "thread.h"
#include "texture.h"
#include "kernel.h"

#include "IL/ilut.h"
#include "GL/glut.h"
#include "GL/glui.h"

#include <cuda_runtime.h>

char strFilePath[MAX_PATH] = {0};

bool GetFilePath(char *strPath)
{
	assert(strPath);

	OPENFILENAME ofn;
	  
	// Initializing  
	ZeroMemory(&ofn, sizeof(ofn));  
	ofn.lStructSize = sizeof(ofn);  
	ofn.hwndOwner = NULL;
	ofn.lpstrFile = strPath;  
	//  
	//  
	ofn.lpstrFile[0] = '\0';  
	ofn.nMaxFile = MAX_PATH;  
	ofn.lpstrFilter = "Scene Files\0*.txt\0\0";  
	ofn.nFilterIndex = 1;  
	ofn.lpstrFileTitle = NULL;  
	ofn.nMaxFileTitle = 0;  
	ofn.lpstrInitialDir = ".";  
	ofn.Flags = OFN_PATHMUSTEXIST | OFN_FILEMUSTEXIST;  
	  
	// Get the path
	if ( GetOpenFileName(&ofn) )  
	{  		
		//	GLM only permits the file path with "/" as the delimiter
		size_t strLen = strlen(strPath);
		for(size_t i = 0; i < strLen; i++)
		{
			if(strPath[i] == '\\')
			{
				strPath[i] = '/';
			}
		}
		return true;
	}  

	return false;
}

//
static
void resize(int w, int h)
{
    glViewport(0, 0, w, h);	
}

///
///		main display logic goes here
///

static 
void destroy()
{	
	global_destroy();

	exit(EXIT_SUCCESS);
}

int iWinId;
void idle() 
{
	glutSetWindow(iWinId);
	glutPostRedisplay();
}

static
void display()
{
	glClear(GL_COLOR_BUFFER_BIT);

	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluOrtho2D(0, 1, 0, 1);

	scene.render();

	//
	glutSwapBuffers();
}


///
///		menu setup goes here	
///
void menu(int val)
{
	if(bIsRunning)
	{
		return;
	}

	switch (val)
	{
		case 1: 
			//	Get File
			if(GetFilePath(strFilePath))
			{
				scene.clear();
				scene.load(strFilePath);
				start_thread();
			}
			break;		

		case 2: 
			destroy();
			break;
	}
}

void makeMenu()
{	
	glutCreateMenu(menu);	
	glutAttachMenu(GLUT_RIGHT_BUTTON);
	
	glutAddMenuEntry("Load scene file...", 1);
	glutAddMenuEntry("Quit", 2);
}

		
static int nImgCount = 0;

static 
void key(unsigned char key, int x, int y)
{		
    switch (key) 
    {
	case ' ':
		start_thread();
		break;

		//	Break rendering
	case 'b':
	case 'B' :
		end_thread();
		scene.clear();
		break;

	case 'c':
	case 'C':
		//	Taking & Saving the screenshot	
		if(ilutGLScreen())
		{					  
		  ilEnable(IL_FILE_OVERWRITE);
		  char path[20] = {0};
		  sprintf(path, "RenderT_%d.bmp", nImgCount ++);
		  if(ilSaveImage(path))
		  {
			 printf("Screenshot saved successfully as \'%s\'!\n", path);
		  }
		  else
		  {
			 printf("Sorry, DevIL cannot save your screenshot...\n");
		  }
		}
		else
		{
		  printf(" Sorry man, DevIL screenshot taking failed...\n");
		}
		break;

    case 27 : 
    case 'q':
        destroy();
        break;
    }

    //glutPostRedisplay();
}

void mouse(int, int, int, int)
{

}

void printUsage()
{
	char *strUsage =	"{ How to Use } \n\n"
						" Right-click Mouse to Load scene file\n\n"
						" C: save image \n"
						" B: break current rendering \n"
						" Q: quit \n\n";
	printf(strUsage);
}

///

GLUI *glui;
GLUI *glui2;

GLUI_Panel *pPMPal;
GLUI_Checkbox *pPMSwitch;
GLUI_Checkbox *pPMGpuChk;

GLUI_Panel *pAOPal;
GLUI_Checkbox *pAOChk;

void callback_pm_enable(int)
{
	if(bPMEnabled)
	{
		pPMPal->enable();
	}
	else
	{
		pPMPal->disable();
		pPMSwitch->enable();
	}
}

void callback_ao_enable(int)
{
	if(bAOEnabled)
	{
		pAOPal->enable();
	}
	else
	{
		pAOPal->disable();
		pAOChk->enable();
	}
}

void callback_gpu_enable(int)
{
	if(bGPUEnabled)
	{
		int deviceCount = 0;
		cudaGetDeviceCount(&deviceCount);

		if(deviceCount == 0)
		{
			::MessageBox(NULL, "No GPU device found !", "Error", MB_OK);
			pPMGpuChk->set_int_val(0);
		}
	}
}

void callback_bloom_btn(int)
{
	 start_bloom_thread();
}

void callback_tone_m_btn(int)
{
	start_tone_m_thread();
}
///
///		
///
int main(int argc, char* argv[])
{
	printUsage();

	//	Window Setup
	glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);	
	
	glutInitWindowSize(WinWidth, WinHeight);
    glutInitWindowPosition(WinLeft, WinTop);
    iWinId = glutCreateWindow(WinTitle);
    
	glutMouseFunc(mouse);
	glutReshapeFunc(resize);
    glutDisplayFunc(display);
	glutIdleFunc(idle);
    glutKeyboardFunc(key);

	makeMenu();
	
	//	GLUI
	glui = GLUI_Master.create_glui( "Param Control", 0, WinWidth + WinLeft, WinTop );
	
	//	GPU setting
	//
	{
		GLUI_Panel *pPGpu = glui->add_panel("GPU Setup");
		pPMGpuChk = glui->add_checkbox_to_panel(pPGpu, "GPU Enabled", &bGPUEnabled, -1, callback_gpu_enable);		
	}

	//	Kernel Part
	//
	{
		GLUI_Panel *pKernPal = glui->add_panel("Ray-Tracing Param");
		
		//	Epsilon
		GLUI_Spinner *pEpsi = glui->add_spinner_to_panel(pKernPal, "Epsilon", GLUI_SPINNER_FLOAT, &epsi);
		pEpsi->set_float_limits(0, 2.99);
		pEpsi->set_speed(0.1);

		//	Kernel Type
		GLUI_Listbox *pKernTypeList = glui->add_listbox_to_panel(pKernPal, "Kernel Type", &eKernelType);
		pKernTypeList->add_item(BOX, "Box Filter");
		pKernTypeList->add_item(TRI, "Triangle Filter");
		pKernTypeList->add_item(GAU, "Gaussian Filter");
		pKernTypeList->add_item(MIT, "Mitchell Filter");

		//	Ray depth	
		GLUI_Spinner *pRayDep = glui->add_spinner_to_panel(pKernPal, "Ray Depth", GLUI_SPINNER_INT, &MaxRayDepth);
		pRayDep->set_int_limits(2, 9);
		pRayDep->set_speed(0.5);
		// Multi-sample Count
		GLUI_Spinner *pMS = glui->add_spinner_to_panel(pKernPal, "Multi-sample Count", GLUI_SPINNER_INT, &nMultiSampleCount);
		pMS->set_int_limits(1, 1000);
		pMS->set_speed(1);
		//	Sampling Delta Factor
		GLUI_Spinner *pSdf = glui->add_spinner_to_panel(pKernPal, "Sampling Delta Factor", GLUI_SPINNER_FLOAT, &fSamplingDeltaFactor);
		pSdf->set_float_limits(0, 10);
		pSdf->set_speed(0.1);
		//	KD-tree hint
		glui->add_statictext_to_panel(pKernPal,"");
		glui->add_statictext_to_panel(pKernPal, "-1:optimized - 0:no kd-tree");
		//	Scene KD-tree depth
		GLUI_Spinner *pScnD = glui->add_spinner_to_panel(pKernPal, "Scene KD-Tree Depth", GLUI_SPINNER_INT, &nSceneDepth);
		pScnD->set_float_limits(-1, 99);
		pScnD->set_speed(2);
		//	Object KD-tree depth
		GLUI_Spinner *pObjD = glui->add_spinner_to_panel(pKernPal, "Object KD-Tree Depth", GLUI_SPINNER_INT, &nObjDepth);
		pObjD->set_float_limits(-1, 99);
		pObjD->set_speed(1);

		glui->add_separator_to_panel(pKernPal);
		//	Lightcuts
		GLUI_Checkbox *pLCChk = glui->add_checkbox_to_panel(pKernPal, "Lightcuts Enabled", &bLCEnabled, -1);
		GLUI_Spinner *pLCThreshold = glui->add_spinner_to_panel(pKernPal, "Lightcuts Threshold", GLUI_SPINNER_FLOAT, &fLCRelevantFactorThreshold);
		pLCThreshold->set_float_limits(0, 99);
		pLCThreshold->set_speed(0.1);
	}

	//	Radiosity Param
	//
	{
		GLUI_Panel *pRadiPal = glui->add_panel("Radiosity Param");
		//	VPL Put Possibility
		GLUI_Spinner *pVPLPoss = glui->add_spinner_to_panel(pRadiPal, "VPL Possibility per Pixel", GLUI_SPINNER_FLOAT, &fVPLPossibility);
		pVPLPoss->set_float_limits(0, 1);
		pVPLPoss->set_speed(0.5);
		//	VPL Put Illumination Threshold
		GLUI_Spinner *pVPLThld = glui->add_spinner_to_panel(pRadiPal, "VPL Minimum-Illumination", GLUI_SPINNER_FLOAT, &fVPLIllmThreshold);
		pVPLThld->set_float_limits(0, 3);
		pVPLThld->set_speed(0.5);
		//	VPL Attenuation
		GLUI_Spinner *pVPLAtten = glui->add_spinner_to_panel(pRadiPal, "VPL Attenuation", GLUI_SPINNER_FLOAT, &fVPLAtten);
		pVPLAtten->set_float_limits(0, 1);
		pVPLAtten->set_speed(0.5);
	}
	
	//	Photon-mapping Param
	//
	{
		pPMPal = glui->add_panel("Photon-Mapping Param");
		pPMSwitch = glui->add_checkbox_to_panel(pPMPal, "Enable", &bPMEnabled, -1, callback_pm_enable);

		GLUI_Spinner *pPMFactor = glui->add_spinner_to_panel(pPMPal, "PM-Factor", GLUI_SPINNER_FLOAT, &fPMFactor);
		pPMFactor->set_float_limits(0, 100000);	pPMFactor->set_speed(0.2);

		GLUI_Spinner *pPMRad = glui->add_spinner_to_panel(pPMPal, "PM-Radius", GLUI_SPINNER_FLOAT, &fPMRad);
		pPMRad->set_float_limits(0, 99999);	pPMRad->set_speed(1);
		
		GLUI_Spinner *pPMMinPtnNUm = glui->add_spinner_to_panel(pPMPal, "Min. Photon Num.", GLUI_SPINNER_INT, &nMinPhotonNum);
		pPMMinPtnNUm->set_int_limits(0, 99999);	pPMMinPtnNUm->set_speed(1);

		GLUI_Spinner *pPMReflT = glui->add_spinner_to_panel(pPMPal, "Reflect-Threshold", GLUI_SPINNER_FLOAT, &ReflectionRatioThreshold);
		pPMReflT->set_float_limits(0, 1);	pPMReflT->set_speed(0.2);

		GLUI_Spinner *pPMRefrT = glui->add_spinner_to_panel(pPMPal, "Refract-Threshold", GLUI_SPINNER_FLOAT, &RefractionRatioThreshold);
		pPMRefrT->set_float_limits(0, 1);	pPMRefrT->set_speed(0.2);

		GLUI_Spinner *pPMDiffT = glui->add_spinner_to_panel(pPMPal, "Diffuse-Threshold", GLUI_SPINNER_FLOAT, &DiffuseThreshold);
		pPMDiffT->set_float_limits(0, 3);	pPMDiffT->set_speed(0.5);

		GLUI_Spinner *pPMSamplingStep = glui->add_spinner_to_panel(pPMPal, "Sampling Step", GLUI_SPINNER_FLOAT, &PhotonStep);
		pPMSamplingStep->set_int_limits(0.01, 50);	pPMSamplingStep->set_speed(1);
		
	}

	//	Photon-mapping Param
	//
	{
		pAOPal = glui->add_panel("Ambient Occlusion Param");
		pAOChk = glui->add_checkbox_to_panel(pAOPal, "Enable", &bAOEnabled, -1, callback_ao_enable);
			
		GLUI_Spinner *pAOFactor = glui->add_spinner_to_panel(pAOPal, "AO Factor", GLUI_SPINNER_FLOAT, &fAOFactor);
		pAOFactor->set_float_limits(0, 1);	pAOFactor->set_speed(0.2);

		GLUI_Spinner *pAOEffDist = glui->add_spinner_to_panel(pAOPal, "AO Effect-Distance", GLUI_SPINNER_FLOAT, &fEffectiveDist);
		pAOEffDist->set_float_limits(1, 900);	pAOEffDist->set_speed(1);

		GLUI_Spinner *pAORotAn = glui->add_spinner_to_panel(pAOPal, "AO Angle Step", GLUI_SPINNER_FLOAT, &fAngleStep);
		pAORotAn->set_float_limits(1, 90);	pAORotAn->set_speed(0.5);

		GLUI_Spinner *pAODwnAn = glui->add_spinner_to_panel(pAOPal, "AO Angle Scope", GLUI_SPINNER_FLOAT, &fAngleScope);
		pAODwnAn->set_int_limits(1, 90);	pAODwnAn->set_speed(1);
	}

	//	GLUI2
	glui2 = GLUI_Master.create_glui( "Film Param", 0, WinWidth + WinLeft + 250, WinTop );
	
	//	GPU setting
	//
	{
		//	Blooming
		//
		GLUI_Panel *pPBlm = glui2->add_panel("Blooming Setup");
		
		GLUI_Spinner *pBlmRad = glui2->add_spinner_to_panel(pPBlm, "Bloom Radius", GLUI_SPINNER_INT, &nBloomRad);		
		pBlmRad->set_int_limits(1, 15);	pBlmRad->set_speed(1);

		GLUI_Spinner *pBlmWei = glui2->add_spinner_to_panel(pPBlm, "Bloom Weight", GLUI_SPINNER_FLOAT, &fBloomWeight);		
		pBlmWei->set_float_limits(0, 1);	pBlmRad->set_speed(0.04);

		GLUI_Button *pBlmBtn = glui2->add_button_to_panel(pPBlm, "Bloom", -1, callback_bloom_btn);

		//	Tone-Mapping
		//
		GLUI_Panel *pPTm = glui2->add_panel("Tone-Mapping Setup");

		GLUI_Spinner *pTmMaxY = glui2->add_spinner_to_panel(pPTm, "Max Display Y", GLUI_SPINNER_FLOAT, &fMaxDisplayY);		
		pTmMaxY->set_float_limits(0, 2000);	pTmMaxY->set_speed(10.f);
		
		//	Tone-map Type
		GLUI_Listbox *pTmapTypeList = glui2->add_listbox_to_panel(pPTm, "Mapping Type", &iTMapType);
		pTmapTypeList->add_item(GLOBAL_TMAP, "Global");
		pTmapTypeList->add_item(LOCAL_TMAP, "Local");

		GLUI_Button *pTmBtn = glui2->add_button_to_panel(pPTm, "Tone-map", -1, callback_tone_m_btn);
	}

	GLUI_Master.set_glutIdleFunc(idle);
	
	//

	scene.init();
	global_init();

	atexit(destroy);

	glutMainLoop();

	destroy();

	return EXIT_SUCCESS;
}

