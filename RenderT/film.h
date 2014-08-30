#ifndef FILM_H
#define FILM_H

#include "GL/glee.h"

#include "integrator.h"
#include <stdlib.h>

class Film
{
public:
	
	static void init(unsigned x, unsigned y);
	static void clear();
	static void destroy();

	static void setRowColor(unsigned iRow, PixelIntegrator *pInts);
	static void appendRowColor(unsigned iRow, PixelIntegrator *pInts);

	static void render();

	static void bloom();

	//	Spatially Varying Nonlinear Scale from PBRT
	static void toneMap();

private:
	static void clampFilmColor(float *);
	static float getPixLum(float *);

	static void globalToneMap();
	static void localToneMap();

private:

	static GLuint nBufferId;

	static unsigned nWidth, nHeight;

	static void *_pFrameBuf;

	static void *_pTmpBuf;

	//	Tone-mapping
	static float vStdYWeight[3];
	static float *_pScaleBuf;
};
#endif