#include "film.h"
#include "consts.h"
#include "ray.h"

#include <assert.h>
#include <stdio.h>

//	static vars init
void *Film::_pFrameBuf = 0;
void *Film::_pTmpBuf = 0;
GLuint Film::nBufferId = 0;
float *Film::_pScaleBuf = NULL;
unsigned Film::nWidth = 0, Film::nHeight = 0;

float Film::vStdYWeight[3] = { 0.212671f, 0.715160f, 0.072169f };

#define BUFFER_OFFSET(i) ((char *)NULL + (i))

//

void Film::init(unsigned x, unsigned y)
{
	nWidth = x; nHeight = y;
	assert(x > 0 && y > 0);

	//	memory setup
	assert(_pFrameBuf == 0);
	_pFrameBuf = malloc( sizeof(float) * 3 * nWidth * nHeight);	
	assert(_pFrameBuf);

	assert(_pTmpBuf == 0);
	_pTmpBuf = malloc( sizeof(float) * 3 * nWidth * nHeight);	
	assert(_pTmpBuf);

	assert(_pScaleBuf == 0);
	_pScaleBuf = (float *)malloc(sizeof(float) * nWidth * nHeight);
	assert(_pScaleBuf);

	clear();

	//	OGL setup
	if(GL_ARB_vertex_buffer_object)
	{
		glPixelStorei(GL_UNPACK_ALIGNMENT, sizeof(float));

		glGenBuffers(1, &nBufferId);
		glBindBuffer(GL_PIXEL_UNPACK_BUFFER, nBufferId);

		//	check error
		GLenum err = glGetError();
		if(err != GL_NO_ERROR)
		{
			printf("[GL ERROR] %s - %d : 0x%x\n", __FILE__, __LINE__, err);
		}
	}
	else
	{
		printf("[ERROR] OpenGL version is too low to support vertex buffer !\n");
	}

}

void Film::clear()
{
	for(int i = 0; i < nWidth; i ++)
	for(int j = 0; j < nHeight; j ++)
	{
		float v = 0;
		if( (i/128 + j/128) % 2 )
		{
			v = 1.f;
		}
		((float*)_pFrameBuf)[(j * nWidth + i) * 3 + 0] = v;
		((float*)_pFrameBuf)[(j * nWidth + i) * 3 + 1] = v;
		((float*)_pFrameBuf)[(j * nWidth + i) * 3 + 2] = v;
	}
}

void Film::destroy()
{
	if(_pFrameBuf)
	{
		free(_pFrameBuf);
		_pFrameBuf = NULL;
	}

	if(_pTmpBuf)
	{
		free(_pTmpBuf);
		_pTmpBuf = NULL;
	}

	if(_pScaleBuf)
	{
		free(_pScaleBuf);
		_pScaleBuf = NULL;
	}
}

void Film::setRowColor(unsigned iRow, PixelIntegrator *pInts)
{
	assert(iRow < WinHeight && pInts != NULL);

	for(int i = 0; i < WinWidth; i ++)
	{
		vect3d fCurrColor;
		pInts[i].getColor(fCurrColor);

		((float*)_pFrameBuf)[(iRow * nWidth + i) * 3 + 0] = fCurrColor[0];
		((float*)_pFrameBuf)[(iRow * nWidth + i) * 3 + 1] = fCurrColor[1];
		((float*)_pFrameBuf)[(iRow * nWidth + i) * 3 + 2] = fCurrColor[2];
	}
}

void Film::appendRowColor(unsigned iRow, PixelIntegrator *pInts)
{
	assert(iRow < WinHeight && pInts != NULL);

	for(int i = 0; i < WinWidth; i ++)
	{
		vect3d fCurrColor;
		pInts[i].getColor(fCurrColor);

		((float*)_pFrameBuf)[(iRow * nWidth + i) * 3 + 0] += fCurrColor[0];
		((float*)_pFrameBuf)[(iRow * nWidth + i) * 3 + 1] += fCurrColor[1];
		((float*)_pFrameBuf)[(iRow * nWidth + i) * 3 + 2] += fCurrColor[2];

		clampFilmColor(((float*)_pFrameBuf) + (iRow * nWidth + i) * 3);
	}
}

void Film::clampFilmColor(float *pColor)
{
	assert(pColor);

	for(int i = 0; i < 3; i ++)
	{
		if(pColor[i] > 1.f) pColor[i] = 1.f;
	}
}

void Film::render()
{
	assert(_pFrameBuf != NULL);	

	glClear(GL_COLOR_BUFFER_BIT);
	glRasterPos2i(0, 0);

	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, nBufferId);
	//	this call copys the pixel data to promise real-time update
	glBufferData( GL_PIXEL_UNPACK_BUFFER, 
				  3 * nWidth * nHeight * sizeof(float), 
				  _pFrameBuf, 
				  GL_DYNAMIC_DRAW); // this buffer is intended to be modified many times
									// and rendered many times

	glDrawPixels(nWidth, nHeight, GL_RGB, GL_FLOAT, BUFFER_OFFSET(0));

	//	check error
	GLenum err = glGetError();
	if(err != GL_NO_ERROR)
	{
		printf("[GL ERROR] %s - %d : 0x%x\n", __FILE__, __LINE__, err);
	}
}

void Film::bloom()
{
	printf("Blooming... %d \n", nBloomRad);

	int rad2 = nBloomRad * nBloomRad;

	for(int i = 0; i < nHeight; i ++)
	{
		for(int j = 0; j < nWidth; j ++)
		{
			float fAccumWeight = 0;
			float accumV[3] = {0};

			//	loop the circle
			for(int y = -nBloomRad; y <= nBloomRad; y ++)
			for(int x = -nBloomRad; x <= nBloomRad; x ++)
			{
				int nCurrX = j + x;
				int nCurrY = i + y;
				
				if( nCurrX >= 0 && nCurrX < nWidth && 
					nCurrY >= 0 && nCurrY < nHeight )
				{
					float dist2 = fabs(x * 1.f) * fabs(x * 1.f) + fabs(y * 1.f) * fabs(y * 1.f); 
					if(dist2 <= rad2)
					{
						float weight = pow((nBloomRad - sqrt(dist2)) / (nBloomRad * 1.f), 4);
						fAccumWeight += weight;

						for(int n = 0; n < 3; n ++)
						{
							accumV[n] += ((float *)_pFrameBuf)[(nCurrX + nCurrY * nWidth) * 3 + n] * weight;
						}
					}// if
				}//	if					
			}//	loop circle

			for(int n = 0; n < 3; n ++)
			{
				((float *)_pTmpBuf)[(i * nWidth + j) * 3 + n] = accumV[n] / fAccumWeight;
			}
					
		}//	for j

		//	copy back
		unsigned offset = 3 * nWidth * i;
		//memcpy((float *)_pFrameBuf + offset, (float *)_pTmpBuf + offset, 3 * nWidth * sizeof(float));
		for(int m = 0; m < 3 * nWidth; m ++)
		{
			float vOrig = *((float *)_pFrameBuf + offset + m);
			float vBlm  = *((float *)_pTmpBuf + offset + m);

			*((float *)_pFrameBuf + offset + m) = vBlm * fBloomWeight + (1 - fBloomWeight) * vOrig;
			if(m % 3 == 0)
			{
				clampFilmColor(((float *)_pFrameBuf + offset + m));
			}
		}

		printf("\b\b\b\b\b%.2f ", i * 1.f / nHeight);
	}//	for i
	
	printf("\n DONE \n");
}

//	Spatially Varying Nonlinear Scale from PBRT

float Film::getPixLum(float *pPix)
{
	return 683.f * ( *(pPix + 0) * vStdYWeight[0] + 
					 *(pPix + 1) * vStdYWeight[1] + 
					 *(pPix + 2) * vStdYWeight[2] );
}


void Film::toneMap()
{
	switch(iTMapType)
	{
	case GLOBAL_TMAP:
		globalToneMap();
		break;
	case LOCAL_TMAP:
		localToneMap();
		break;
	}
}

void Film::localToneMap()
{
	printf("Not implemented... \n");
}

void Film::globalToneMap()
{
	printf("Global Tone-mapping... \n");

	//	Get Y Info
	float maxY = -1.f;
	float *pTmpYBuf = (float *)malloc(sizeof(float) * nWidth * nHeight);
	assert(pTmpYBuf);

	//float Ywa = 0.f;
	for(int i = 0; i < nHeight; i ++)
	for(int j = 0; j < nWidth; j ++)
	{
		float vy = getPixLum((float *)_pFrameBuf + (i * nWidth + j) * 3) / 683.f;
		if(vy > maxY)
		{
			maxY = vy;
		}

		//if(vy > 0) Ywa += logf(vy);
		*(pTmpYBuf + (i * nWidth + j)) = vy;
	}
	printf("Max : %.3f \n", maxY);

	//	Compute Scales
	//
	float invY2 = maxY > 0 ? (1.f/(maxY * maxY)) : 1;
	//Ywa = expf(Ywa / (nWidth * nHeight * 1.f));
	//invY2 = 1.f / (Ywa * Ywa);

	for(int i = 0; i < nHeight; i ++)
	for(int j = 0; j < nWidth; j ++)
	{
		float ys = *(pTmpYBuf + (i * nWidth + j));
		*(_pScaleBuf + (i * nWidth + j)) = (1.f + ys * invY2) / (1.f + ys);
	}

	//	Apply pixel values
	//
	float displayTo01 = 683.f / fMaxDisplayY;

	for(int i = 0; i < nHeight; i ++)
	{
		for(int j = 0; j < nWidth; j ++)
		{
			for(int n = 0; n < 3; n ++)
			{
				*((float *)_pFrameBuf + (i * nWidth + j) * 3 + n) *= *(_pScaleBuf + (i * nWidth + j)) * displayTo01;
			}
			clampFilmColor((float *)_pFrameBuf + (i * nWidth + j) * 3);
		}
		
		printf("\b\b\b\b\b%.2f", i * 1.f / nHeight);
	}

	free(pTmpYBuf);

	printf("\nDONE \n");
}