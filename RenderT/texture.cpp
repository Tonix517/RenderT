#include "texture.h"

#include "IL/ilu.h"
#include "IL/ilut.h"

#include <algorithm>
using namespace std;

///
///
///

texture_s::texture_s()
	: data(NULL)
	, nWidth(0)
	, nHeight(0)
	, pNextMip(NULL)
	, id(0)
{
	
}

texture_s::~texture_s()
{
	if(data)
	{
		delete [] data;
		data = NULL;
	}
	if(pNextMip)
	{
		delete pNextMip;
	}
}

void texture_s::getTexAt(unsigned nWidInx, unsigned nHeiInx, vect3d &ret)
{
	assert( (nWidInx < nWidth) && (nHeiInx < nHeight));
	
	float *pColor = ((float*)data) + (nHeiInx * nWidth + nWidInx) * 3;
	ret[0] = pColor[0];
	ret[1] = pColor[1];
	ret[2] = pColor[2];
}

///
///
///

std::vector< texture_s* > TextureManager::_texVec;

int TextureManager::loadTexture(char *pPath, bool bBuildMipMap)
{
	assert(pPath);

	//	Loaded yet?
	vector<texture_s*>::iterator iter = _texVec.begin();
	for(; iter != _texVec.end(); iter ++)
	{
		if( (*iter)->path == pPath)
		{
			return (*iter)->id;
		}
	}

	//	Start to load
	ILuint nCurrTexImg = 0;
	ilGenImages(1, &nCurrTexImg);
	ilBindImage(nCurrTexImg);	

	{	
		//	Get Image Info
		if(ilLoadImage(pPath))
		{
			texture_s *pTex = NULL;
			unsigned nCount = 0;	
				
			if(iluBuildMipmaps())
			{
				pTex = new texture_s;
				texture_s *pNextTex = pTex;

				ilActiveMipmap(nCount ++);
				ILint nWidth = ilGetInteger(IL_IMAGE_WIDTH);
				ILint nHeight = ilGetInteger(IL_IMAGE_HEIGHT);

				while(nWidth != 1 && nHeight != 1)
				{
					void *data_buf = new float[ nWidth * nHeight * 3 * sizeof(float) ];
					ilCopyPixels( 0, 0, 0, nWidth, nHeight, 1, IL_RGB, IL_FLOAT, data_buf);			

					pNextTex->data = data_buf;
					pNextTex->nWidth = nWidth;
					pNextTex->nHeight = nHeight;
					pNextTex->path.assign(pPath);

					//	Update to next
					if( !ilActiveMipmap(nCount ++))
					{
						break;
					}

					pNextTex->pNextMip = new texture_s;
					pNextTex = pNextTex->pNextMip;

					nWidth = ilGetInteger(IL_IMAGE_WIDTH);
					nHeight = ilGetInteger(IL_IMAGE_HEIGHT);
				}

				int id = _texVec.size();
				pTex->id = id;
				_texVec.push_back(pTex);

				ilDeleteImages(1, &nCurrTexImg);
			}
			else
			{
				printf("Mipmap Building Failure... \n");
				return -1;
			}

			printf( "Texture Loaded : %s [Mipmap Count: %d]\n", pPath, --nCount);
			return pTex->id;
		}
		else
		{
			ILenum ilErr = ilGetError();
			//const char* sErrMsg = iluErrorString(IL_INVALID_ENUM);
			printf("Error in LoadImage: %d [%s]\n", ilErr, pPath);

			ilDeleteImages(1, &nCurrTexImg);
			return -1;
		}			
	}

	ilDeleteImages(1, &nCurrTexImg);
	return -1;	// loading failed
}

texture_s * TextureManager::getTexture(int id)
{
	assert(id != -1);

	return _texVec[id];
}

//	Mipmap search supported
int TextureManager::find(texture_s *pTex, int *nMipInx)
{
	for(int i = 0; i < _texVec.size(); i ++)
	{
		texture_s *pCurrTex = _texVec[i];
		int j = 0;
		while(pCurrTex)
		{
			if(pTex == pCurrTex)
			{
				*nMipInx = j;
				return i;
			}
			j ++;
			pCurrTex = pCurrTex->pNextMip;
		}
	}
	return -1;
}

void TextureManager::clear()
{
	vector<texture_s*>::iterator iter = _texVec.begin();
	for(; iter != _texVec.end(); iter ++)
	{
		delete *iter;
	}	

	_texVec.clear();
}
