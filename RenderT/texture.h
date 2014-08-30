#ifndef TEXTURE_H
#define TEXTURE_H

#include "vector.h"

#include <string>
#include <vector>

enum TexMapType {STRAIGHT = 0, STRETCH = 1, REPEAT = 2};

struct texture_s
{

	texture_s();
	~texture_s();

	//

	void *data;

	unsigned nWidth;
	unsigned nHeight;

	texture_s * pNextMip;

	unsigned id;
	std::string path;

	//

	void getTexAt(unsigned nWidInx, unsigned nHeiInx, vect3d &ret);
};

//
//
//

class TextureManager
{
public:

	static int loadTexture(char *pPath, bool bBuildMipMap = false);
	static texture_s *getTexture(int id);
	static unsigned getTextureCount()
	{	return _texVec.size();	}

	static int find(texture_s *pTex, int *nMipInx);

	static void clear();

private:

	static std::vector< texture_s* > _texVec;

};

#endif