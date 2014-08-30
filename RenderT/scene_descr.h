#ifndef SCENE_DESCR_H
#define SCENE_DESCR_H

#include "scene.h"
#include <string>

class SceneLoader
{
public:
	static void load(char *pPath, Scene &);

private:
	static void loadScene(std::string &, Scene &);
	static void loadCamera(std::string &, Scene &);
	static void loadLights(std::string &, Scene &);
	static void loadObjObject(std::string &, Scene &);
	static void loadPrmObject(std::string &, Scene &);
};

#endif