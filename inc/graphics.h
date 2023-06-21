#ifndef SIMPLEVOXELTRACER_GRAPHICS_H
#define SIMPLEVOXELTRACER_GRAPHICS_H

#include "terrain.h"
typedef struct RenderSettings {

} RenderSettings;

void graphics_init(void);

void graphics_drawFrame(Terrain* terrain, vec3 camPos, vec3 forward);

void graphics_destroy(void);

void graphics_reloadShaders(void);

uvec2 graphics_getRes(void);

#endif //SIMPLEVOXELTRACER_GRAPHICS_H
