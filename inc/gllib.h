#ifndef FASTDRAW_GLLIB_H
#define FASTDRAW_GLLIB_H

#include "cpmath.h"

typedef struct Texture
{
    u32 handle;
    u32 internalFormat;
} Texture;

u32 gllib_makePipeline(const char* vertPath, const char* fragPath);
u32 gllib_makeCompute(const char* shaderPath);

Texture gllib_makeDefaultTexture(u32 width, u32 height, u32 glInternalFormat, u32 glFilter);
void gllib_destroyTexture(Texture* texture);
void gllib_bindTexture(const Texture* texture, u32 idx, u32 glUsage);

#endif //FASTDRAW_GLLIB_H
