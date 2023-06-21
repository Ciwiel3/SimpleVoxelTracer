#ifndef SIMPLEVOXELTRACER_TERRAIN_H
#define SIMPLEVOXELTRACER_TERRAIN_H

#include "cpmath.h"
#include "pool_allocator.h"

typedef struct Terrain {
    // top level array holding info about each 8x8x8 chunk:
    // leading 00 : chunk is empty and the next 30 bits are used for the distance field value (GPU memory only)
    // leading 10 : chunk is not empty and the next 30 bits are the index into the chunk pool
    // leading 11 : chunk is filled uniformly and the remaining 30 bits are the block ID
    // bitmask always matches the first bit of the top level array
    u32* topLevelArray;

    // pool allocators that hold all 8x8x8 chunks and their bitmasks
    PoolAllocator chunkPool;
    PoolAllocator chunkBitmaskPool;

    u32 width;
    u32 height;
    u32 widthChunkC;
    u32 heightChunkC;
    u32 chunkCount;

    bool dirty;
} Terrain;

void terrain_init(Terrain* terrain, u32 width, u32 height);

void terrain_destroy(Terrain* terrain);

void terrain_setBlock(Terrain* terrain, u32 x, u32 y, u32 z, u8 value);

u8 terrain_getBlock(Terrain* terrain, u32 x, u32 y, u32 z);

#endif //SIMPLEVOXELTRACER_TERRAIN_H
