#include <memory.h>
#include <pthread.h>
#include "terrain.h"
#include "cplog.h"
#include "pool_allocator.h"

#define FNL_IMPL
#include "FastNoiseLite.h"

static void generate(Terrain* terrain);

void terrain_init(Terrain* terrain, u32 width, u32 height)
{
    if (width % 64 != 0 || height % 64 != 0)
        PANIC("Terrain dimensions must be multiples of 64");

    terrain->width = width;
    terrain->height = height;

    terrain->widthChunkC = terrain->width / 8;
    terrain->heightChunkC = terrain->height / 8;

    terrain->chunkCount = terrain->widthChunkC * terrain->widthChunkC * terrain->heightChunkC;


    terrain->topLevelArray = _mm_malloc(terrain->chunkCount * sizeof(u32), 64);

    memset(terrain->topLevelArray, 0, terrain->chunkCount * sizeof(u32));

    // initially this allocated ~32 mb of space
    u32 initialPoolSize = 65536;
    poolAllocatorCreate(&terrain->chunkPool, initialPoolSize, 512, NULL);
    poolAllocatorCreate(&terrain->chunkBitmaskPool, initialPoolSize, 64, NULL);

    terrain->dirty = true;

    generate(terrain);
}

void terrain_destroy(Terrain* terrain)
{
    _mm_free(terrain->topLevelArray);

    poolAllocatorDestroy(&terrain->chunkPool);
    poolAllocatorDestroy(&terrain->chunkBitmaskPool);
}

static INLINE u32 getChunkIdx(u32 x, u32 y, u32 z, u32 width, u32 height)
{
    u32 superChunkIdx = (((x >> 4) * (width >> 4) + (z >> 4)) * (height >> 4) + (y >> 4));
    u32 withinSuperChunkIdx = ((((x >> 3) & 1u) << 2) + (((z >> 3) & 1u) << 1) + ((y >> 3) & 1u));
    return (superChunkIdx << 3) + withinSuperChunkIdx;
}

static INLINE u32 getWithinChunkIdx(u32 x, u32 y, u32 z)
{
    return ((x & 0b111) * 64u) + ((z & 0b111) * 8u) + (y & 0b111);
}

static INLINE void setBit(u32* memory, u32 idx, bool value)
{
    u32 mask = (value ? 1u : 0u) << (31 - idx);
    *memory &= ~mask;
    *memory |= mask;
}

static INLINE u8 packColor(u8 r, u8 g, u8 b)
{
    r = r >> 5;
    g = g >> 5;
    b = b >> 6;

    if (r == 0 && g == 0 && b == 0)
    {
        r = 1;
        g = 1;
        b = 1;
    }

    return (r << 5) | (g << 2) | b;
}

void terrain_setBlock(Terrain* terrain, u32 x, u32 y, u32 z, u8 value)
{
#ifdef DEBUG_MODE
    if (x >= terrain->width || z >= terrain->width || y >= terrain->height)
    {
        LOG_WARNING("setBlock out of bounds.");
        return;
    }
#endif

    // read top level array
    u32 chunkIdx = getChunkIdx(x, y, z, terrain->width, terrain->height);
    u32 chunkVal = terrain->topLevelArray[chunkIdx];
    u32 check = chunkVal >> 30;
    chunkVal = chunkVal << 2 >> 2;

    // check if chunk is empty
    if (check == 0b00 && value == 0)
        return;

    // check if chunk is uniformly filled
    if (check == 0b11 && value == (chunkVal))
        return;

    u32 withinChunkIdx = getWithinChunkIdx(x, y, z);

    if (check == 0b10)
    {
        // chunk is already non uniformly filled, just set the correct byte
        u8* chunkData = poolAllocatorGet(&terrain->chunkPool, chunkVal);
        chunkData[withinChunkIdx] = value;

        // update bitmask
        u32* bitmaskData = poolAllocatorGet(&terrain->chunkBitmaskPool, chunkVal);
        setBit(&bitmaskData[withinChunkIdx / 32], withinChunkIdx % 32, value);

        // check if chunk is now uniform and can be deallocated / simplified
        if (memcmp(chunkData, chunkData + 1, 511) == 0)
        {
            u8 uniformValue = chunkData[0];
            poolAllocatorDealloc(&terrain->chunkPool, chunkVal);
            poolAllocatorDealloc(&terrain->chunkBitmaskPool, chunkVal);
            terrain->topLevelArray[chunkIdx] = uniformValue;

            // set appropriate flag
            if (uniformValue != 0)
                terrain->topLevelArray[chunkIdx] |= 0b11u << 30;
        }
    }
    else
    {
        // chunk must be newly allocated
        u32 newChunkIdx = poolAllocatorAlloc(&terrain->chunkPool);
        u8* chunkData = poolAllocatorGet(&terrain->chunkPool, newChunkIdx);
        memset(chunkData, chunkVal, 512);
        chunkData[withinChunkIdx] = value;
        terrain->topLevelArray[chunkIdx] = (0b10u << 30) | newChunkIdx;

        // update bitmask
        poolAllocatorAlloc(&terrain->chunkBitmaskPool);
        u32* bitmaskData = poolAllocatorGet(&terrain->chunkBitmaskPool, newChunkIdx);
        memset(chunkData, chunkVal == 0 ? 0 : 0xFF, 64);
        setBit(&bitmaskData[withinChunkIdx / 32], withinChunkIdx % 32, value);
    }

    terrain->dirty = true;
}

u8 terrain_getBlock(Terrain* terrain, u32 x, u32 y, u32 z)
{
#ifdef DEBUG_MODE
    if (x >= terrain->width || y >= terrain->height || z >= terrain->width)
    {
        LOG_WARNING("getBlock out of bounds.");
        return 0;
    }
#endif

    u32 chunkIdx = getChunkIdx(x, y, z, terrain->width, terrain->height);

    u32 chunkVal = terrain->topLevelArray[chunkIdx];
    u32 check = chunkVal >> 30;

    // check if chunk is empty
    if (check == 0b00)
        return 0;

    // check if chunk is uniformly filled
    if (check == 0b11)
        return chunkVal & 0xFF;

    // chunk is non uniformly filled
    u8* chunkData = poolAllocatorGet(&terrain->chunkPool, chunkVal << 2 >> 2);

    u32 withinChunkIdx = getWithinChunkIdx(x, y, z);

    return chunkData[withinChunkIdx];
}

static void generate(Terrain* terrain)
{
    srand(41233125);

    fnl_state noiseGen2D = fnlCreateState();
    noiseGen2D.noise_type = FNL_NOISE_OPENSIMPLEX2;
    noiseGen2D.fractal_type = FNL_FRACTAL_RIDGED;
    noiseGen2D.octaves = 3;
    noiseGen2D.seed = 41233125;
    noiseGen2D.frequency = 1;

    for (u32 cx = 0; cx < terrain->width / 8; cx++)
        for (u32 cz = 0; cz < terrain->width / 8; cz++)
        {
            u16 heightMap[8][8];
            for (u32 x = 0; x < 8; x++)
                for (u32 z = 0; z < 8; z++)
                {
                    heightMap[x][z] = 0.1 * terrain->height + 0.25 * terrain->height * (fnlGetNoise2D(&noiseGen2D, (cx*8 + x) * 0.005, (cz*8 + z) * 0.005) * 0.5 + 0.5);
                }

            for (u32 cy = 0; cy < terrain->height / 8; cy++)
            {
                u32 chunkIdx = getChunkIdx(cx * 8, cy * 8, cz * 8, terrain->width, terrain->height);
                bool chunkEmpty = true;

                u8* blockData;
                u32* bitmask;

                for (u32 dx = 0; dx < 8; dx++)
                    for (u32 dz = 0; dz < 8; dz++)
                    {
                        u32 x = cx * 8 + dx;
                        u32 z = cz * 8 + dz;
                        int height = min(8, heightMap[dx][dz] - cy * 8);

                        for (int dy = 0; dy < height; dy++)
                        {
                            u32 y = cy * 8 + dy;

//                            double noise = (fnlGetNoise3D(&noiseGen2D, x * 0.005, y * 0.005, z * 0.005) * 0.5 + 0.5) - (y / (float) terrain->height);
//                            if (noise <= 0)
//                                break;

                            if (chunkEmpty)
                            {
                                u32 poolIdx = poolAllocatorAlloc(&terrain->chunkPool);
                                poolAllocatorAlloc(&terrain->chunkBitmaskPool);

                                terrain->topLevelArray[chunkIdx] = (0b10 << 30) | poolIdx;

                                blockData = poolAllocatorGet(&terrain->chunkPool, poolIdx);
                                bitmask = poolAllocatorGet(&terrain->chunkBitmaskPool, poolIdx);

                                memset(blockData, 0, 512);
                                memset(bitmask, 0, 64);

                                chunkEmpty = false;
                            }

                            u32 blockIdx = getWithinChunkIdx(x, y, z);

                            u8 color = packColor(135, 135, 135);
                            if (y <= 0.25 * terrain->height && y >= heightMap[dx][dz] - 3)
                                color = packColor(86, 125, 70);
                            if (y <= 0.15 * terrain->height && y >= heightMap[dx][dz] - 3)
                                color = packColor(92,73,73);

                            blockData[blockIdx] = color;
                            setBit(&bitmask[blockIdx / 32], blockIdx % 32, 1);
                        }
                    }

                if (!chunkEmpty)
                {
                    if (memcmp(blockData, blockData + 1, 511) == 0)
                    {
                        u32 poolIdx = terrain->topLevelArray[chunkIdx] << 2 >> 2;
                        u8 uniformValue = blockData[0];
                        poolAllocatorDealloc(&terrain->chunkPool, poolIdx);
                        poolAllocatorDealloc(&terrain->chunkBitmaskPool, poolIdx);
                        terrain->topLevelArray[chunkIdx] = (0b11u << 30) | uniformValue;
                    }
                }
            }
        }
}
