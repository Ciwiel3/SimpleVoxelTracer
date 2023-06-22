#version 460
//#extension GL_ARB_shader_clock : enable
layout(local_size_x = 8,  local_size_y = 8) in;

layout(rgba8, binding = 0) uniform writeonly image2D outImage;

uniform uvec2 screenSize;
uniform uvec3 terrainSize;
uniform vec3 camPos;
uniform mat4 viewMat;
uniform mat4 projMat;

layout(std430, binding = 0) readonly buffer top_level_array
{
    uint topLevelArray[];
};

layout(std430, binding = 1) readonly buffer chunk_pool_data
{
    uint chunkPoolData[];
};

layout(std430, binding = 2) readonly buffer chunk_pool_bits
{
    uint chunkPoolBits[];
};

layout(std430, binding = 3) buffer photon_map
{
    uint photonMap[];
};

struct RayHit {
    vec3 hitPos;
    uint hitId;
    uint faceId;
};

// the face ID is returned as a single number, this array converts it to a normal
// (useful for passing the normal around / storing it in just 3 bits)
const vec3 normals[] = {
    vec3(-1,0,0),
    vec3(1,0,0),
    vec3(0,-1,0),
    vec3(0,1,0),
    vec3(0,0,-1),
    vec3(0,0,1)
};

uint getChunkIdx(uvec3 pos)
{
    // "super chunks" are 64x64x64 units, that only exist conceptually for memory alignment (bitmask)
    // they aren't sparsely allocated, don't have their own bitmask and aren't considered during tracing
    uint superChunkIdx = (((pos.x >> 4) * (terrainSize.z >> 4) + (pos.z >> 4)) * (terrainSize.y >> 4) + (pos.y >> 4));
    uint withinSuperChunkIdx = ((((pos.x >> 3) & 1u) << 2) + (((pos.z >> 3) & 1u) << 1) + ((pos.y >> 3) & 1u));
    return (superChunkIdx << 3) + withinSuperChunkIdx;
}

RayHit intersectTerrain(vec3 rayPos, vec3 rayDir)
{
    // delta to avoid grid aligned rays
    if (rayDir.x == 0)
        rayDir.x = 0.001;
    if (rayDir.y == 0)
        rayDir.y = 0.001;
    if (rayDir.z == 0)
        rayDir.z = 0.001;

    rayDir = normalize(rayDir);

    // some helper values used for DDA steps
    ivec3 raySign = ivec3(sign(rayDir));
    ivec3 rayPositivity = (1 + raySign) >> 1;
    vec3 rayInverse = 1 / rayDir;

    float inverseDirY = 1.0f / rayDir.y;

    // converting manhattan distance to euclidean is direction dependent,
    // precalculate this factor, since the direction doesn't change
    float distanceFactor = 0.9999f / dot(rayDir, raySign);

    int minIdx = 1;
    vec3 t = vec3(1);

    // these variables keep track of the current position (rayPos = gridCoords + withinGridCoords)
    ivec3 gridCoords = ivec3(rayPos);
    vec3 withinGridCoords = rayPos - gridCoords;

    ivec3 bounds = ivec3(terrainSize);

    // this is the step size used for DDA, it's dynamically changed inside the traversal loop
    // the actual number of blocks (NxNxN) that are stepped over is N = 2^stepSize
    // default is single block steps
    uint stepSize = 0;

    // the distance field contains 2 values: one general distance value and one that ignores any solid blocks that are BELOW
    // in the case that rayDir.y > 0, we can
    uint dfShift = rayDir.y < 0 ? 0 : 15;

   // this counter is set to equal the DF value and incremented each frame
   // the bitmask for chunks is used while this value is below a threshold to be more cache efficient
   // setting it to a high number (e.g. 100) means the DF is always read on the first step
    const uint dfReadThreshold = 0;

    while ((!any(greaterThanEqual(gridCoords, bounds)) && !any(lessThan(gridCoords, ivec3(0)))))
    {
        // calculate the index of the current chunk in the top level array
        uvec3 pos = uvec3(gridCoords) + uvec3(withinGridCoords);
        uint chunkIdx = getChunkIdx(pos);

        // read the value of the current chunk
        // the first two bits (check) indicate whether the chunk is:
        // empty - the remaining bits are the distance field value
        // filled - the remaining bits are the uniform block ID
        // normal - the remaining bits are the index of the chunk data in the chunk data pool
        uint chunkVal = topLevelArray[chunkIdx];
        uint check = chunkVal >> 30u;
        chunkVal = chunkVal << 2 >> 2;

        // check if chunk is not empty
        if (check != 0)
        {
            // check if chunk is non uniformly filled
            uint blockId = chunkVal;
            if (check == 2u)
            {
                // calc the index of the current block inside the chunk
                uint withinChunkIdx = ((pos.x & 7u) << 6) + ((pos.z & 7u) << 3) + (pos.y & 7u);
                uint poolIndex = (chunkVal << 9) + withinChunkIdx;

                // check the current block in the chunk data pool
                if (((chunkPoolBits[poolIndex >> 5] >> (31 - (withinChunkIdx & 31u))) & 1u) == 0)
                {
                    blockId = 0;
                }
                else
                {
                    // read the block id from the data pool
                    // the shifting after reading 4 bytes, takes into account endianess
                    blockId = (chunkPoolData[poolIndex >> 2] >> (8 * (poolIndex & 3u))) & 0xFFu;
                }
            }

            // return on hit (single block or uniformly filled chunk)
            if (blockId != 0)
            {
                // calculate the normal / face that was hit from the last minIdx that's calculated during DDA
                // this works because the DDA keeps track of the axis over which it last stepped
                uint faceId = 0;
                if (minIdx == 0)
                {
                    faceId = -rayPositivity.x + 2;
                }
                if (minIdx == 1)
                {
                    faceId = -rayPositivity.y + 4;
                }
                if (minIdx == 2)
                {
                    faceId = -rayPositivity.z + 6;
                }

                // return the hit
                return RayHit(vec3(gridCoords + withinGridCoords), blockId, faceId);
            }
            else
            {
                // no hit, but because the current chunk is filled normally, change to single block steps
                if (stepSize != 0)
                {
                    gridCoords += ivec3(withinGridCoords);
                    withinGridCoords = fract(withinGridCoords);
                    stepSize = 0;
                }
            }
        }
        else
        {
            // the chunk is empty --> read the distance field value and convert it to euclidean
            float dfValue1 = (((chunkVal & 0x7FFFu) - 1) << 3) * distanceFactor;
            float dfValue2 = ((((chunkVal >> 15) & 0x7FFFu) - 1) << 3) * distanceFactor;

            float dfValue = dfValue2;
            if (rayDir.y < 0)
            {
                float distToBottomOfChunk = (withinGridCoords.y + (gridCoords.y & 7)) * inverseDirY;
                dfValue = max(dfValue1, min(dfValue2, distToBottomOfChunk));
            }

            // if the DF value is at least 1, jump by that amount
            if (dfValue > 0)
            {
                rayPos = gridCoords + withinGridCoords + rayDir * dfValue;
                gridCoords = ivec3(rayPos);
                withinGridCoords = fract(rayPos);
                stepSize = 0;

                // we could take an additional step here, since we safely jumped into an empty voxel (DF value -1)
                // benchmarking showed that it's not worth it, so we terminate this step and check the new position instead
                continue;
            }

            // ray is very close to a filled chunk
            // make the next DDA step at the 8x8x8 chunk scale
            if (stepSize != 3)
            {
                withinGridCoords += gridCoords & 7;
                gridCoords -= gridCoords & 7;
                stepSize = 3;
            }
        }

        // do DDA step at appropriate scale (0 = single block, 3 = 8x8x8 chunk)
        // first we find the distance to the voxel border
        t = ((rayPositivity << stepSize) - withinGridCoords) * rayInverse;

        // determine the nearest axis (this is the axis on which we will cross the voxel border)
        minIdx = t.x < t.y ? (t.x < t.z ? 0 : 2) : (t.y < t.z ? 1 : 2);

        // increment / decrement the voxel border on the determined axis
        gridCoords[minIdx] += int(raySign[minIdx] << stepSize);

        // advance the ray (within grid coords) by the amount stepped
        // (this updates the other two dimensions that we didn't account for in the grid coord increment)
        withinGridCoords += rayDir * t[minIdx];

        // set the within voxel coord of the axis on which we stepped statically
        // this sets it to either 0 or 0.999, depending on the direction of the step
        // this ensures that we don't skip a block or get stuck on a border because of floating point issues
        withinGridCoords[minIdx] = ((1 - rayPositivity[minIdx]) << stepSize) * 0.999f;
    }

    // nothing was hit, but ray has exited bounds --> return 0
    return RayHit(vec3(gridCoords + withinGridCoords), 0, 0);
}

vec3 getRayDir(ivec2 screenPos)
{
    vec2 screenSpace = (screenPos + vec2(0.5)) / vec2(screenSize);
	vec4 clipSpace = vec4(screenSpace * 2.0f - 1.0f, -1.0, 1.0);
	vec4 eyeSpace = vec4(vec2(inverse(projMat) * clipSpace), -1.0, 0.0);
	return normalize(vec3(inverse(viewMat) * eyeSpace));
}

float AABBIntersect(vec3 bmin, vec3 bmax, vec3 orig, vec3 invdir)
{
    vec3 t0 = (bmin - orig) * invdir;
    vec3 t1 = (bmax - orig) * invdir;

    vec3 vmin = min(t0, t1);
    vec3 vmax = max(t0, t1);

    float tmin = max(vmin.x, max(vmin.y, vmin.z));
    float tmax = min(vmax.x, min(vmax.y, vmax.z));

    if (!(tmax < tmin) && (tmax >= 0))
        return max(0, tmin);
    return -1;
}

void main()
{
    // make sure current thread is inside the window bounds
    if (any(greaterThanEqual(gl_GlobalInvocationID.xy, screenSize)))
        return;

    // calc ray direction for current pixel
    vec3 rayDir = getRayDir(ivec2(gl_GlobalInvocationID.xy));

    vec3 rayPos = camPos;

    RayHit hit = RayHit(vec3(0), 0, 0);

    // check if the camera is outside the voxel volume
    float intersect = AABBIntersect(vec3(0), vec3(terrainSize - 1), camPos, 1.0f / rayDir);
    if (intersect > 0)
    {
        // calc ray start pos
        rayPos += rayDir * (intersect + 0.001);
    }

    // intersect the ray agains the terrain if it crosses the terrain volume
    vec3 colorTime = vec3(0);
    if (intersect >= 0)
    {
//        uvec2 start = clock2x32ARB();

        hit = intersectTerrain(rayPos, rayDir);

//        uvec2 end = clock2x32ARB();
//        uint time = end.x - start.x;
//        colorTime = vec3(time, 0, 0) / 1000000.0f;
    }

    // choose color (sky or voxel color)
    vec3 color = vec3(0.69, 0.88, 0.90);
    if (hit.hitId != 0)
    {
        vec3 normal = normals[hit.faceId - 1];

        // the color is packed as R3G3B2, a color palette would be preferable
        color = vec3((hit.hitId >> 5) / 7.0f, ((hit.hitId >> 2) & 7u) / 7.0f, (hit.hitId & 3u) / 3.0f);

        // simple normal based light
        color *= vec3(abs(dot(normal, normalize(vec3(1, 3, 1.5)))));
    }

    // output color to texture

    // uncomment the following line to color pixels based on the time it took to compute them
    // color = colorTime;
    imageStore(outImage, ivec2(gl_GlobalInvocationID.xy), vec4(color, 1));
}
