#version 450 core
layout(local_size_x = 8, local_size_y = 8) in;

layout(std430, binding = 0) buffer top_level_array
{
    uint topLevelArray[];
};

layout(location=0) uniform uvec3 terrainSize;

uint getChunkIdx(uvec3 pos)
{
    uint superChunkIdx = (((pos.x >> 1) * (terrainSize.z >> 4) + (pos.z >> 1)) * (terrainSize.y >> 4) + (pos.y >> 1));
    uint withinSuperChunkIdx = (((pos.x & 1u) << 2) + ((pos.z & 1u) << 1) + (pos.y & 1u));
    return (superChunkIdx << 3) + withinSuperChunkIdx;
}

bool isChunkFilled(uint idx)
{
    return (topLevelArray[idx] >> 30) != 0;
}

void main()
{
    for (uint y = 0; y < (terrainSize.y >> 3); y++)
    {
        uint idx = getChunkIdx(uvec3(gl_GlobalInvocationID.x, y, gl_GlobalInvocationID.y));
        if (!isChunkFilled(idx))
        {
            // initialize every empty chunk with the max distance value
            topLevelArray[idx] = 0xFFF;
        }
    }
}
