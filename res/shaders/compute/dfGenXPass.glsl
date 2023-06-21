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

uint readDistanceValue(uint idx)
{
    uint value = topLevelArray[idx];
    if (value >> 30 == 0)
        return value << 2 >> 2;
    else
        return 0; // chunk is filled
}

void main()
{
    // Two axis sweeps (+X and -X)
    uint idx = getChunkIdx(uvec3(0, gl_GlobalInvocationID.x, gl_GlobalInvocationID.y));
    uint prevValue = readDistanceValue(idx);
    for (int x = 1; x < (terrainSize.x >> 3); x++)
    {
        // compare current distance value to previous distance value and limit it to previous + 1
        idx = getChunkIdx(uvec3(x, gl_GlobalInvocationID.x, gl_GlobalInvocationID.y));
        uint thisValue = readDistanceValue(idx);
        if (prevValue + 1 < thisValue)
        {
            topLevelArray[idx] = prevValue + 1;
            thisValue = prevValue + 1;
        }
        prevValue = thisValue;
    }

    for (int x = int(terrainSize.x >> 3) - 2; x >= 0; x--)
    {
        // compare current distance value to previous distance value and limit it to previous + 1
        idx = getChunkIdx(uvec3(x, gl_GlobalInvocationID.x, gl_GlobalInvocationID.y));
        uint thisValue = readDistanceValue(idx);
        if (prevValue + 1 < thisValue)
        {
            topLevelArray[idx] = prevValue + 1;
            thisValue = prevValue + 1;
        }
        prevValue = thisValue;
    }
}
