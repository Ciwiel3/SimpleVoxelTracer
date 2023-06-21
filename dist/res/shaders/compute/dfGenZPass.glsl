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
    // Two axis sweeps (+Z and -Z)
    uint idx = getChunkIdx(uvec3(gl_GlobalInvocationID.x, gl_GlobalInvocationID.y, 0));
    uint prevValue = readDistanceValue(idx);
    for (int z = 1; z < (terrainSize.z >> 3); z++)
    {
        // compare current distance value to previous distance value and limit it to previous + 1
        idx = getChunkIdx(uvec3(gl_GlobalInvocationID.x, gl_GlobalInvocationID.y, z));
        uint thisValue = readDistanceValue(idx);
        if (prevValue + 1 < thisValue)
        {
            topLevelArray[idx] = prevValue + 1;
            thisValue = prevValue + 1;
        }
        prevValue = thisValue;
    }

    for (int z = int(terrainSize.z >> 3) - 2; z >= 0; z--)
    {
        // compare current distance value to previous distance value and limit it to previous + 1
        idx = getChunkIdx(uvec3(gl_GlobalInvocationID.x, gl_GlobalInvocationID.y, z));
        uint thisValue = readDistanceValue(idx);
        if (prevValue + 1 < thisValue)
        {
            topLevelArray[idx] = prevValue + 1;
            thisValue = prevValue + 1;
        }
        prevValue = thisValue;
    }
}
