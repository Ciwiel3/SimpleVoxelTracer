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

uint readDistanceValueFull(uint idx)
{
    uint value = topLevelArray[idx];
    if (value >> 30 == 0)
        return value << 2 >> 2;
    else
        return 0; // chunk is filled
}

uint readDistanceValueHalf(uint idx)
{
    uint value = topLevelArray[idx];
    if (value >> 30 == 0)
        return value << 2 >> 17;
    else
        return 0; // chunk is filled
}

void main()
{
    /*
    * This pass deviates from the X and Z pass and is ran last.
    * It outputs the normal manhattan distance field value to the least significant 15 bits of the chunk value.
    * Additionally it outputs the value BEFORE the final +Y pass to the 15 bits next to the first value.
    * This creates an "anisotropic" distance field,
    * where rays that travel in +Y direction can use the potentially larger second valuef or larger jumps.
    */

    // Two axis sweeps (-Y and +Y)
    uint idx = getChunkIdx(uvec3(gl_GlobalInvocationID.x, (terrainSize.y >> 3) - 1, gl_GlobalInvocationID.y));
    uint prevValue = readDistanceValueFull(idx);

    // move the first distance value 15 bits to the left
    if (prevValue != 0)
        topLevelArray[idx] = prevValue << 15;

    for (int y = int((terrainSize.y >> 3) - 2); y >= 0; y--)
    {
        idx = getChunkIdx(uvec3(gl_GlobalInvocationID.x, y, gl_GlobalInvocationID.y));
        uint thisValue = readDistanceValueFull(idx);
        prevValue = prevValue + 1 < thisValue ? min(0x7FFF, prevValue + 1) : thisValue;

        // always move the distance value 15 bits to the left, if the chunk is empty
        if (thisValue != 0)
            topLevelArray[idx] = prevValue << 15;
    }

    for (int y = 1; y < (terrainSize.y >> 3); y++)
    {
        idx = getChunkIdx(uvec3(gl_GlobalInvocationID.x, y, gl_GlobalInvocationID.y));
        uint thisValue = readDistanceValueHalf(idx);
        prevValue = prevValue + 1 < thisValue ? min(0x7FFF, prevValue + 1) : thisValue;

        // write the final DF value to the 15 least significant bits
        if (thisValue != 0)
            topLevelArray[idx] |= prevValue;
    }
}
