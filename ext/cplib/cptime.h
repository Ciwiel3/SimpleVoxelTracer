// Copyright (c) 2021, Lars Knof, All rights reserved.

#ifndef CPTIME_H
#define CPTIME_H

#ifdef __linux__
#define CLOCK_MONOTONIC 1
#endif

#include <stdint.h>
#include <time.h>

static uint64_t nclock(void)
{
    struct timespec time;
    clock_gettime(CLOCK_MONOTONIC, &time);

    static _Thread_local uint64_t
    INITIAL_SEC = 0;

    if (INITIAL_SEC == 0)
        INITIAL_SEC = time.tv_sec;

    return (time.tv_sec - INITIAL_SEC) * 1000000000ull + time.tv_nsec;
}

static inline uint64_t uclock(void)
{
    return nclock() / 1000ull;
}

static inline uint64_t mclock(void)
{
    return nclock() / 1000000ull;
}

#endif //CPTIME_H
