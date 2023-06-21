#ifndef SIMPLEVOXELTRACER_POOL_ALLOCATOR_H
#define SIMPLEVOXELTRACER_POOL_ALLOCATOR_H

#include "cplog.h"
#include "memory.h"
typedef struct PoolAllocator
{
    void* nextFree;
    u32 unused;
    u32 unitSize;

    void* memory;
    u32 maxSize;
    u32 size;
    bool ownsMemory;

} __attribute__((aligned(32))) PoolAllocator;

static INLINE void poolAllocatorFreeAll(PoolAllocator* poolAllocator)
{
    poolAllocator->size = 0;
    poolAllocator->unused = poolAllocator->maxSize;
    poolAllocator->nextFree = NULL;
}

static INLINE void poolAllocatorDestroy(PoolAllocator* poolAllocator)
{
    if (poolAllocator->ownsMemory)
        _mm_free(poolAllocator->memory);
    poolAllocator->maxSize = 0;
}

static void* poolAllocatorAllocPtr(PoolAllocator* poolAllocator)
{
    void* ptr;
        if (poolAllocator->nextFree != NULL)
        {
            ptr = poolAllocator->nextFree;
            poolAllocator->nextFree = (void*) *((uintptr_t*) poolAllocator->nextFree);
        } else if (poolAllocator->unused > 0)
        {
            ptr = (void*) (((uintptr_t) poolAllocator->memory) + (poolAllocator->maxSize - poolAllocator->unused) * poolAllocator->unitSize);
            poolAllocator->unused--;
        } else // allocator is full
        {
            if (poolAllocator->ownsMemory)
            {
                // resize by 2x
                poolAllocator->unused += poolAllocator->maxSize;
                poolAllocator->maxSize *= 2;
                void* oldMemory = poolAllocator->memory;
                poolAllocator->memory = _mm_malloc(poolAllocator->maxSize * poolAllocator->unitSize, 64);
                memcpy(poolAllocator->memory, oldMemory, poolAllocator->maxSize / 2 * poolAllocator->unitSize);
                _mm_free(oldMemory);

                ptr = (void*) (((uintptr_t) poolAllocator->memory) + (poolAllocator->maxSize - poolAllocator->unused) * poolAllocator->unitSize);
                poolAllocator->unused--;
            }
            else
            {
                LOG_ERROR("Pool Allocator is full!");
                return NULL;
            }
        }

        poolAllocator->size++;
        return ptr;
}

static INLINE u32 poolAllocatorAlloc(PoolAllocator* poolAllocator)
{
    void* ptr = poolAllocatorAllocPtr(poolAllocator);
    return ptr == NULL ? 0 : (((uintptr_t) ptr) - ((uintptr_t) poolAllocator->memory)) / poolAllocator->unitSize;
}

static INLINE void poolAllocatorDeallocPtr(PoolAllocator* poolAllocator, void* ptr)
{
    void** tmp = (void**) ptr;
    *tmp = poolAllocator->nextFree;
    poolAllocator->nextFree = ptr;
    poolAllocator->size--;
}

static INLINE void poolAllocatorDealloc(PoolAllocator* poolAllocator, u32 idx)
{
    poolAllocatorDeallocPtr(poolAllocator, (void*) (((uintptr_t) poolAllocator->memory) + idx * poolAllocator->unitSize));
}

static INLINE void* poolAllocatorGet(const PoolAllocator* poolAllocator, u32 idx)
{
    return (void*) (((uintptr_t) poolAllocator->memory) + idx * poolAllocator->unitSize);
}

// creates memory internally if memory = NULL
static void poolAllocatorCreate(PoolAllocator* allocator, u32 maxCount, u32 itemByteSize, void* memory)
{
    allocator->maxSize = maxCount;
    allocator->unitSize = itemByteSize;

    if (memory != NULL)
    {
        allocator->memory = memory;
        allocator->ownsMemory = false;
    }
    else
    {
        allocator->memory = _mm_malloc(((size_t) itemByteSize) * maxCount, 64);
        allocator->ownsMemory = true;
    }

    poolAllocatorFreeAll(allocator);
}

#endif //SIMPLEVOXELTRACER_POOL_ALLOCATOR_H
