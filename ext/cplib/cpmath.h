#ifndef CPMATH_H
#define CPMATH_H

#include <tmmintrin.h>
#include <smmintrin.h>
#include <immintrin.h>
#include <stdint.h>
#include <math.h>
#include "stdbool.h"

typedef uint8_t u8;
typedef uint16_t u16;
typedef uint32_t u32;
typedef uint64_t u64;

typedef int8_t i8;
typedef int16_t i16;
typedef int32_t i32;
typedef int64_t i64;

#define CP_M_PI		3.14159265358979323846

#if defined(__GNUC__) || defined(__clang__)
    #define ALIGN(x) __attribute__ ((aligned(x)))
    #define FORCE_INLINE __attribute__ ((always_inline)) inline
    #define INLINE __attribute__ ((always_inline)) inline
#elif defined(_MSC_VER)
    #define ALIGN(x) __declspec(align(x))
    #define FORCE_INLINE __forceinline inline
    #define INLINE __forceinline inline
#else
#  error "Unknown Compiler"
#endif

/**
 * Settings
 */

/*
 * NOTE: this library stores vec3s as vec4s (16 bytes).
 *
 * Requirements:
 *  -   SSE2 is required and there is no fallback.
 *      I might add support to disable it at compile time in the future (not currently planned).
 *
 *  -   FMA instruction set is required as well (FMA3). There should also be an option to disable this, but it's not yet implemented.
 *
 *  -   C11 (or equivalent compiler support for _Generic).
 *      If you do not have support for this, feel free to disable the macros at the bottom and use the explicit functions.
 *
 *
 * ###################
 * ##### VECTORS #####
 * ###################
 *
 * Feel free to adjust the type / function names in the alias sections at the bottom if you prefer or have naming conflicts.
 * Default type naming / usage (same as glsl):
 *  vec4, vec3, vec2
 *  ivec4, ivec3, ivec2
 *  uvec4, uvec3, uvec2
 *
 * Functions (using C11 _Generic):
 *  add(), sub(), mul(), div() (<- these support scalar parameters)
 *  dot(), normalize(), cross(), length()
 *
 * -    Operations on vec3s and vec4s use SSE intrinsics, except for integer division and dot.
 * -    Operations on vec2s use regular arithmetic operators for each component.
 *
 * 
 * 
 * ####################
 * ##### MATRICES #####
 * ####################
 *
 *  
 *
 */

/**
 * Macros
 */

/**
 * Types
 */

typedef union ALIGN(16) cp_vec4
{
    struct
    {
        float x;
        float y;
        float z;
        float w;
    };
    struct
    {
        float r;
        float g;
        float b;
        float a;
    };
    float arr[4];
    __m128 i;
} cp_vec4;

typedef union ALIGN(16) cp_vec3
{
    struct
    {
        float x;
        float y;
        float z;
    };
    struct
    {
        float r;
        float g;
        float b;
    };
    float arr[3];
    __m128 i;
} cp_vec3;

typedef union ALIGN(8) cp_vec2
{
    struct
    {
        float x;
        float y;
    };
    struct
    {
        float r;
        float g;
    };
    float arr[2];
} cp_vec2;


typedef union ALIGN(16) cp_ivec4
{
    struct
    {
        int32_t x;
        int32_t y;
        int32_t z;
        int32_t w;
    };
    struct
    {
        int32_t r;
        int32_t g;
        int32_t b;
        int32_t a;
    };
    int32_t arr[4];
    __m128i i;
} cp_ivec4;

typedef union ALIGN(16) cp_ivec3
{
    struct
    {
        int32_t x;
        int32_t y;
        int32_t z;
    };
    struct
    {
        int32_t r;
        int32_t g;
        int32_t b;
    };
    int32_t arr[3];
    __m128i i;
} cp_ivec3;

typedef union ALIGN(8) cp_ivec2
{
    struct
    {
        int32_t x;
        int32_t y;
    };
    struct
    {
        int32_t r;
        int32_t g;
    };
    int32_t arr[2];
} cp_ivec2;


typedef union ALIGN(16) cp_uvec4
{
    struct
    {
        uint32_t x;
        uint32_t y;
        uint32_t z;
        uint32_t w;
    };
    struct
    {
        uint32_t r;
        uint32_t g;
        uint32_t b;
        uint32_t a;
    };
    uint32_t arr[4];
    __m128i i;
} cp_uvec4;

typedef union ALIGN(16) cp_uvec3
{
    struct
    {
        uint32_t x;
        uint32_t y;
        uint32_t z;
    };
    struct
    {
        uint32_t r;
        uint32_t g;
        uint32_t b;
    };
    uint32_t arr[3];
    __m128i i;
} cp_uvec3;

typedef union ALIGN(8) cp_uvec2
{
    struct
    {
        uint32_t x;
        uint32_t y;
    };
    struct
    {
        uint32_t r;
        uint32_t g;
    };
    uint32_t arr[2];
} cp_uvec2;

/**
 * Functions
 */
// add
static FORCE_INLINE cp_vec4 cp_vec4_add(cp_vec4 v1, cp_vec4 v2)
{
    return (cp_vec4) {
        .i = _mm_add_ps(v1.i, v2.i)
    };
}

static FORCE_INLINE cp_vec3 cp_vec3_add(cp_vec3 v1, cp_vec3 v2)
{
    return (cp_vec3) {
        .i = _mm_add_ps(v1.i, v2.i)
    };
}

static FORCE_INLINE cp_vec2 cp_vec2_add(cp_vec2 v1, cp_vec2 v2)
{
    return (cp_vec2) {
        .x = v1.x + v2.x,
        .y = v1.y + v2.y
    };
}


static FORCE_INLINE cp_ivec4 cp_ivec4_add(cp_ivec4 v1, cp_ivec4 v2)
{
    return (cp_ivec4) {
        .i = _mm_add_epi32(v1.i, v2.i)
    };
}

static FORCE_INLINE cp_ivec3 cp_ivec3_add(cp_ivec3 v1, cp_ivec3 v2)
{
    return (cp_ivec3) {
        .i = _mm_add_epi32(v1.i, v2.i)
    };
}

static FORCE_INLINE cp_ivec2 cp_ivec2_add(cp_ivec2 v1, cp_ivec2 v2)
{
    return (cp_ivec2) {
        .x = v1.x + v2.x,
        .y = v1.y + v2.y
    };
}


static FORCE_INLINE cp_uvec4 cp_uvec4_add(cp_uvec4 v1, cp_uvec4 v2)
{
    return (cp_uvec4) {
        .i = _mm_add_epi32(v1.i, v2.i)
    };
}

static FORCE_INLINE cp_uvec3 cp_uvec3_add(cp_uvec3 v1, cp_uvec3 v2)
{
    return (cp_uvec3) {
        .i = _mm_add_epi32(v1.i, v2.i)
    };
}

static FORCE_INLINE cp_uvec2 cp_uvec2_add(cp_uvec2 v1, cp_uvec2 v2)
{
    return (cp_uvec2) {
        .x = v1.x + v2.x,
        .y = v1.y + v2.y
    };
}


static FORCE_INLINE cp_vec4 cp_vec4_adds1(cp_vec4 v1, float s)
{
    return (cp_vec4) {
        .i = _mm_add_ps(v1.i, _mm_set1_ps(s))
    };
}

static FORCE_INLINE cp_vec3 cp_vec3_adds1(cp_vec3 v1, float s)
{
    return (cp_vec3) {
        .i = _mm_add_ps(v1.i, _mm_set1_ps(s))
    };
}

static FORCE_INLINE cp_vec2 cp_vec2_adds1(cp_vec2 v1, float s)
{
    return (cp_vec2) {
        .x = v1.x + s,
        .y = v1.y + s
    };
}


static FORCE_INLINE cp_ivec4 cp_ivec4_adds1(cp_ivec4 v1, int32_t s)
{
    return (cp_ivec4) {
        .i = _mm_add_epi32(v1.i, _mm_set1_epi32(s))
    };
}

static FORCE_INLINE cp_ivec3 cp_ivec3_adds1(cp_ivec3 v1, int32_t s)
{
    return (cp_ivec3) {
        .i = _mm_add_epi32(v1.i, _mm_set1_epi32(s))
    };
}

static FORCE_INLINE cp_ivec2 cp_ivec2_adds1(cp_ivec2 v1, int32_t s)
{
    return (cp_ivec2) {
        .x = v1.x + s,
        .y = v1.y + s
    };
}


static FORCE_INLINE cp_uvec4 cp_uvec4_adds1(cp_uvec4 v1, uint32_t s)
{
    return (cp_uvec4) {
        .i = _mm_add_epi32(v1.i, _mm_set1_epi32(s))
    };
}

static FORCE_INLINE cp_uvec3 cp_uvec3_adds1(cp_uvec3 v1, uint32_t s)
{
    return (cp_uvec3) {
        .i = _mm_add_epi32(v1.i, _mm_set1_epi32(s))
    };
}

static FORCE_INLINE cp_uvec2 cp_uvec2_adds1(cp_uvec2 v1, uint32_t s)
{
    return (cp_uvec2) {
        .x = v1.x + s,
        .y = v1.y + s
    };
}


static FORCE_INLINE cp_vec4 cp_vec4_adds2(float s, cp_vec4 v1)
{
    return (cp_vec4) {
        .i = _mm_add_ps(v1.i, _mm_set1_ps(s))
    };
}

static FORCE_INLINE cp_vec3 cp_vec3_adds2(float s, cp_vec3 v1)
{
    return (cp_vec3) {
        .i = _mm_add_ps(v1.i, _mm_set1_ps(s))
    };
}

static FORCE_INLINE cp_vec2 cp_vec2_adds2(float s, cp_vec2 v1)
{
    return (cp_vec2) {
        .x = v1.x + s,
        .y = v1.y + s
    };
}


static FORCE_INLINE cp_ivec4 cp_ivec4_adds2(int32_t s, cp_ivec4 v1)
{
    return (cp_ivec4) {
        .i = _mm_add_epi32(v1.i, _mm_set1_epi32(s))
    };
}

static FORCE_INLINE cp_ivec3 cp_ivec3_adds2(int32_t s, cp_ivec3 v1)
{
    return (cp_ivec3) {
        .i = _mm_add_epi32(v1.i, _mm_set1_epi32(s))
    };
}

static FORCE_INLINE cp_ivec2 cp_ivec2_adds2(int32_t s, cp_ivec2 v1)
{
    return (cp_ivec2) {
        .x = v1.x + s,
        .y = v1.y + s
    };
}


static FORCE_INLINE cp_uvec4 cp_uvec4_adds2(uint32_t s, cp_uvec4 v1)
{
    return (cp_uvec4) {
        .i = _mm_add_epi32(v1.i, _mm_set1_epi32(s))
    };
}

static FORCE_INLINE cp_uvec3 cp_uvec3_adds2(uint32_t s, cp_uvec3 v1)
{
    return (cp_uvec3) {
        .i = _mm_add_epi32(v1.i, _mm_set1_epi32(s))
    };
}

static FORCE_INLINE cp_uvec2 cp_uvec2_adds2(uint32_t s, cp_uvec2 v1)
{
    return (cp_uvec2) {
        .x = v1.x + s,
        .y = v1.y + s
    };
}

// FMA
static FORCE_INLINE cp_vec4 cp_vec4_fma(cp_vec4 v1, cp_vec4 v2, cp_vec4 v3)
{
    return (cp_vec4) {
        .i = _mm_fmadd_ps(v1.i, v2.i, v3.i)
    };
}

static FORCE_INLINE cp_vec3 cp_vec3_fma(cp_vec3 v1, cp_vec3 v2, cp_vec3 v3)
{
    return (cp_vec3) {
        .i = _mm_fmadd_ps(v1.i, v2.i, v3.i)
    };
}

static FORCE_INLINE cp_vec2 cp_vec2_fma(cp_vec2 v1, cp_vec2 v2, cp_vec2 v3)
{
    return (cp_vec2) {
        .x = v1.x * v2.x + v3.x,
        .y = v1.y * v2.y + v3.y
    };
}


static FORCE_INLINE cp_vec4 cp_vec4_fmas1(float s, cp_vec4 v1, cp_vec4 v2)
{
    return (cp_vec4) {
        .i = _mm_fmadd_ps(v1.i, _mm_set1_ps(s), v2.i)
    };
}

static FORCE_INLINE cp_vec3 cp_vec3_fmas1(float s, cp_vec3 v1, cp_vec3 v2)
{
    return (cp_vec3) {
        .i = _mm_fmadd_ps(v1.i, _mm_set1_ps(s), v2.i)
    };
}

static FORCE_INLINE cp_vec2 cp_vec2_fmas1(float s, cp_vec2 v1, cp_vec2 v2)
{
    return (cp_vec2) {
        .x = v1.x * s + v2.x,
        .y = v1.y * s + v2.y
    };
}


static FORCE_INLINE cp_vec4 cp_vec4_fmas2(cp_vec4 v1, float s, cp_vec4 v2)
{
    return (cp_vec4) {
        .i = _mm_fmadd_ps(v1.i, _mm_set1_ps(s), v2.i)
    };
}

static FORCE_INLINE cp_vec3 cp_vec3_fmas2(cp_vec3 v1, float s, cp_vec3 v2)
{
    return (cp_vec3) {
        .i = _mm_fmadd_ps(v1.i, _mm_set1_ps(s), v2.i)
    };
}

static FORCE_INLINE cp_vec2 cp_vec2_fmas2(cp_vec2 v1, float s, cp_vec2 v2)
{
    return (cp_vec2) {
        .x = v1.x * s + v2.x,
        .y = v1.y * s + v2.y
    };
}


static FORCE_INLINE cp_vec4 cp_vec4_fmas3(cp_vec4 v1, cp_vec4 v2, float s)
{
    return (cp_vec4) {
        .i = _mm_fmadd_ps(v1.i, v2.i, _mm_set1_ps(s))
    };
}

static FORCE_INLINE cp_vec3 cp_vec3_fmas3(cp_vec3 v1, cp_vec3 v2, float s)
{
    return (cp_vec3) {
        .i = _mm_fmadd_ps(v1.i, v2.i, _mm_set1_ps(s))
    };
}

static FORCE_INLINE cp_vec2 cp_vec2_fmas3(cp_vec2 v1, cp_vec2 v2, float s)
{
    return (cp_vec2) {
        .x = v1.x * v2.x + s,
        .y = v1.y * v2.y + s
    };
}


static FORCE_INLINE cp_vec4 cp_vec4_fmas12(float s1, float s2, cp_vec4 v)
{
    return (cp_vec4) {
        .i = _mm_fmadd_ps(_mm_set1_ps(s1), _mm_set1_ps(s2), v.i)
    };
}

static FORCE_INLINE cp_vec3 cp_vec3_fmas12(float s1, float s2, cp_vec3 v)
{
    return (cp_vec3) {
        .i = _mm_fmadd_ps(_mm_set1_ps(s1), _mm_set1_ps(s2), v.i)
    };
}

static FORCE_INLINE cp_vec2 cp_vec2_fmas12(float s1, float s2, cp_vec2 v)
{
    return (cp_vec2) {
        .x = s1 * s2 + v.x,
        .y = s1 * s2 + v.y
    };
}


static FORCE_INLINE cp_vec4 cp_vec4_fmas13(float s1, cp_vec4 v, float s2)
{
    return (cp_vec4) {
        .i = _mm_fmadd_ps(_mm_set1_ps(s1), v.i, _mm_set1_ps(s2))
    };
}

static FORCE_INLINE cp_vec3 cp_vec3_fmas13(float s1, cp_vec3 v, float s2)
{
    return (cp_vec3) {
        .i = _mm_fmadd_ps(_mm_set1_ps(s1), v.i, _mm_set1_ps(s2))
    };
}

static FORCE_INLINE cp_vec2 cp_vec2_fmas13(float s1, cp_vec2 v, float s2)
{
    return (cp_vec2) {
        .x = s1 * v.x + s2,
        .y = s1 * v.y + s2
    };
}


static FORCE_INLINE cp_vec4 cp_vec4_fmas23(cp_vec4 v, float s1, float s2)
{
    return (cp_vec4) {
        .i = _mm_fmadd_ps(_mm_set1_ps(s1), v.i, _mm_set1_ps(s2))
    };
}

static FORCE_INLINE cp_vec3 cp_vec3_fmas23(cp_vec3 v, float s1, float s2)
{
    return (cp_vec3) {
        .i = _mm_fmadd_ps(_mm_set1_ps(s1), v.i, _mm_set1_ps(s2))
    };
}

static FORCE_INLINE cp_vec2 cp_vec2_fmas23(cp_vec2 v, float s1, float s2)
{
    return (cp_vec2) {
        .x = s1 * v.x + s2,
        .y = s1 * v.y + s2
    };
}

// FMS
static FORCE_INLINE cp_vec4 cp_vec4_fms(cp_vec4 v1, cp_vec4 v2, cp_vec4 v3)
{
    return (cp_vec4) {
        .i = _mm_fmsub_ps(v1.i, v2.i, v3.i)
    };
}

static FORCE_INLINE cp_vec3 cp_vec3_fms(cp_vec3 v1, cp_vec3 v2, cp_vec3 v3)
{
    return (cp_vec3) {
        .i = _mm_fmsub_ps(v1.i, v2.i, v3.i)
    };
}

static FORCE_INLINE cp_vec2 cp_vec2_fms(cp_vec2 v1, cp_vec2 v2, cp_vec2 v3)
{
    return (cp_vec2) {
        .x = v1.x * v2.x + v3.x,
        .y = v1.y * v2.y + v3.y
    };
}


static FORCE_INLINE cp_vec4 cp_vec4_fmss1(float s, cp_vec4 v1, cp_vec4 v2)
{
    return (cp_vec4) {
        .i = _mm_fmsub_ps(v1.i, _mm_set1_ps(s), v2.i)
    };
}

static FORCE_INLINE cp_vec3 cp_vec3_fmss1(float s, cp_vec3 v1, cp_vec3 v2)
{
    return (cp_vec3) {
        .i = _mm_fmsub_ps(v1.i, _mm_set1_ps(s), v2.i)
    };
}

static FORCE_INLINE cp_vec2 cp_vec2_fmss1(float s, cp_vec2 v1, cp_vec2 v2)
{
    return (cp_vec2) {
        .x = v1.x * s - v2.x,
        .y = v1.y * s - v2.y
    };
}


static FORCE_INLINE cp_vec4 cp_vec4_fmss2(cp_vec4 v1, float s, cp_vec4 v2)
{
    return (cp_vec4) {
        .i = _mm_fmsub_ps(v1.i, _mm_set1_ps(s), v2.i)
    };
}

static FORCE_INLINE cp_vec3 cp_vec3_fmss2(cp_vec3 v1, float s, cp_vec3 v2)
{
    return (cp_vec3) {
        .i = _mm_fmsub_ps(v1.i, _mm_set1_ps(s), v2.i)
    };
}

static FORCE_INLINE cp_vec2 cp_vec2_fmss2(cp_vec2 v1, float s, cp_vec2 v2)
{
    return (cp_vec2) {
        .x = v1.x * s - v2.x,
        .y = v1.y * s - v2.y
    };
}


static FORCE_INLINE cp_vec4 cp_vec4_fmss3(cp_vec4 v1, cp_vec4 v2, float s)
{
    return (cp_vec4) {
        .i = _mm_fmsub_ps(v1.i, v2.i, _mm_set1_ps(s))
    };
}

static FORCE_INLINE cp_vec3 cp_vec3_fmss3(cp_vec3 v1, cp_vec3 v2, float s)
{
    return (cp_vec3) {
        .i = _mm_fmsub_ps(v1.i, v2.i, _mm_set1_ps(s))
    };
}

static FORCE_INLINE cp_vec2 cp_vec2_fmss3(cp_vec2 v1, cp_vec2 v2, float s)
{
    return (cp_vec2) {
        .x = v1.x * v2.x - s,
        .y = v1.y * v2.y - s
    };
}


static FORCE_INLINE cp_vec4 cp_vec4_fmss12(float s1, float s2, cp_vec4 v)
{
    return (cp_vec4) {
        .i = _mm_fmsub_ps(_mm_set1_ps(s1), _mm_set1_ps(s2), v.i)
    };
}

static FORCE_INLINE cp_vec3 cp_vec3_fmss12(float s1, float s2, cp_vec3 v)
{
    return (cp_vec3) {
        .i = _mm_fmsub_ps(_mm_set1_ps(s1), _mm_set1_ps(s2), v.i)
    };
}

static FORCE_INLINE cp_vec2 cp_vec2_fmss12(float s1, float s2, cp_vec2 v)
{
    return (cp_vec2) {
        .x = s1 * s2 - v.x,
        .y = s1 * s2 - v.y
    };
}


static FORCE_INLINE cp_vec4 cp_vec4_fmss13(float s1, cp_vec4 v, float s2)
{
    return (cp_vec4) {
        .i = _mm_fmsub_ps(_mm_set1_ps(s1), v.i, _mm_set1_ps(s2))
    };
}

static FORCE_INLINE cp_vec3 cp_vec3_fmss13(float s1, cp_vec3 v, float s2)
{
    return (cp_vec3) {
        .i = _mm_fmsub_ps(_mm_set1_ps(s1), v.i, _mm_set1_ps(s2))
    };
}

static FORCE_INLINE cp_vec2 cp_vec2_fmss13(float s1, cp_vec2 v, float s2)
{
    return (cp_vec2) {
        .x = s1 * v.x - s2,
        .y = s1 * v.y - s2
    };
}


static FORCE_INLINE cp_vec4 cp_vec4_fmss23(cp_vec4 v, float s1, float s2)
{
    return (cp_vec4) {
        .i = _mm_fmsub_ps(_mm_set1_ps(s1), v.i, _mm_set1_ps(s2))
    };
}

static FORCE_INLINE cp_vec3 cp_vec3_fmss23(cp_vec3 v, float s1, float s2)
{
    return (cp_vec3) {
        .i = _mm_fmsub_ps(_mm_set1_ps(s1), v.i, _mm_set1_ps(s2))
    };
}

static FORCE_INLINE cp_vec2 cp_vec2_fmss23(cp_vec2 v, float s1, float s2)
{
    return (cp_vec2) {
        .x = s1 * v.x - s2,
        .y = s1 * v.y - s2
    };
}

// subtract
static FORCE_INLINE cp_vec4 cp_vec4_sub(cp_vec4 v1, cp_vec4 v2)
{
    return (cp_vec4) {
        .i = _mm_sub_ps(v1.i, v2.i)
    };
}

static FORCE_INLINE cp_vec3 cp_vec3_sub(cp_vec3 v1, cp_vec3 v2)
{
    return (cp_vec3) {
        .i = _mm_sub_ps(v1.i, v2.i)
    };
}

static FORCE_INLINE cp_vec2 cp_vec2_sub(cp_vec2 v1, cp_vec2 v2)
{
    return (cp_vec2) {
        .x = v1.x - v2.x,
        .y = v1.y - v2.y
    };
}


static FORCE_INLINE cp_ivec4 cp_ivec4_sub(cp_ivec4 v1, cp_ivec4 v2)
{
    return (cp_ivec4) {
        .i = _mm_sub_epi32(v1.i, v2.i)
    };
}

static FORCE_INLINE cp_ivec3 cp_ivec3_sub(cp_ivec3 v1, cp_ivec3 v2)
{
    return (cp_ivec3) {
        .i = _mm_sub_epi32(v1.i, v2.i)
    };
}

static FORCE_INLINE cp_ivec2 cp_ivec2_sub(cp_ivec2 v1, cp_ivec2 v2)
{
    return (cp_ivec2) {
        .x = v1.x - v2.x,
        .y = v1.y - v2.y
    };
}


static FORCE_INLINE cp_uvec4 cp_uvec4_sub(cp_uvec4 v1, cp_uvec4 v2)
{
    return (cp_uvec4) {
        .i = _mm_sub_epi32(v1.i, v2.i)
    };
}

static FORCE_INLINE cp_uvec3 cp_uvec3_sub(cp_uvec3 v1, cp_uvec3 v2)
{
    return (cp_uvec3) {
        .i = _mm_sub_epi32(v1.i, v2.i)
    };
}

static FORCE_INLINE cp_uvec2 cp_uvec2_sub(cp_uvec2 v1, cp_uvec2 v2)
{
    return (cp_uvec2) {
        .x = v1.x - v2.x,
        .y = v1.y - v2.y
    };
}


static FORCE_INLINE cp_vec4 cp_vec4_subs1(cp_vec4 v1, float s)
{
    return (cp_vec4) {
        .i = _mm_sub_ps(v1.i, _mm_set1_ps(s))
    };
}

static FORCE_INLINE cp_vec3 cp_vec3_subs1(cp_vec3 v1, float s)
{
    return (cp_vec3) {
        .i = _mm_sub_ps(v1.i, _mm_set1_ps(s))
    };
}

static FORCE_INLINE cp_vec2 cp_vec2_subs1(cp_vec2 v1, float s)
{
    return (cp_vec2) {
        .x = v1.x - s,
        .y = v1.y - s
    };
}


static FORCE_INLINE cp_ivec4 cp_ivec4_subs1(cp_ivec4 v1, int32_t s)
{
    return (cp_ivec4) {
        .i = _mm_sub_epi32(v1.i, _mm_set1_epi32(s))
    };
}

static FORCE_INLINE cp_ivec3 cp_ivec3_subs1(cp_ivec3 v1, int32_t s)
{
    return (cp_ivec3) {
        .i = _mm_sub_epi32(v1.i, _mm_set1_epi32(s))
    };
}

static FORCE_INLINE cp_ivec2 cp_ivec2_subs1(cp_ivec2 v1, int32_t s)
{
    return (cp_ivec2) {
        .x = v1.x - s,
        .y = v1.y - s
    };
}


static FORCE_INLINE cp_uvec4 cp_uvec4_subs1(cp_uvec4 v1, uint32_t s)
{
    return (cp_uvec4) {
        .i = _mm_sub_epi32(v1.i, _mm_set1_epi32(s))
    };
}

static FORCE_INLINE cp_uvec3 cp_uvec3_subs1(cp_uvec3 v1, uint32_t s)
{
    return (cp_uvec3) {
        .i = _mm_sub_epi32(v1.i, _mm_set1_epi32(s))
    };
}

static FORCE_INLINE cp_uvec2 cp_uvec2_subs1(cp_uvec2 v1, uint32_t s)
{
    return (cp_uvec2) {
        .x = v1.x - s,
        .y = v1.y - s
    };
}


static FORCE_INLINE cp_vec4 cp_vec4_subs2(float s, cp_vec4 v1)
{
    return (cp_vec4) {
        .i = _mm_sub_ps(v1.i, _mm_set1_ps(s))
    };
}

static FORCE_INLINE cp_vec3 cp_vec3_subs2(float s, cp_vec3 v1)
{
    return (cp_vec3) {
        .i = _mm_sub_ps(v1.i, _mm_set1_ps(s))
    };
}

static FORCE_INLINE cp_vec2 cp_vec2_subs2(float s, cp_vec2 v1)
{
    return (cp_vec2) {
        .x = v1.x + s,
        .y = v1.y + s
    };
}


static FORCE_INLINE cp_ivec4 cp_ivec4_subs2(int32_t s, cp_ivec4 v1)
{
    return (cp_ivec4) {
        .i = _mm_sub_epi32(v1.i, _mm_set1_epi32(s))
    };
}

static FORCE_INLINE cp_ivec3 cp_ivec3_subs2(int32_t s, cp_ivec3 v1)
{
    return (cp_ivec3) {
        .i = _mm_sub_epi32(v1.i, _mm_set1_epi32(s))
    };
}

static FORCE_INLINE cp_ivec2 cp_ivec2_subs2(int32_t s, cp_ivec2 v1)
{
    return (cp_ivec2) {
        .x = v1.x + s,
        .y = v1.y + s
    };
}


static FORCE_INLINE cp_uvec4 cp_uvec4_subs2(uint32_t s, cp_uvec4 v1)
{
    return (cp_uvec4) {
        .i = _mm_sub_epi32(v1.i, _mm_set1_epi32(s))
    };
}

static FORCE_INLINE cp_uvec3 cp_uvec3_subs2(uint32_t s, cp_uvec3 v1)
{
    return (cp_uvec3) {
        .i = _mm_sub_epi32(v1.i, _mm_set1_epi32(s))
    };
}

static FORCE_INLINE cp_uvec2 cp_uvec2_subs2(uint32_t s, cp_uvec2 v1)
{
    return (cp_uvec2) {
        .x = v1.x + s,
        .y = v1.y + s
    };
}

// multiply
static FORCE_INLINE cp_vec4 cp_vec4_mul(cp_vec4 v1, cp_vec4 v2)
{
    return (cp_vec4) {
        .i = _mm_mul_ps(v1.i, v2.i)
    };
}

static FORCE_INLINE cp_vec3 cp_vec3_mul(cp_vec3 v1, cp_vec3 v2)
{
    return (cp_vec3) {
        .i = _mm_mul_ps(v1.i, v2.i)
    };
}

static FORCE_INLINE cp_vec2 cp_vec2_mul(cp_vec2 v1, cp_vec2 v2)
{
    return (cp_vec2) {
        .x = v1.x * v2.x, 
        .y = v1.y * v2.y
    };
}


static FORCE_INLINE cp_ivec4 cp_ivec4_mul(cp_ivec4 v1, cp_ivec4 v2)
{
    return (cp_ivec4) {
        .i = _mm_mul_epi32(v1.i, v2.i)
    };
}

static FORCE_INLINE cp_ivec3 cp_ivec3_mul(cp_ivec3 v1, cp_ivec3 v2)
{
    return (cp_ivec3) {
        .i = _mm_mul_epi32(v1.i, v2.i)
    };
}

static FORCE_INLINE cp_ivec2 cp_ivec2_mul(cp_ivec2 v1, cp_ivec2 v2)
{
    return (cp_ivec2) {
        .x = v1.x * v2.x, 
        .y = v1.y * v2.y
    };
}


static FORCE_INLINE cp_uvec4 cp_uvec4_mul(cp_uvec4 v1, cp_uvec4 v2)
{
    return (cp_uvec4) {
        .i = _mm_mul_epi32(v1.i, v2.i)
    };
}

static FORCE_INLINE cp_uvec3 cp_uvec3_mul(cp_uvec3 v1, cp_uvec3 v2)
{
    return (cp_uvec3) {
        .i = _mm_mul_epi32(v1.i, v2.i)
    };
}

static FORCE_INLINE cp_uvec2 cp_uvec2_mul(cp_uvec2 v1, cp_uvec2 v2)
{
    return (cp_uvec2) {
        .x = v1.x * v2.x, 
        .y = v1.y * v2.y
    };
}


static FORCE_INLINE cp_vec4 cp_vec4_muls1(cp_vec4 v1, float s)
{
    return (cp_vec4) {
        .i = _mm_mul_ps(v1.i, _mm_set1_ps(s))
    };
}

static FORCE_INLINE cp_vec3 cp_vec3_muls1(cp_vec3 v1, float s)
{
    return (cp_vec3) {
        .i = _mm_mul_ps(v1.i, _mm_set1_ps(s))
    };
}

static FORCE_INLINE cp_vec2 cp_vec2_muls1(cp_vec2 v1, float s)
{
    return (cp_vec2) {
        .x = v1.x * s,
        .y = v1.y * s
    };
}


static FORCE_INLINE cp_ivec4 cp_ivec4_muls1(cp_ivec4 v1, int32_t s)
{
    return (cp_ivec4) {
        .i = _mm_mul_epi32(v1.i, _mm_set1_epi32(s))
    };
}

static FORCE_INLINE cp_ivec3 cp_ivec3_muls1(cp_ivec3 v1, int32_t s)
{
    return (cp_ivec3) {
        .i = _mm_mul_epi32(v1.i, _mm_set1_epi32(s))
    };
}

static FORCE_INLINE cp_ivec2 cp_ivec2_muls1(cp_ivec2 v1, int32_t s)
{
    return (cp_ivec2) {
        .x = v1.x * s,
        .y = v1.y * s
    };
}


static FORCE_INLINE cp_uvec4 cp_uvec4_muls1(cp_uvec4 v1, uint32_t s)
{
    return (cp_uvec4) {
        .i = _mm_mul_epi32(v1.i, _mm_set1_epi32(s))
    };
}

static FORCE_INLINE cp_uvec3 cp_uvec3_muls1(cp_uvec3 v1, uint32_t s)
{
    return (cp_uvec3) {
        .i = _mm_mul_epi32(v1.i, _mm_set1_epi32(s))
    };
}

static FORCE_INLINE cp_uvec2 cp_uvec2_muls1(cp_uvec2 v1, uint32_t s)
{
    return (cp_uvec2) {
        .x = v1.x * s,
        .y = v1.y * s
    };
}


static FORCE_INLINE cp_vec4 cp_vec4_muls2(float s, cp_vec4 v1)
{
    return (cp_vec4) {
        .i = _mm_mul_ps(v1.i, _mm_set1_ps(s))
    };
}

static FORCE_INLINE cp_vec3 cp_vec3_muls2(float s, cp_vec3 v1)
{
    return (cp_vec3) {
        .i = _mm_mul_ps(v1.i, _mm_set1_ps(s))
    };
}

static FORCE_INLINE cp_vec2 cp_vec2_muls2(float s, cp_vec2 v1)
{
    return (cp_vec2) {
        .x = v1.x + s,
        .y = v1.y + s
    };
}


static FORCE_INLINE cp_ivec4 cp_ivec4_muls2(int32_t s, cp_ivec4 v1)
{
    return (cp_ivec4) {
        .i = _mm_mul_epi32(v1.i, _mm_set1_epi32(s))
    };
}

static FORCE_INLINE cp_ivec3 cp_ivec3_muls2(int32_t s, cp_ivec3 v1)
{
    return (cp_ivec3) {
        .i = _mm_mul_epi32(v1.i, _mm_set1_epi32(s))
    };
}

static FORCE_INLINE cp_ivec2 cp_ivec2_muls2(int32_t s, cp_ivec2 v1)
{
    return (cp_ivec2) {
        .x = v1.x + s,
        .y = v1.y + s
    };
}


static FORCE_INLINE cp_uvec4 cp_uvec4_muls2(uint32_t s, cp_uvec4 v1)
{
    return (cp_uvec4) {
        .i = _mm_mul_epi32(v1.i, _mm_set1_epi32(s))
    };
}

static FORCE_INLINE cp_uvec3 cp_uvec3_muls2(uint32_t s, cp_uvec3 v1)
{
    return (cp_uvec3) {
        .i = _mm_mul_epi32(v1.i, _mm_set1_epi32(s))
    };
}

static FORCE_INLINE cp_uvec2 cp_uvec2_muls2(uint32_t s, cp_uvec2 v1)
{
    return (cp_uvec2) {
        .x = v1.x + s,
        .y = v1.y + s
    };
}

// divide
static FORCE_INLINE cp_vec4 cp_vec4_div(cp_vec4 v1, cp_vec4 v2)
{
    return (cp_vec4) {
        .i = _mm_div_ps(v1.i, v2.i)
    };
}

static FORCE_INLINE cp_vec3 cp_vec3_div(cp_vec3 v1, cp_vec3 v2)
{
    return (cp_vec3) {
        .i = _mm_div_ps(v1.i, v2.i)
    };
}

static FORCE_INLINE cp_vec2 cp_vec2_div(cp_vec2 v1, cp_vec2 v2)
{
    return (cp_vec2) {
        .x = v1.x / v2.x,
        .y = v1.y / v2.y
    };
}


static FORCE_INLINE cp_ivec4 cp_ivec4_div(cp_ivec4 v1, cp_ivec4 v2)
{
    return (cp_ivec4) {
        .x = v1.x / v2.x,
        .y = v1.y / v2.y,
        .z = v1.z / v2.z,
        .w = v1.w / v2.w
    };
}

static FORCE_INLINE cp_ivec3 cp_ivec3_div(cp_ivec3 v1, cp_ivec3 v2)
{
    return (cp_ivec3) {
        .x = v1.x / v2.x,
        .y = v1.y / v2.y,
        .z = v1.z / v2.z
    };
}

static FORCE_INLINE cp_ivec2 cp_ivec2_div(cp_ivec2 v1, cp_ivec2 v2)
{
    return (cp_ivec2) {
        .x = v1.x / v2.x,
        .y = v1.y / v2.y
    };
}


static FORCE_INLINE cp_uvec4 cp_uvec4_div(cp_uvec4 v1, cp_uvec4 v2)
{
    return (cp_uvec4) {
        .x = v1.x / v2.x,
        .y = v1.y / v2.y,
        .z = v1.z / v2.z,
        .w = v1.w / v2.w
    };
}

static FORCE_INLINE cp_uvec3 cp_uvec3_div(cp_uvec3 v1, cp_uvec3 v2)
{
    return (cp_uvec3) {
        .x = v1.x / v2.x,
        .y = v1.y / v2.y,
        .z = v1.z / v2.z
    };
}

static FORCE_INLINE cp_uvec2 cp_uvec2_div(cp_uvec2 v1, cp_uvec2 v2)
{
    return (cp_uvec2) {
        .x = v1.x / v2.x,
        .y = v1.y / v2.y
    };
}


static FORCE_INLINE cp_vec4 cp_vec4_divs1(cp_vec4 v1, float s)
{
    return (cp_vec4) {
        .i = _mm_div_ps(v1.i, _mm_set1_ps(s))
    };
}

static FORCE_INLINE cp_vec3 cp_vec3_divs1(cp_vec3 v1, float s)
{
    return (cp_vec3) {
        .i = _mm_div_ps(v1.i, _mm_set1_ps(s))
    };
}

static FORCE_INLINE cp_vec2 cp_vec2_divs1(cp_vec2 v1, float s)
{
    return (cp_vec2) {
        .x = v1.x / s,
        .y = v1.y / s
    };
}


static FORCE_INLINE cp_ivec4 cp_ivec4_divs1(cp_ivec4 v1, int32_t s)
{
    return (cp_ivec4) {
        .x = v1.x / s,
        .y = v1.y / s,
        .z = v1.z / s,
        .w = v1.w / s
    };
}

static FORCE_INLINE cp_ivec3 cp_ivec3_divs1(cp_ivec3 v1, int32_t s)
{
    return (cp_ivec3) {
        .x = v1.x / s,
        .y = v1.y / s,
        .z = v1.z / s
    };
}

static FORCE_INLINE cp_ivec2 cp_ivec2_divs1(cp_ivec2 v1, int32_t s)
{
    return (cp_ivec2) {
        .x = v1.x / s,
        .y = v1.y / s
    };
}


static FORCE_INLINE cp_uvec4 cp_uvec4_divs1(cp_uvec4 v1, uint32_t s)
{
    return (cp_uvec4) {
        .x = v1.x / s,
        .y = v1.y / s,
        .z = v1.z / s,
        .w = v1.w / s
    };
}

static FORCE_INLINE cp_uvec3 cp_uvec3_divs1(cp_uvec3 v1, uint32_t s)
{
    return (cp_uvec3) {
        .x = v1.x / s,
        .y = v1.y / s,
        .z = v1.z / s
    };
}

static FORCE_INLINE cp_uvec2 cp_uvec2_divs1(cp_uvec2 v1, uint32_t s)
{
    return (cp_uvec2) {
        .x = v1.x / s,
        .y = v1.y / s
    };
}


static FORCE_INLINE cp_vec4 cp_vec4_divs2(float s, cp_vec4 v1)
{
    return (cp_vec4) {
        .i = _mm_div_ps(v1.i, _mm_set1_ps(s))
    };
}

static FORCE_INLINE cp_vec3 cp_vec3_divs2(float s, cp_vec3 v1)
{
    return (cp_vec3) {
        .i = _mm_div_ps(v1.i, _mm_set1_ps(s))
    };
}

static FORCE_INLINE cp_vec2 cp_vec2_divs2(float s, cp_vec2 v1)
{
    return (cp_vec2) {
        .x = v1.x / s,
        .y = v1.y / s
    };
}


static FORCE_INLINE cp_ivec4 cp_ivec4_divs2(int32_t s, cp_ivec4 v1)
{
    return (cp_ivec4) {
        .x = v1.x / s,
        .y = v1.y / s,
        .z = v1.z / s,
        .w = v1.w / s
    };
}

static FORCE_INLINE cp_ivec3 cp_ivec3_divs2(int32_t s, cp_ivec3 v1)
{
    return (cp_ivec3) {
        .x = v1.x / s,
        .y = v1.y / s,
        .z = v1.z / s
    };
}

static FORCE_INLINE cp_ivec2 cp_ivec2_divs2(int32_t s, cp_ivec2 v1)
{
    return (cp_ivec2) {
        .x = v1.x / s,
        .y = v1.y / s
    };
}


static FORCE_INLINE cp_uvec4 cp_uvec4_divs2(uint32_t s, cp_uvec4 v1)
{
    return (cp_uvec4) {
        .x = v1.x / s,
        .y = v1.y / s,
        .z = v1.z / s,
        .w = v1.w / s
    };
}

static FORCE_INLINE cp_uvec3 cp_uvec3_divs2(uint32_t s, cp_uvec3 v1)
{
    return (cp_uvec3) {
        .x = v1.x / s,
        .y = v1.y / s,
        .z = v1.z / s
    };
}

static FORCE_INLINE cp_uvec2 cp_uvec2_divs2(uint32_t s, cp_uvec2 v1)
{
    return (cp_uvec2) {
        .x = v1.x / s,
        .y = v1.y / s
    };
}


// dot
static FORCE_INLINE float cp_vec4_dot(cp_vec4 v1, cp_vec4 v2)
{
    return _mm_cvtss_f32(_mm_dp_ps(v1.i, v2.i, UINT8_MAX));
}

static FORCE_INLINE float cp_vec3_dot(cp_vec3 v1, cp_vec3 v2)
{
    return _mm_cvtss_f32(_mm_dp_ps(v1.i, v2.i, UINT8_MAX));
}

static FORCE_INLINE float cp_vec2_dot(cp_vec2 v1, cp_vec2 v2)
{
    return v1.x * v2.x + v1.y * v2.y;
}


static FORCE_INLINE int32_t cp_ivec4_dot(cp_ivec4 v1, cp_ivec4 v2)
{
    return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z + v1.w * v2.w;
}

static FORCE_INLINE int32_t cp_ivec3_dot(cp_ivec3 v1, cp_ivec3 v2)
{
    return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
}

static FORCE_INLINE int32_t cp_ivec2_dot(cp_ivec2 v1, cp_ivec2 v2)
{
    return v1.x * v2.x + v1.y * v2.y;
}


static FORCE_INLINE uint32_t cp_uvec4_dot(cp_uvec4 v1, cp_uvec4 v2)
{
    return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z + v1.w * v2.w;
}

static FORCE_INLINE uint32_t cp_uvec3_dot(cp_uvec3 v1, cp_uvec3 v2)
{
    return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
}

static FORCE_INLINE uint32_t cp_uvec2_dot(cp_uvec2 v1, cp_uvec2 v2)
{
    return v1.x * v2.x + v1.y * v2.y;
}


// normalize
static FORCE_INLINE cp_vec4 cp_vec4_normalize(cp_vec4 v)
{
    __m128 d = _mm_dp_ps(v.i, v.i, UINT8_MAX);
    __m128 rsqrt = _mm_rsqrt_ps(d);
    return (cp_vec4) {
        .i = _mm_mul_ps(v.i, rsqrt)
    };
}

static FORCE_INLINE cp_vec3 cp_vec3_normalize(cp_vec3 v)
{
    __m128 d = _mm_dp_ps(v.i, v.i, UINT8_MAX);
    __m128 rsqrt = _mm_rsqrt_ps(d);
    return (cp_vec3) {
        .i = _mm_mul_ps(v.i, rsqrt)
    };
}

static FORCE_INLINE cp_vec2 cp_vec2_normalize(cp_vec2 v)
{
    __m128 vi = _mm_set_ps(v.x, v.y, 0, 0);
    __m128 d = _mm_dp_ps(vi, vi, UINT8_MAX);
    __m128 rsqrt = _mm_rsqrt_ps(d);
    float rsqrtf = _mm_cvtss_f32(rsqrt);
    return (cp_vec2) {
        .x = v.x * rsqrtf,
        .y = v.y * rsqrtf
    };
}


// length
static FORCE_INLINE float cp_vec4_length(cp_vec4 v)
{
    __m128 d = _mm_dp_ps(v.i, v.i, UINT8_MAX);
    __m128 l = _mm_sqrt_ps(d);
    return _mm_cvtss_f32(l);
}

static FORCE_INLINE float cp_vec3_length(cp_vec3 v)
{
    __m128 d = _mm_dp_ps(v.i, v.i, UINT8_MAX);
    __m128 l = _mm_sqrt_ps(d);
    return _mm_cvtss_f32(l);
}

static FORCE_INLINE float cp_vec2_length(cp_vec2 v)
{
    __m128 vi = _mm_set_ps(v.x, v.y, 0, 0);
    __m128 d = _mm_dp_ps(vi, vi, UINT8_MAX);
    __m128 l = _mm_sqrt_ps(d);
    return _mm_cvtss_f32(l);

}


// cross
static FORCE_INLINE cp_vec3 cp_vec3_cross(cp_vec3 v1, cp_vec3 v2)
{
    __m128 s2 = _mm_mul_ps(_mm_shuffle_ps(v1.i, v1.i, _MM_SHUFFLE(3, 1, 0, 2)), _mm_shuffle_ps(v2.i, v2.i, _MM_SHUFFLE(3, 0, 2, 1)));
    return (cp_vec3) {
        .i = _mm_fmsub_ps(_mm_shuffle_ps(v1.i, v1.i, _MM_SHUFFLE(3, 0, 2, 1)), _mm_shuffle_ps(v2.i, v2.i, _MM_SHUFFLE(3, 1, 0, 2)), s2)
    };
}

/**
 * Type Aliases and Generic Functions
 */

 // Feel free to adjust according to your preferences / in case of naming conflicts.
typedef cp_vec4 vec4;
typedef cp_vec3 vec3;
typedef cp_vec2 vec2;

typedef cp_ivec4 ivec4;
typedef cp_ivec3 ivec3;
typedef cp_ivec2 ivec2;

typedef cp_uvec4 uvec4;
typedef cp_uvec3 uvec3;
typedef cp_uvec2 uvec2;

/**
 * Misc stuff
 */
FORCE_INLINE static float cp_radiansf(float d)
{
  return d / 180.0f * CP_M_PI;
}

FORCE_INLINE static double cp_radiansd(double d)
{
  return d / 180.0f * CP_M_PI;
}

FORCE_INLINE static float cp_degreef(float r)
{
  return r * 180.0f / CP_M_PI;
}

FORCE_INLINE static double cp_degreed(double r)
{
  return r * 180.0f / CP_M_PI;
}

FORCE_INLINE static float cp_minf(float f1, float f2)
{
  return f1 > f2 ? f2 : f1;
}

FORCE_INLINE static float cp_maxf(float f1, float f2)
{
  return f1 > f2 ? f1 : f2;
}

FORCE_INLINE static int32_t cp_mini(int32_t f1, int32_t f2)
{
  return f1 > f2 ? f2 : f1;
}

FORCE_INLINE static int32_t cp_maxi(int32_t f1, int32_t f2)
{
  return f1 > f2 ? f1 : f2;
}

FORCE_INLINE static uint32_t cp_minu(uint32_t f1, uint32_t f2)
{
  return f1 > f2 ? f2 : f1;
}

FORCE_INLINE static uint32_t cp_maxu(uint32_t f1, uint32_t f2)
{
  return f1 > f2 ? f1 : f2;
}

#define min(x, y) _Generic((x),\
    float:      cp_minf,              \
    int32_t:    cp_mini,          \
    uint32_t:   cp_minu) (x,y)

#define max(x, y) _Generic((x),\
    float:      cp_maxf,              \
    int32_t:    cp_maxi,          \
    uint32_t:   cp_maxu) (x,y)

#define radians(x) _Generic((x),\
    float:      cp_radiansf,              \
    double:    cp_radiansd) (x)

#define degree(x) _Generic((x),\
    float:      cp_degreef,              \
    double:    cp_degreed) (x)

typedef struct PackedVec3
{
    float x, y, z;
} PackedVec3;

FORCE_INLINE static PackedVec3 packVec3(vec3 v)
{
    return (PackedVec3) {v.x, v.y, v.z};
}

typedef union ALIGN(16) cp_Mat4
{
    struct
    {
        float m00;
        float m10;
        float m20;
        float m30;
        float m01;
        float m11;
        float m21;
        float m31;
        float m02;
        float m12;
        float m22;
        float m32;
        float m03;
        float m13;
        float m23;
        float m33;
    };
    float arr[16];
    __m128 i[4];
    vec4 v[4];
} cp_Mat4;

typedef cp_Mat4 mat4;


FORCE_INLINE static vec4 cp_mat4_vec_mul(mat4 mat, vec4 vec) {
    return (vec4) {
        mat.m00 * vec.x + mat.m01 * vec.x + mat.m02 * vec.x + mat.m03 * vec.x,
        mat.m10 * vec.x + mat.m11 * vec.x + mat.m12 * vec.x + mat.m13 * vec.x,
        mat.m20 * vec.x + mat.m21 * vec.x + mat.m22 * vec.x + mat.m23 * vec.x,
        mat.m30 * vec.x + mat.m31 * vec.x + mat.m32 * vec.x + mat.m33 * vec.x
    };
}

typedef struct INVALID_GENERIC_ARGUMENT {int i;} INVALID_GENERIC_ARGUMENT;
INVALID_GENERIC_ARGUMENT *invalid_type_for_generic_function(void);

#define add(v1, v2) _Generic((v1),      \
    cp_vec4:    _Generic((v2),          \
        cp_vec4:    cp_vec4_add,        \
        float:      cp_vec4_adds1,      \
        default:    invalid_type_for_generic_function),     \
    cp_vec3:    _Generic((v2),          \
        cp_vec3:    cp_vec3_add,        \
        float:      cp_vec3_adds1,      \
        default:    invalid_type_for_generic_function),     \
    cp_vec2:    _Generic((v2),          \
        cp_vec2:    cp_vec2_add,        \
        float:      cp_vec2_adds1,      \
        default:    invalid_type_for_generic_function),     \
    cp_ivec4:    _Generic((v2),         \
         cp_ivec4:    cp_ivec4_add,     \
         int32_t:      cp_ivec4_adds1,      \
         default:    invalid_type_for_generic_function), \
    cp_ivec3:    _Generic((v2),         \
         cp_ivec3:    cp_ivec3_add,     \
         int32_t:      cp_ivec3_adds1,      \
         default:    invalid_type_for_generic_function), \
    cp_ivec2:    _Generic((v2),         \
         cp_ivec2:    cp_ivec2_add,     \
         int32_t:      cp_ivec2_adds1,      \
         default:    invalid_type_for_generic_function), \
    cp_uvec4:    _Generic((v2),         \
          cp_uvec4:    cp_uvec4_add,    \
          uint32_t:      cp_uvec4_adds1,      \
          default:    invalid_type_for_generic_function),\
    cp_uvec3:    _Generic((v2),         \
          cp_uvec3:    cp_uvec3_add,    \
          uint32_t:      cp_uvec3_adds1,      \
          default:    invalid_type_for_generic_function),\
    cp_uvec2:    _Generic((v2),         \
           cp_uvec2:    cp_uvec2_add,    \
           uint32_t:      cp_uvec2_adds1,      \
           default:    invalid_type_for_generic_function),\
    float:      _Generic((v2),          \
        cp_vec4:    cp_vec4_adds2,      \
        cp_vec3:    cp_vec3_adds2,      \
        cp_vec2:    cp_vec2_adds2,      \
        default:    invalid_type_for_generic_function),     \
    int32_t:      _Generic((v2),        \
        cp_ivec4:    cp_ivec4_adds2,    \
        cp_ivec3:    cp_ivec3_adds2,    \
        cp_ivec2:    cp_ivec2_adds2,      \
        default:    invalid_type_for_generic_function),   \
    uint32_t:      _Generic((v2),       \
        cp_uvec4:    cp_uvec4_adds2,    \
        cp_uvec3:    cp_uvec3_adds2,    \
        cp_uvec2:    cp_uvec2_adds2,      \
        default:    invalid_type_for_generic_function)    \
    ) (v1, v2)

#define FMA(v1, v2, v3) _Generic((v1),      \
    cp_vec4:    _Generic((v2),          \
        cp_vec4:    _Generic((v3),           \
            cp_vec4:    cp_vec4_fma,           \
            float:      cp_vec4_fmas3,      \
            default:    invalid_type_for_generic_function),        \
        float:      _Generic((v3),           \
            cp_vec4:    cp_vec4_fmas2,           \
            float:      cp_vec4_fmas23,      \
            default:    invalid_type_for_generic_function),      \
        default:    invalid_type_for_generic_function),     \
    cp_vec3:    _Generic((v2),          \
        cp_vec3:    _Generic((v3),           \
            cp_vec3:    cp_vec3_fma,           \
            float:      cp_vec3_fmas3,      \
            default:    invalid_type_for_generic_function),        \
        float:      _Generic((v3),           \
            cp_vec3:    cp_vec3_fmas2,           \
            float:      cp_vec3_fmas23,      \
            default:    invalid_type_for_generic_function),      \
        default:    invalid_type_for_generic_function),     \
    cp_vec2:    _Generic((v2),          \
        cp_vec2:    _Generic((v3),           \
            cp_vec2:    cp_vec2_fma,           \
            float:      cp_vec2_fmas3,      \
            default:    invalid_type_for_generic_function),        \
        float:      _Generic((v3),           \
            cp_vec2:    cp_vec2_fmas2,           \
            float:      cp_vec2_fmas23,      \
            default:    invalid_type_for_generic_function),      \
        default:    invalid_type_for_generic_function),     \
    float:      _Generic((v2),          \
        cp_vec4:    _Generic((v3),           \
            cp_vec4:    cp_vec4_fmas1,      \
            float:      cp_vec4_fmas12,     \
            default:    invalid_type_for_generic_function),      \
        cp_vec3:    _Generic((v3),           \
            cp_vec3:    cp_vec3_fmas1,      \
            float:      cp_vec3_fmas12,     \
            default:    invalid_type_for_generic_function),      \
        cp_vec2:    _Generic((v3),           \
            cp_vec2:    cp_vec2_fmas1,      \
            float:      cp_vec2_fmas12,     \
            default:    invalid_type_for_generic_function),      \
        default:    invalid_type_for_generic_function)      \
    ) (v1, v2, v3)

#define FMS(v1, v2, v3) _Generic((v1),      \
    cp_vec4:    _Generic((v2),          \
        cp_vec4:    _Generic((v3),           \
            cp_vec4:    cp_vec4_fms,           \
            float:      cp_vec4_fmss3,      \
            default:    invalid_type_for_generic_function),        \
        float:      _Generic((v3),           \
            cp_vec4:    cp_vec4_fmss2,           \
            float:      cp_vec4_fmss23,      \
            default:    invalid_type_for_generic_function),      \
        default:    invalid_type_for_generic_function),     \
    cp_vec3:    _Generic((v2),          \
        cp_vec3:    _Generic((v3),           \
            cp_vec3:    cp_vec3_fms,           \
            float:      cp_vec3_fmss3,      \
            default:    invalid_type_for_generic_function),        \
        float:      _Generic((v3),           \
            cp_vec3:    cp_vec3_fmss2,           \
            float:      cp_vec3_fmss23,      \
            default:    invalid_type_for_generic_function),      \
        default:    invalid_type_for_generic_function),     \
    cp_vec2:    _Generic((v2),          \
        cp_vec2:    _Generic((v3),           \
            cp_vec2:    cp_vec2_fms,           \
            float:      cp_vec2_fmss3,      \
            default:    invalid_type_for_generic_function),        \
        float:      _Generic((v3),           \
            cp_vec2:    cp_vec2_fmss2,           \
            float:      cp_vec2_fmss23,      \
            default:    invalid_type_for_generic_function),      \
        default:    invalid_type_for_generic_function),     \
    float:      _Generic((v2),          \
        cp_vec4:    _Generic((v3),           \
            cp_vec4:    cp_vec4_fmss1,      \
            float:      cp_vec4_fmss12,     \
            default:    invalid_type_for_generic_function),      \
        cp_vec3:    _Generic((v3),           \
            cp_vec3:    cp_vec3_fmss1,      \
            float:      cp_vec3_fmss12,     \
            default:    invalid_type_for_generic_function),      \
        cp_vec2:    _Generic((v3),           \
            cp_vec2:    cp_vec2_fmss1,      \
            float:      cp_vec2_fmss12,     \
            default:    invalid_type_for_generic_function),      \
        default:    invalid_type_for_generic_function)      \
    ) (v1, v2, v3)

#define sub(v1, v2) _Generic((v1),      \
    cp_vec4:    _Generic((v2),          \
        cp_vec4:    cp_vec4_sub,        \
        float:      cp_vec4_subs1,      \
        default:    invalid_type_for_generic_function),     \
    cp_vec3:    _Generic((v2),          \
        cp_vec3:    cp_vec3_sub,        \
        float:      cp_vec3_subs1,      \
        default:    invalid_type_for_generic_function),     \
    cp_vec2:    _Generic((v2),          \
        cp_vec2:    cp_vec2_sub,        \
        float:      cp_vec2_subs1,      \
        default:    invalid_type_for_generic_function),     \
    cp_ivec4:    _Generic((v2),         \
         cp_ivec4:    cp_ivec4_sub,     \
         int32_t:      cp_ivec4_subs1,      \
         default:    invalid_type_for_generic_function), \
    cp_ivec3:    _Generic((v2),         \
         cp_ivec3:    cp_ivec3_sub,     \
         int32_t:      cp_ivec3_subs1,      \
         default:    invalid_type_for_generic_function), \
    cp_ivec2:    _Generic((v2),         \
         cp_ivec2:    cp_ivec2_sub,     \
         int32_t:      cp_ivec2_subs1,      \
         default:    invalid_type_for_generic_function), \
    cp_uvec4:    _Generic((v2),         \
          cp_uvec4:    cp_uvec4_sub,    \
          uint32_t:      cp_uvec4_subs1,      \
          default:    invalid_type_for_generic_function),\
    cp_uvec3:    _Generic((v2),         \
          cp_uvec3:    cp_uvec3_sub,    \
          uint32_t:      cp_uvec3_subs1,      \
          default:    invalid_type_for_generic_function),\
    cp_uvec2:    _Generic((v2),         \
           cp_uvec2:    cp_uvec2_sub,    \
           uint32_t:      cp_uvec2_subs1,      \
           default:    invalid_type_for_generic_function),\
    float:      _Generic((v2),          \
        cp_vec4:    cp_vec4_subs2,      \
        cp_vec3:    cp_vec3_subs2,      \
        cp_vec2:    cp_vec2_subs2,      \
        default:    invalid_type_for_generic_function),     \
    int32_t:      _Generic((v2),        \
        cp_ivec4:    cp_ivec4_subs2,    \
        cp_ivec3:    cp_ivec3_subs2,    \
        cp_ivec2:    cp_ivec2_subs2,      \
        default:    invalid_type_for_generic_function),   \
    uint32_t:      _Generic((v2),       \
        cp_uvec4:    cp_uvec4_subs2,    \
        cp_uvec3:    cp_uvec3_subs2,    \
        cp_uvec2:    cp_uvec2_subs2,      \
        default:    invalid_type_for_generic_function)    \
    ) (v1, v2)

#define mul(v1, v2) _Generic((v1),      \
    cp_vec4:    _Generic((v2),          \
        cp_vec4:    cp_vec4_mul,        \
        float:      cp_vec4_muls1,      \
        default:    invalid_type_for_generic_function),     \
    cp_vec3:    _Generic((v2),          \
        cp_vec3:    cp_vec3_mul,        \
        float:      cp_vec3_muls1,      \
        default:    invalid_type_for_generic_function),     \
    cp_vec2:    _Generic((v2),          \
        cp_vec2:    cp_vec2_mul,        \
        float:      cp_vec2_muls1,      \
        default:    invalid_type_for_generic_function),     \
    cp_ivec4:    _Generic((v2),         \
         cp_ivec4:    cp_ivec4_mul,     \
         int32_t:      cp_ivec4_muls1,      \
         default:    invalid_type_for_generic_function), \
    cp_ivec3:    _Generic((v2),         \
         cp_ivec3:    cp_ivec3_mul,     \
         int32_t:      cp_ivec3_muls1,      \
         default:    invalid_type_for_generic_function), \
    cp_ivec2:    _Generic((v2),         \
         cp_ivec2:    cp_ivec2_mul,     \
         int32_t:      cp_ivec2_muls1,      \
         default:    invalid_type_for_generic_function), \
    cp_uvec4:    _Generic((v2),         \
          cp_uvec4:    cp_uvec4_mul,    \
          uint32_t:      cp_uvec4_muls1,      \
          default:    invalid_type_for_generic_function),\
    cp_uvec3:    _Generic((v2),         \
          cp_uvec3:    cp_uvec3_mul,    \
          uint32_t:      cp_uvec3_muls1,      \
          default:    invalid_type_for_generic_function),\
    cp_uvec2:    _Generic((v2),         \
           cp_uvec2:    cp_uvec2_mul,    \
           uint32_t:      cp_uvec2_muls1,      \
           default:    invalid_type_for_generic_function),  \
    mat4:       _Generic((v2),          \
           vec4: cp_mat4_vec_mul,       \
           default:    invalid_type_for_generic_function),\
    float:      _Generic((v2),          \
        cp_vec4:    cp_vec4_muls2,      \
        cp_vec3:    cp_vec3_muls2,      \
        cp_vec2:    cp_vec2_muls2,      \
        default:    invalid_type_for_generic_function),     \
    int32_t:      _Generic((v2),        \
        cp_ivec4:    cp_ivec4_muls2,    \
        cp_ivec3:    cp_ivec3_muls2,    \
        cp_ivec2:    cp_ivec2_muls2,      \
        default:    invalid_type_for_generic_function),   \
    uint32_t:      _Generic((v2),       \
        cp_uvec4:    cp_uvec4_muls2,    \
        cp_uvec3:    cp_uvec3_muls2,    \
        cp_uvec2:    cp_uvec2_muls2,      \
        default:    invalid_type_for_generic_function)    \
    ) (v1, v2)

#define div(v1, v2) _Generic((v1),      \
    cp_vec4:    _Generic((v2),          \
        cp_vec4:    cp_vec4_div,        \
        float:      cp_vec4_divs1,      \
        default:    invalid_type_for_generic_function),     \
    cp_vec3:    _Generic((v2),          \
        cp_vec3:    cp_vec3_div,        \
        float:      cp_vec3_divs1,      \
        default:    invalid_type_for_generic_function),     \
    cp_vec2:    _Generic((v2),          \
        cp_vec2:    cp_vec2_div,        \
        float:      cp_vec2_divs1,      \
        default:    invalid_type_for_generic_function),     \
    cp_ivec4:    _Generic((v2),         \
         cp_ivec4:    cp_ivec4_div,     \
         int32_t:      cp_ivec4_divs1,      \
         default:    invalid_type_for_generic_function), \
    cp_ivec3:    _Generic((v2),         \
         cp_ivec3:    cp_ivec3_div,     \
         int32_t:      cp_ivec3_divs1,      \
         default:    invalid_type_for_generic_function), \
    cp_ivec2:    _Generic((v2),         \
         cp_ivec2:    cp_ivec2_div,     \
         int32_t:      cp_ivec2_divs1,      \
         default:    invalid_type_for_generic_function), \
    cp_uvec4:    _Generic((v2),         \
          cp_uvec4:    cp_uvec4_div,    \
          uint32_t:      cp_uvec4_divs1,      \
          default:    invalid_type_for_generic_function),\
    cp_uvec3:    _Generic((v2),         \
          cp_uvec3:    cp_uvec3_div,    \
          uint32_t:      cp_uvec3_divs1,      \
          default:    invalid_type_for_generic_function),\
    cp_uvec2:    _Generic((v2),         \
           cp_uvec2:    cp_uvec2_div,    \
           uint32_t:      cp_uvec2_divs1,      \
           default:    invalid_type_for_generic_function),\
    float:      _Generic((v2),          \
        cp_vec4:    cp_vec4_divs2,      \
        cp_vec3:    cp_vec3_divs2,      \
        cp_vec2:    cp_vec2_divs2,      \
        default:    invalid_type_for_generic_function),     \
    int32_t:      _Generic((v2),        \
        cp_ivec4:    cp_ivec4_divs2,    \
        cp_ivec3:    cp_ivec3_divs2,    \
        cp_ivec2:    cp_ivec2_divs2,      \
        default:    invalid_type_for_generic_function),   \
    uint32_t:      _Generic((v2),       \
        cp_uvec4:    cp_uvec4_divs2,    \
        cp_uvec3:    cp_uvec3_divs2,    \
        cp_uvec2:    cp_uvec2_divs2,      \
        default:    invalid_type_for_generic_function)    \
    ) (v1, v2)

#define dot(v1, v2) _Generic((v1), \
    cp_vec4:    cp_vec4_dot,       \
    cp_vec3:    cp_vec3_dot,       \
    cp_vec2:    cp_vec2_dot,       \
    cp_ivec4:    cp_ivec4_dot,     \
    cp_ivec3:    cp_ivec3_dot,     \
    cp_ivec2:    cp_ivec2_dot,     \
    cp_uvec4:    cp_uvec4_dot,     \
    cp_uvec3:    cp_uvec3_dot,     \
    cp_uvec2:    cp_uvec2_dot     \
    ) (v1, v2)

#define normalize(v1) _Generic((v1), \
    cp_vec4:    cp_vec4_normalize,       \
    cp_vec3:    cp_vec3_normalize,       \
    cp_vec2:    cp_vec2_normalize       \
    ) (v1)

#define length(v1) _Generic((v1), \
    cp_vec4:    cp_vec4_length,       \
    cp_vec3:    cp_vec3_length,       \
    cp_vec2:    cp_vec2_length       \
    ) (v1)

#define cross(v1, v2) _Generic((v1), \
    cp_vec3:    cp_vec3_cross       \
    ) (v1, v2)

FORCE_INLINE static mat4 scaleMat(vec3 s)
{
    return (mat4) {
            s.x, 0, 0, 0,
            0, s.y, 0, 0,
            0, 0, s.z, 0,
            0, 0, 0, 1
    };
}

FORCE_INLINE static mat4 translateMat(vec3 t)
{
    return (mat4) {
            1, 0, 0, 0,
            0, 1, 0, 0,
            0, 0, 1, 0,
            t.x, t.y, t.z, 1
    };
}

FORCE_INLINE static mat4 rotateMat(vec4 r)
{
    float n = 2.0f / (r.w * r.w + r.x * r.x + r.y * r.y + r.z * r.z);

    return (mat4) {
            1.0f - n * r.y * r.y - n * r.z * r.z, n * r.x * r.y + n * r.z * r.w, n * r.x * r.z - n * r.y * r.w, 0.0f,
            n * r.x * r.y - n * r.z * r.w, 1.0f - n * r.x * r.x - n * r.z * r.z, n * r.y * r.z + n * r.x * r.w, 0.0f,
            n * r.x * r.z + n * r.y * r.w, n * r.y * r.z - n * r.x * r.w, 1.0f - n * r.x * r.x - n * r.y * r.y, 0.0f,
            0, 0, 0, 1
    };
}

FORCE_INLINE static mat4 modelMatrix(vec3 pos, vec4 rot, vec3 scale)
{
    return (mat4) {
            (1.0f - 2 * rot.y * rot.y - 2 * rot.z * rot.z) * scale.x, (2 * rot.x * rot.y + 2 * rot.z * rot.w) * scale.y, (2 * rot.x * rot.z - 2 * rot.y * rot.w) * scale.z, 0.0f,
            (2 * rot.x * rot.y - 2 * rot.z * rot.w) * scale.x, (1.0f - 2 * rot.x * rot.x - 2 * rot.z * rot.z) * scale.y, (2 * rot.y * rot.z + 2 * rot.x * rot.w) * scale.z, 0.0f,
            (2 * rot.x * rot.z + 2 * rot.y * rot.w) * scale.x, (2 * rot.y * rot.z - 2 * rot.x * rot.w) * scale.y, (1.0f - 2 * rot.x * rot.x - 2 * rot.y * rot.y) * scale.z, 0.0f,
            pos.x, pos.y, pos.z, 1.0f
    };
}

FORCE_INLINE static mat4 perspectiveProjectionMatrix(float fov, float aspect_ratio, float zNear, float zFar)
{
  const float tanFovHalf = tanf(fov / 2.0f);

  return (mat4) {
          1 / (aspect_ratio * tanFovHalf), 0, 0, 0,
          0, 1 / tanFovHalf, 0, 0,
          0, 0, (zFar + zNear) / (zNear - zFar), -1,
          0, 0, 2 * zFar * zNear / (zNear - zFar), 0
  };
}

FORCE_INLINE static mat4 worldToCamMatrix(vec3 camPos, vec3 forward, vec3 globalUp)
{
    const vec3 right = normalize(cross(forward, globalUp));
    const vec3 up = normalize(cross(right, forward));
    const vec3 backward = normalize(mul(forward, -1.0f));

    return (mat4) {
            right.x, up.x, backward.x, 0,
            right.y, up.y, backward.y, 0,
            right.z, up.z, backward.z, 0,
            -dot(right, camPos), -dot(up, camPos), -dot(backward, camPos), 1
    };
}

FORCE_INLINE static mat4 camToWorldMatrix(vec3 camPos, vec3 forward, vec3 globalUp)
{
    const vec3 right = normalize(cross(forward, globalUp));
    const vec3 up = normalize(cross(right, forward));

    return (mat4) {
            right.x, right.y, right.z, 0,
            up.x, up.y, up.z, 0,
            forward.x, forward.y, forward.z, 0,
            camPos.x, camPos.y, camPos.z, 1
    };
}

FORCE_INLINE static mat4 matMul(mat4 m1, mat4 m2)
{
    mat4 r;
    __m128 broad;
    #pragma GCC unroll 4
    for (u32 i = 0; i < 4; i++)
    {
        broad = _mm_set1_ps(m2.arr[i * 4]);
        r.i[i] = _mm_mul_ps(m1.i[0], broad);

        #pragma GCC unroll 4
        for (u32 j = 1; j < 4; j++)
        {
            broad = _mm_set1_ps(m2.arr[(i * 4) + j]);
            r.i[i] = _mm_add_ps(_mm_mul_ps(m1.i[j], broad), r.i[i]);
        }
    }
    return r;
}

FORCE_INLINE static mat4 modelSlow(vec3 pos, vec4 rot, vec3 scale)
{
    return matMul(scaleMat(scale), matMul(translateMat(pos), rotateMat(rot)));
}

FORCE_INLINE static mat4 transpose(mat4 m)
{
    return (mat4) {
        m.arr[0], m.arr[4], m.arr[8], m.arr[12],
        m.arr[1], m.arr[5], m.arr[9], m.arr[13],
        m.arr[2], m.arr[6], m.arr[10], m.arr[14],
        m.arr[3], m.arr[7], m.arr[11], m.arr[15],
    };
}

FORCE_INLINE static mat4 matIdentity(void)
{
    return (mat4) {
            1, 0, 0, 0,
            0, 1, 0, 0,
            0, 0, 1, 0,
            0, 0, 0, 1
    };
}

FORCE_INLINE static vec4 quatIdentity(void)
{
    return (vec4) {0, 0, 0, 1};
}

FORCE_INLINE static vec4 quat_compose(vec4 q, vec4 r)
{
    // TODO: SSE shuffle + benchmark
    return (vec4) {
            r.x * q.w - r.y * q.z + r.z * q.y + r.w * q.x,
            r.w * q.y + r.y * q.w + r.z * q.x - r.x * q.z,
            r.w * q.z + r.z * q.w + r.x * q.y - r.y * q.x,
            r.w * q.w - r.x * q.x - r.y * q.y - r.z * q.z
    };
}

FORCE_INLINE static vec4 quat_betweenUnitVecs(vec3 from, vec3 to)
{
    vec3 a = cross(from, to);
    return normalize(((vec4) {.x = a.x, .y = a.y, .z = a.z, .w = 1 + dot(from, to)}));
}

FORCE_INLINE static vec4 quat_fromUnitAxisRotation(vec3 axis, float angle)
{
    angle *= 0.5f;
    float sa = sinf(angle);

    return normalize(((vec4) {
           axis.x * sa,
           axis.y * sa,
           axis.z * sa,
           cosf(angle)
    }));
}

FORCE_INLINE static vec4 quat_fromEulerX(float x)
{
    return normalize(((vec4) {
            sinf(x / 2),
            0,
            0,
            cosf(x / 2)
    }));
}

FORCE_INLINE static vec4 quat_fromEulerY(float y)
{
    return normalize(((vec4) {
            0,
            sinf(y / 2),
            0,
            cosf(y / 2)
    }));
}

FORCE_INLINE static vec4 quat_fromEulerZ(float z)
{
    return normalize(((vec4) {
            0,
            0,
            sinf(z / 2),
            cosf(z / 2)
    }));
}

FORCE_INLINE static vec4 lerp(vec4 v0, vec4 v1, float t)
{
    return normalize(add(v0, mul(sub(v1, v0), t)));
}

FORCE_INLINE static vec4 slerp(vec4 q1, vec4 q2, float t)
{
    float af = 1.0f - t;
    float bf = t;
    float l = dot(q1, q2);
    float al = fabsf(l);

    if(al < 0.999f)
    {
        al = acosf(al);
        bf = 1 / sinf(al);
        af = sinf(af * al) *bf;
        bf *= sinf(t * al);

        if(l < 0)
            bf = -bf;
    }

    return add(mul(q1, af), mul(q2, bf));
}

//FORCE_INLINE static u32 log2ui(u32 x) {
//  u32 y;
//  asm ( "\tbsr %1, %0\n"
//      : "=r"(y)
//      : "r" (x)
//  );
//  return y;
//}

FORCE_INLINE static u32 log2ui(u32 x)
{
    int r = 0;
    while (x >>= 1) ++r;
    return r;
}


#endif //CPMATH_H
