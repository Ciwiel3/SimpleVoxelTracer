// Copyright (c) 2021, Lars Knof, All rights reserved.

#ifndef CPLOG_H
#define CPLOG_H

/**
 * LOGGING
 */
// *********** CONFIGURE ***********
#define LOG_ENABLED
//#define LOG_SUPPRESS_ERROR
//#define LOG_SUPPRESS_WARNING
//#define LOG_SUPPRESS_INFO
//#define LOG_SUPPRESS_DEBUG
// *********************************

#ifdef LOG_ENABLED

#include <stdio.h>

#define PANIC(msg, ...) {fprintf(stderr, "[ERR] "); fprintf(stderr, msg, ##__VA_ARGS__); fprintf(stderr, "\n"); exit(1);}

#ifndef LOG_SUPPRESS_ERROR
#define LOG_ERROR(msg, ...) {fprintf(stderr, "[ERR] "); fprintf(stderr, msg, ##__VA_ARGS__); fprintf(stderr, "\n");}
#endif
#ifdef LOG_SUPPRESS_ERROR
#define LOG_ERROR(msg, ...)
#endif

#ifndef LOG_SUPPRESS_WARNING
#define LOG_WARNING(msg, ...) {fprintf(stderr, "[WAR] "); fprintf(stderr, msg, ##__VA_ARGS__); fprintf(stderr, "\n");}
#endif
#ifdef LOG_SUPPRESS_WARNING
#define LOG_WARNING(msg, ...)
#endif

#ifndef LOG_SUPPRESS_INFO
#define LOG_INFO(msg, ...) {printf("[INF] "); printf(msg, ##__VA_ARGS__); printf("\n");}
#endif
#ifdef LOG_SUPPRESS_INFO
#define LOG_INFO(msg, ...)
#endif

#ifndef LOG_SUPPRESS_DEBUG
#define LOG_DEBUG(msg, ...) {printf("[DEB] "); printf(msg, ##__VA_ARGS__); printf("\n");}
#endif
#ifdef LOG_SUPPRESS_DEBUG
#define LOG_DEBUG(msg, ...)
#endif

#endif // LOGS ENABLED
#ifndef LOG_ENABLED
#define LOG_ERROR(msg, ...) (msg)
#define LOG_WARNING(msg, ...) (msg)
#define LOG_INFO(msg, ...) (msg)
#define LOG_DEBUG(msg, ...) (msg)
#endif // LOGS NOT ENABLED

#define ASSERT(condition, msg) if(!(condition)) {LOG_ERROR(msg); exit(-1);}

#endif //CPLOG_H
