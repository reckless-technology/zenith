#pragma once
// Thin portability shim: threads + a monotonic millisecond clock. On POSIX toolchains with C11 threads
// (glibc >= 2.28) we wrap <threads.h>; on Apple / toolchains without C11 threads we fall back to pthreads.
#include <stdint.h>

typedef int (*zen_thread_fn)(void *);

#if defined(__APPLE__) || defined(__STDC_NO_THREADS__)

#include <pthread.h>
#include <stdlib.h>

typedef struct
{
    pthread_t     handle;
    zen_thread_fn fn;
    void         *arg;
} zen_thread_t;

static inline void *zen_thread_trampoline(void *raw)
{
    zen_thread_t *thread = (zen_thread_t *)raw;
    thread->fn(thread->arg);
    return NULL;
}

static inline int zen_thread_create(zen_thread_t *thread, zen_thread_fn fn, void *arg)
{
    thread->fn  = fn;
    thread->arg = arg;
    return pthread_create(&thread->handle, NULL, zen_thread_trampoline, thread) == 0 ? 0 : -1;
}

static inline void zen_thread_join(zen_thread_t *thread)
{
    pthread_join(thread->handle, NULL);
}

#else

#include <threads.h>

typedef struct
{
    thrd_t handle;
} zen_thread_t;

static inline int zen_thread_create(zen_thread_t *thread, zen_thread_fn fn, void *arg)
{
    return thrd_create(&thread->handle, fn, arg) == thrd_success ? 0 : -1;
}

static inline void zen_thread_join(zen_thread_t *thread)
{
    thrd_join(thread->handle, NULL);
}

#endif

// Monotonic wall clock in milliseconds (CLOCK_MONOTONIC on POSIX, QueryPerformanceCounter on Windows).
#if defined(_WIN32)

#include <windows.h>

static inline int64_t platform_now_ms(void)
{
    LARGE_INTEGER frequency, counter;
    QueryPerformanceFrequency(&frequency);
    QueryPerformanceCounter(&counter);
    return (int64_t)(counter.QuadPart * 1000 / frequency.QuadPart);
}

#else

#include <time.h>

static inline int64_t platform_now_ms(void)
{
    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    return (int64_t)now.tv_sec * 1000 + now.tv_nsec / 1000000;
}

#endif
