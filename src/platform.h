// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Jonny Reckless
/**
 * @file
 * @brief Thin portability shim: threads, a monotonic millisecond clock, and stdout terminal queries.
 *
 * On POSIX toolchains with C11 threads (glibc >= 2.28) we wrap <threads.h>; on Apple and toolchains that
 * ship no <threads.h> (e.g. mingw-w64 clang on Windows) we fall back to pthreads (winpthreads on Windows).
 */
#pragma once
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>

typedef int (*zen_thread_fn)(void *); ///< thread entry point (return value ignored)

#if defined(__APPLE__) || defined(__STDC_NO_THREADS__) || !__has_include(<threads.h>)

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

// Monotonic wall clock in milliseconds (CLOCK_MONOTONIC on POSIX, QueryPerformanceCounter on Windows),
// and the OS-reported logical CPU count (GetSystemInfo / sysconf).
#if defined(_WIN32)

#include <windows.h>

static inline int64_t platform_now_ms(void)
{
    LARGE_INTEGER frequency, counter;
    QueryPerformanceFrequency(&frequency);
    QueryPerformanceCounter(&counter);
    return (int64_t)(counter.QuadPart * 1000 / frequency.QuadPart);
}

/** @brief Logical CPUs the OS reports (>= 1). */
static inline int platform_cpu_count(void)
{
    SYSTEM_INFO info;
    GetSystemInfo(&info);
    return info.dwNumberOfProcessors > 0 ? (int)info.dwNumberOfProcessors : 1;
}

#else

#include <time.h>

static inline int64_t platform_now_ms(void)
{
    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    return (int64_t)now.tv_sec * 1000 + now.tv_nsec / 1000000;
}

/** @brief Logical CPUs the OS reports (>= 1). */
#if defined(__APPLE__)
#include <sys/sysctl.h>
#include <sys/types.h>

// Darwin hides _SC_NPROCESSORS_ONLN (an XSI extension) under strict _POSIX_C_SOURCE, so use the native
// sysctl. Broke the release-only macOS build unnoticed: CI never compiles for macOS (10x billing).
static inline int platform_cpu_count(void)
{
    int    count = 0;
    size_t size  = sizeof count;
    return sysctlbyname("hw.logicalcpu", &count, &size, NULL, 0) == 0 && count > 0 ? count : 1;
}
#else
#include <unistd.h>

static inline int platform_cpu_count(void)
{
    const long count = sysconf(_SC_NPROCESSORS_ONLN);
    return count > 0 ? (int)count : 1;
}
#endif

#endif

// Stdout terminal queries, for output that only makes sense on a terminal (the test gates' in-place progress
// lines). Windows has no <sys/ioctl.h>/TIOCGWINSZ, which is why these live behind the shim rather than in the
// caller — a raw ioctl in a shared header broke the Windows build (CI's only non-Linux compile).
#if defined(_WIN32)

#include <io.h>

/** @brief Whether stdout is a terminal rather than a pipe or file. */
static inline bool platform_stdout_is_terminal(void)
{
    return _isatty(_fileno(stdout)) != 0;
}

/** @brief Stdout's width in columns (80 when it cannot be determined). */
static inline int platform_terminal_columns(void)
{
    CONSOLE_SCREEN_BUFFER_INFO info;
    if (GetConsoleScreenBufferInfo(GetStdHandle(STD_OUTPUT_HANDLE), &info))
    {
        const int columns = info.srWindow.Right - info.srWindow.Left + 1;
        if (columns > 0)
        {
            return columns;
        }
    }
    return 80;
}

#else

#include <sys/ioctl.h>
#include <unistd.h>

/** @brief Whether stdout is a terminal rather than a pipe or file. */
static inline bool platform_stdout_is_terminal(void)
{
    return isatty(fileno(stdout)) != 0;
}

/** @brief Stdout's width in columns (80 when it cannot be determined). */
static inline int platform_terminal_columns(void)
{
    struct winsize window;
    if (ioctl(fileno(stdout), TIOCGWINSZ, &window) == 0 && window.ws_col > 0)
    {
        return (int)window.ws_col;
    }
    return 80;
}

#endif
