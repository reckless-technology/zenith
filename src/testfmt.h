// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Jonny Reckless
/**
 * @file
 * @brief Shared formatting for the CLI self-test gates so every test prints uniform, well-formed output.
 *
 * Every per-case and summary line begins with a "[PASS]" or "[FAIL]" tag (via test_result), followed by a
 * short test name and that case's specific, column-aligned details. u64_commas groups large counts for
 * readability. Keeping this in one place is what makes `make check` read as one clean table.
 */
#pragma once
#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>

/** @brief Print one result line: "[PASS] "/"[FAIL] " then the formatted detail, with a trailing newline. */
static inline void test_result(const bool is_pass, const char *fmt, ...)
{
    fputs(is_pass ? "[PASS] " : "[FAIL] ", stdout);
    va_list args;
    va_start(args, fmt);
    vprintf(fmt, args);
    va_end(args);
    putchar('\n');
}

/**
 * @brief Write @p value as a thousands-grouped decimal string into @p out (needs >= 27 bytes). @return @p out.
 *
 * Portable (no locale): e.g. 119060324 -> "119,060,324". Handy for the big perft / node counts.
 */
static inline const char *u64_commas(uint64_t value, char *out)
{
    char digits[20];
    int  length = 0;
    do
    {
        digits[length++] = (char)('0' + (int)(value % 10));
        value /= 10;
    } while (value);

    int written = 0;
    for (int i = 0; i < length; i++)
    {
        if (i != 0 && (length - i) % 3 == 0)
        {
            out[written++] = ',';
        }
        out[written++] = digits[length - 1 - i];
    }
    out[written] = '\0';
    return out;
}
