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

static inline const char *u64_commas(uint64_t value, char *out);

/**
 * @brief Format the shared column grid every gate prints with (into @p out, >= 192 bytes).
 *
 * One grid so the fields common to several tests (node count, time, Mnps, and the trailing FEN/note) sit at
 * the same column in every test's output, with absent fields left blank but padded:
 *
 *     name   detail    <          nodes>  <aux — per-test extras >    <time>  <    Mnps>  FEN / note
 *     %-6s   %-9s      <%15s nodes>       <%-25s>                    <%8.3fs> <%6.1f Mnps>
 *
 * @param name test name ("perft", "see", ...).
 * @param detail leading per-case info: "depth 13", the SEE move, "reject"; summaries use "passed/total".
 * @param nodes node count, or a negative value to leave the column blank.
 * @param aux per-test extra fields (mismatch count, SEE scores, polyglot key), or NULL.
 * @param secs elapsed seconds, or a negative value to leave the column blank.
 * @param mnps throughput in Mnps, or a negative value to leave the column blank.
 * @param tail trailing FEN or "(note)"; always the last column, so FENs align across all tests.
 *
 * The fixed prefix is 94 characters; with the longest legal FEN the line stays comfortably under 200.
 */
static inline const char *test_columns(char *out, const char *name, const char *detail, const int64_t nodes,
                                       const char *aux, const double secs, const double mnps, const char *tail)
{
    char nodes_col[32] = "";
    if (nodes >= 0)
    {
        char commas[27];
        snprintf(nodes_col, sizeof nodes_col, "%15s nodes", u64_commas((uint64_t)nodes, commas));
    }
    char time_col[16] = "";
    if (secs >= 0.0)
    {
        snprintf(time_col, sizeof time_col, "%8.3fs", secs);
    }
    char mnps_col[16] = "";
    if (mnps >= 0.0)
    {
        snprintf(mnps_col, sizeof mnps_col, "%6.1f Mnps", mnps);
    }
    snprintf(out, 192, "%-6s %-9s %21s %-25s %9s %11s %s", name, detail ? detail : "", nodes_col, aux ? aux : "",
             time_col, mnps_col, tail ? tail : "");
    return out;
}

/** @brief test_columns + the [PASS]/[FAIL] tag: the one-call form used by every gate's per-case/summary line. */
static inline void test_result_columns(const bool is_pass, const char *name, const char *detail, const int64_t nodes,
                                       const char *aux, const double secs, const double mnps, const char *tail)
{
    char line[192];
    test_result(is_pass, "%s", test_columns(line, name, detail, nodes, aux, secs, mnps, tail));
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
