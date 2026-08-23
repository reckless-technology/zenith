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
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <sys/ioctl.h>
#include <unistd.h>

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
 * @param out destination buffer for the formatted line (>= 192 bytes).
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

/// @name In-place progress for the long-running walks
/// @{

enum
{
    TEST_TAG_WIDTH = 7 ///< width of the leading "[PASS] " / "[FAIL] " / "[ .. ] " tag, common to every line
};

/** @brief The terminal's width in columns (80 if it cannot be determined). */
static inline size_t test_terminal_columns(void)
{
    struct winsize window;
    if (ioctl(fileno(stdout), TIOCGWINSZ, &window) == 0 && window.ws_col > 0)
    {
        return (size_t)window.ws_col;
    }
    return 80;
}

/**
 * @brief Rewrite an in-place progress line for a gate that is mid-case, on the shared column grid.
 *
 * The deep differential walks (legalcheck, nnuecheck) spend seconds inside a single position and otherwise
 * print nothing until it finishes, which reads as a hang. This keeps a live node count on screen meanwhile.
 *
 * Formatted through test_columns, so the node and time columns sit exactly where the [PASS] result lines put
 * them and the progress line reads as the same table mid-flight. Trailing blank columns are trimmed so the
 * cursor parks just after the last populated field. @p detail carries the case counter ("#3/8") in the column
 * the summary line uses for passed/total.
 *
 * Emitted ONLY to a terminal: redirected output (CI logs, `make check | tee`) stays byte-for-byte identical to
 * a run without progress. The '\r' rewrite cannot undo a wrapped line, so if the full grid would not fit the
 * terminal this drops the time column, and if even that would not fit it prints nothing.
 */
static inline void test_progress(const char *name, const char *detail, const int64_t nodes, const double secs)
{
    if (!isatty(fileno(stdout)))
    {
        return;
    }
    const size_t columns = test_terminal_columns();
    // Widest form first (nodes + elapsed); on a narrow terminal retry with the time column blanked, which the
    // trailing-blank trim below then drops entirely.
    for (int attempt = 0; attempt < 2; attempt++)
    {
        char   line[192];
        size_t length = strlen(test_columns(line, name, detail, nodes, NULL, attempt == 0 ? secs : -1.0, -1.0, NULL));
        while (length > 0 && line[length - 1] == ' ')
        {
            length--;
        }
        if (TEST_TAG_WIDTH + length <= columns)
        {
            printf("\r[ .. ] %.*s\033[K", (int)length, line); // \033[K: erase the previous, longer line's tail
            fflush(stdout);
            return;
        }
    }
}

/** @brief Erase a test_progress line so the result line that follows starts on clean columns. */
static inline void test_progress_clear(void)
{
    if (!isatty(fileno(stdout)))
    {
        return;
    }
    fputs("\r\033[K", stdout);
    fflush(stdout);
}

/// @}

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
