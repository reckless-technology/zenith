// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Jonny Reckless
/**
 * @file
 * @brief CRC-64/XZ — the integrity check the shared Reckless NNUE net container (`RNNUE1`) carries.
 *
 * One algorithm, every engine: `pawnstar`, `pawnstar-rs`, `pawnstar-go` and Zenith stamp and verify a net
 * with this same checksum, so an implementation that differs by a bit rejects every net the others accept.
 * It is **CRC-64/XZ**, also called CRC-64/GO-ECMA:
 *
 *     polynomial  0x42F0E1EBA9EA3693, reflected (so the table constant is 0xC96C5795D7870F42)
 *     init        0xFFFFFFFFFFFFFFFF
 *     reflect     in and out
 *     final XOR   0xFFFFFFFFFFFFFFFF
 *
 * The **check value** every implementation must reproduce — CRC-64/XZ of the nine ASCII bytes `123456789`
 * is `0x995DC9BBDF1939FA` — is asserted by `crc64_is_check_value_reproduced()`, which `nnuecheck` runs
 * before it looks at a net. C17 has no constant-evaluated loop, so unlike the C++ port's `static_assert`
 * this cannot be a compile-time check; `make check` is where a wrong table is caught, and the check value
 * exists precisely because a CRC with the wrong reflection still looks like a plausible 64-bit number.
 *
 * A 64-bit CRC and not a 32-bit one because the payload is ~12.6 MB; a CRC and not a cryptographic hash
 * because this guards against accidental corruption (a truncated download, a half-written file, a net
 * stamped by the wrong build), not against tampering.
 *
 * The 256-entry table is built on the stack by each call rather than kept in a file-scope array: 2 KB and
 * 2,048 iterations against a 12.6 MB payload is nothing, a net is checksummed a handful of times in a
 * process's life, and it keeps the no-globals rule whole (see CLAUDE.md, "No globals").
 */
#pragma once
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

enum
{
    CRC64_TABLE_SIZE = 256 ///< entries in the byte-at-a-time lookup table
};

/// The reflected form of the CRC-64/XZ polynomial 0x42F0E1EBA9EA3693.
#define CRC64_REFLECTED_POLYNOMIAL UINT64_C(0xC96C5795D7870F42)
/// The value a CRC-64/XZ register starts at, and the value it is XORed with at the end.
#define CRC64_INITIAL_AND_FINAL_XOR UINT64_C(0xFFFFFFFFFFFFFFFF)
/// The published CRC-64/XZ check value: the checksum of the nine ASCII bytes "123456789".
#define CRC64_CHECK_VALUE UINT64_C(0x995DC9BBDF1939FA)

/** @brief Fill the 256-entry byte-at-a-time table: entry i is the register state after feeding byte i
 *  into a zeroed register. @p table receives CRC64_TABLE_SIZE entries. */
static inline void crc64_fill_table(uint64_t *const table)
{
    for (int byte = 0; byte < CRC64_TABLE_SIZE; byte++)
    {
        uint64_t remainder = (uint64_t)byte;
        for (int bit = 0; bit < 8; bit++)
        {
            remainder = (remainder & 1u) ? (remainder >> 1) ^ CRC64_REFLECTED_POLYNOMIAL : remainder >> 1;
        }
        table[byte] = remainder;
    }
}

/** @brief CRC-64/XZ of a byte range, init and final XOR applied.
 *  @param data first byte (may be NULL only when @p length is zero). @param length bytes to checksum.
 *  @return the checksum. */
static inline uint64_t crc64_checksum(const void *const data, const size_t length)
{
    uint64_t table[CRC64_TABLE_SIZE];
    crc64_fill_table(table);
    const unsigned char *const bytes   = (const unsigned char *)data;
    uint64_t                   running = CRC64_INITIAL_AND_FINAL_XOR;
    for (size_t index = 0; index < length; index++)
    {
        running = table[(running ^ bytes[index]) & 0xFFu] ^ (running >> 8);
    }
    return running ^ CRC64_INITIAL_AND_FINAL_XOR;
}

/** @brief Whether this build reproduces the published CRC-64/XZ check value — the one assertion that tells
 *  a correct implementation from a plausible-looking wrong one. @return true when it does. */
static inline bool crc64_is_check_value_reproduced(void)
{
    return crc64_checksum("123456789", 9) == CRC64_CHECK_VALUE && crc64_checksum(NULL, 0) == 0;
}
