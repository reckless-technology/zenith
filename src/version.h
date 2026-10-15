// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Jonny Reckless
/**
 * @file
 * @brief The engine's "major.minor.build" version string.
 *
 * The version comes from git alone (tools/version.sh): MAJOR.MINOR is the nearest annotated tag vMAJOR.MINOR reachable
 * from the commit built, BUILD the number of commits since it. The Makefile defines ZENITH_VERSION_STRING from it; the
 * 0.0.0 default lets editor tooling and builds outside the Makefile compile without the define. Nothing here is edited
 * to release: tag main and push the tag.
 */
#pragma once

#ifndef ZENITH_VERSION_STRING
#define ZENITH_VERSION_STRING "0.0.0" ///< the engine version, e.g. "2.0.5"
#endif
