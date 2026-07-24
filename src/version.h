#pragma once
// Engine version constants and the "major.minor.build" version string.
//
// The major and minor numbers are edited by hand below (the human-meaningful version). The build number is
// supplied at compile time via the ZENITH_BUILD_NUMBER macro, which the Makefile derives from the git commit
// count (`git rev-list --count HEAD`) — a globally-reproducible number, identical for every clone at a given
// commit (mirrors pawnstar's scheme). It defaults to 0 so that editor tooling and builds outside a git
// checkout still compile cleanly without the define.

#define ZENITH_VERSION_MAJOR 1
#define ZENITH_VERSION_MINOR 0

#ifndef ZENITH_BUILD_NUMBER
#define ZENITH_BUILD_NUMBER 0
#endif

// Preprocessor stringification so the full version is a compile-time string literal.
#define ZENITH_STRINGIFY_INNER(x) #x
#define ZENITH_STRINGIFY(x)       ZENITH_STRINGIFY_INNER(x)

// The engine version as a "major.minor.build" string literal (e.g. "1.0.59").
#define ZENITH_VERSION_STRING              \
    ZENITH_STRINGIFY(ZENITH_VERSION_MAJOR) \
    "." ZENITH_STRINGIFY(ZENITH_VERSION_MINOR) "." ZENITH_STRINGIFY(ZENITH_BUILD_NUMBER)
