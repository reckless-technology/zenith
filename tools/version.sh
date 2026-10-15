#!/usr/bin/env bash
# Print the engine's version, MAJOR.MINOR.BUILD, from git alone - the rule every reckless-technology project follows:
# MAJOR.MINOR is the nearest annotated tag vMAJOR.MINOR reachable from HEAD, made by hand on main, and BUILD the number
# of commits since it, the tagged commit itself being 0. The release workflow's own three-component tags
# (vMAJOR.MINOR.BUILD) are not version marks and are excluded. A tree with no mark reachable - a source tarball, a
# shallow clone, a fork without the tags - prints 0.0.0, with a warning on stderr, which the release workflow refuses.
#
# The Makefile compiles it into the binary (-DZENITH_VERSION_STRING; src/version.h).
set -euo pipefail
describe=$(git -C "$(dirname "$0")/.." describe --tags --long --match 'v[0-9]*.[0-9]*' --exclude 'v*.*.*' 2> /dev/null || true)
if [[ "$describe" =~ ^v([0-9]+)\.([0-9]+)-([0-9]+)-g[0-9a-f]+$ ]]; then
    echo "${BASH_REMATCH[1]}.${BASH_REMATCH[2]}.${BASH_REMATCH[3]}"
else
    echo "tools/version.sh: no vMAJOR.MINOR tag is reachable from HEAD: the version is 0.0.0 (git fetch --tags)" >&2
    echo "0.0.0"
fi
