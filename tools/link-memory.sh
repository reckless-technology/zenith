#!/usr/bin/env bash
# Wire Claude Code's memory system to this repo's versioned memories.
#
# The memory files live in <repo>/.claude/memory/ (committed to git, so every clone gets them). Claude Code,
# however, reads/writes ~/.claude/projects/<project-slug>/memory/ — a per-machine path git cannot carry. Run
# this once after cloning so that path points at the repo copy; then memories are shared via the repository.
set -euo pipefail

repo="$(cd "$(dirname "$0")/.." && pwd)"
# Claude Code derives the project directory from the repo's absolute path with every '/' replaced by '-'.
slug="$(printf '%s' "$repo" | sed 's/\//-/g')"
target="$HOME/.claude/projects/$slug/memory"

mkdir -p "$(dirname "$target")"
if [ -e "$target" ] && [ ! -L "$target" ]; then
    echo "refusing to clobber existing directory $target" >&2
    echo "move its contents into $repo/.claude/memory/ and re-run" >&2
    exit 1
fi
rm -f "$target"
ln -s "$repo/.claude/memory" "$target"
echo "linked $target -> $repo/.claude/memory"
