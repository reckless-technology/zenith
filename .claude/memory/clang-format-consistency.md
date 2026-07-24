---
name: clang-format-consistency
description: Always clang-format every source/header file with the repo .clang-format; keep formatting consistent
metadata:
  type: feedback
---

Run `clang-format -i src/*.c src/*.h` before finishing any C work in Zenith (the engine is C17 since 2026-07-24; C++ history at tag cpp-final). The repo `.clang-format`
was copied from pawnstar (BasedOnStyle Microsoft, ColumnLimit 120, InsertBraces true, aligned
consecutive macros/assignments/declarations, SeparateDefinitionBlocks Always).

**Why:** The user keeps a single consistent style across their engines and dislikes format drift.

**How to apply:** After editing any `.c`/`.h`, format it, then rebuild to confirm nothing broke.
clang-format 18 lives at `~/.local/bin/clang-format`. Related: [[prefer-full-names]]
