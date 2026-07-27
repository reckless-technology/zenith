---
name: partial-commit-staged-state
description: When excluding files from a commit, verify the COMMITTED state builds — not the working tree
metadata:
  type: feedback
---

While an experiment lives uncommitted in a file (e.g. search.c during an SPRT), a refactor that touches the
same file (an include-path move, a rename) can get its fix silently excluded when that file is left out of
the commit — the working tree builds fine while the committed tree is broken (5 red CI runs in Zenith,
2026-07-26, ln_tables.inc move).

**Why:** local verification used the working tree; the breakage existed only in the committed state, and no
CI watcher was armed for those pushes.

**How to apply:** (1) when excluding files from a commit, grep the excluded files for references to anything
the commit changes; (2) verify the staged state itself (`git stash --keep-index` + build, or build a clean
checkout of HEAD after committing); (3) arm a CI watcher for every push, no exceptions.
