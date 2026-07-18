---
name: prefer-full-names
description: User prefers full descriptive names for classes, functions, and variables over short abbreviations
metadata:
  type: feedback
---

Use full, descriptive names for classes, functions, and variables rather than terse abbreviations
(e.g. `whiteScore` not `ws`, `bestMove` not `m` where practical, `adjudicationSide` not `adjSide`).

**Why:** The user values readability and consistency across their engine codebases.

**How to apply:** In new Zenith code, spell names out. Short loop indices (`i`, `j`) are still fine. The
existing engine core is heavily abbreviated (`stm`, `epSq`, `byColor`, `pop_lsb`); match new code to the
preference and consider a rename pass if the user asks. Related: [[clang-format-consistency]]
