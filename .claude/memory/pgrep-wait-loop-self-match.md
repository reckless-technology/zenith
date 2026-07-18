---
name: pgrep-wait-loop-self-match
description: Wait-loops using `pgrep -f '<prog> <arg>'` deadlock by matching their own script's command line
metadata:
  type: feedback
---

`until [ "$(pgrep -f 'zenith datagen' | wc -l)" -eq 0 ]; do sleep 3; done` never terminates: the wait
script (and any sibling script whose command line contains the literal string "zenith datagen", including
a chained trainer launcher) is itself matched by `pgrep -f`, so the count never reaches 0.

**Why:** `pgrep -f` matches the full command line of every process, including the polling shell itself.

**How to apply:** To wait on a real binary finishing, match the executable name exactly with `pgrep -x zenith`
(process name, not args), or use the bracket trick `pgrep -f '[z]enith datagen'`. Better: launch background
jobs and rely on the harness completion notification instead of a self-referential poll loop.
