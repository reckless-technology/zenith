---
name: prefer-simple-config-solutions
description: "For cosmetic/config tweaks, Jonny prefers an imperfect simple setting over invasive machinery with ongoing maintenance"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 9f1d08cf-47ee-4603-9901-030b61ef9010
  modified: 2026-07-31T05:04:21.735Z
---

When a cosmetic or configuration goal can only be hit exactly via invasive machinery (patching system files, changing root ownership, per-update re-fix rituals), Jonny would rather settle for a close-enough simple setting.

**Why:** In July 2026, sizing the Claude Code panel's Bash IN/OUT text precisely required the Custom UI Style extension to patch VS Code's install files (root chown, CSP re-signing, re-patching after every apt update). Midway through the permission-error mess he cut it off with "Forget it. Just make the chat font a bit bigger" — the simple `chat.fontSize` bump won even though it scales the prose too.

**How to apply:** When an exact solution is invasive and an approximate one is a plain setting, lead with the simple option and state the invasive one's recurring maintenance cost up front — let him opt into the machinery, and expect him not to.
