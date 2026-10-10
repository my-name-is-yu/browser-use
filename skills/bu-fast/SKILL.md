---
name: bu-fast
description: Run a natural-language browser goal with Decisions, supply missing input text in Codex, and compose sourced answers from saved observations.
---

Use this checkout's CLI through uv. Read BU_FAST.md for the current interface.

1. Pass the authorized start URL, the entire browser goal and any known exact inputs to `uv run bu-fast start --url URL --goal GOAL --input VALUE`. No Job or extraction contract is needed. Do not print API keys.
2. If status is needs_input, inspect input_target and the returned current page. Derive the exact input string from the user's goal and observations. Supply it with `uv run bu-fast resume RESULT_PATH --input VALUE`. The saved browser, goal, history and budget remain unchanged.
3. For done, read the saved journal observations and compose the requested answer or files in Codex. Preserve quote/source URLs, linked-row identity and baseline/changed state attribution. A DONE claim alone is not evidence.
4. For blocked, error, timeout or unknown, report the reason or use separately authorized direct browser inspection. Do not automatically replay unknown-effect operations. No patch, recover, details or generic replanning API exists.

The dedicated browser remains open on needs_input or unknown. Saved run paths are returned by the CLI. Never attach to an unrelated user's browser tab. Site-specific action recipes are not part of this skill.
