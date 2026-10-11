---
name: bu-fast
description: Run a natural-language browser goal with Decisions, supply missing input text in Codex, and compose sourced answers from saved observations.
---

Use this checkout's CLI through uv. Read BU_FAST.md for the current interface.

1. Pass the authorized start URL, the entire browser goal and any known exact inputs to `uv run bu-fast start --url URL --goal GOAL --input VALUE`. No Job or extraction contract is needed. Do not print API keys.
2. If status is needs_input, inspect input_target and the returned current page. Derive the exact input string from the user's goal and observations. Supply it with `uv run bu-fast resume RESULT_PATH --input VALUE`. The saved browser, goal, history and budget remain unchanged.
3. For done, read the needed saved journal observations. Use the adopted A delivery: compose one CompleteDelivery JSON with disposition, browser_summary, records (id/value/quote/source_url), issues, and a cited final_answer. Then call `uv run bu-fast complete RESULT_PATH --delivery DELIVERY_JSON_PATH --required-field ID` (repeat --required-field for each requested item). Code validates citations and renders results.json, report.md and final.txt from that single delivery. Return the saved final_answer to the user with artifact links; do not generate the same records separately for each file. Preserve linked-row identity and baseline/changed state attribution. Missing fields use null value/quote/source_url and disposition incomplete. If more evidence is needed, read it first; continue is not a terminal delivery. A DONE claim alone is not evidence, and citation validation is not semantic grading.
4. For blocked, error, timeout or unknown, report the reason or use separately authorized direct browser inspection. Do not automatically replay unknown-effect operations. No patch, recover, details or generic replanning API exists.

The dedicated browser remains open on needs_input or unknown. Saved run paths are returned by the CLI. Never attach to an unrelated user's browser tab. Site-specific action recipes are not part of this skill.
