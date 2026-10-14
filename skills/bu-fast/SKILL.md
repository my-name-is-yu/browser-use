---
name: bu-fast
description: Complete a natural-language browser goal with Decisions, cited answers and requested files through the run workflow.
---

# bu-fast

Use this checkout's CLI through uv. Read BU_FAST.md for the current interface.

1. Preserve the user's whole authorized goal, requested files and constraints. Use `uv run bu-fast --url URL --goal GOAL` and repeat `--input VALUE` for known exact inputs. For multiple requirements, derive a PurposePlan with `conditions` containing each `id` and full `requirement`, then pass `--plan PATH`. Alternatively pass a WorkflowTask JSON through `--task PATH`. Never print API keys.
2. The workflow handles exploration, input completion, retained evidence search, browser recovery and cited delivery. Set `--timeout`, `--parent-turns` and `--max-decisions` when needed. It starts its own browser.
3. Read the returned `status`, `directory`, `final_answer` and `error`. Completion requires exit code 0 and status complete, which is returned only after citation validation and file saving. Return the saved answer and links to requested files under `outputs`.
4. For incomplete, blocked, error or timeout, report the specific gap. Preserve unsupported fields as missing. Do not silently rerun the task or replay unknown-effect operations. Check saved observations when reviewing citations or input/output states; quote validation does not guarantee semantic correctness.
