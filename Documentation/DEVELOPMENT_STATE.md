# Development state checkpoint (2026-10-14)

Written when work was paused. HEAD at the time: `b68446e` (local, 1 commit ahead of `origin/main` = `3b5cf51`). Nothing uncommitted except the untracked `CLAUDE.md`. `stash@{0}` ("local action_schema header comment") is untouched.

## Established (pushed unless noted)
- Per-agent PilotAdapters; Procedural Baseline through the canonical contract; validation progression; capacity accounting; `bridge_bc` progression; rejected-action feedback; deterministic run seed in the manifest; simulator-owned construction correctness; real final measures + terminal state.
- `d783bd9`: carried bricks count as a transport source; build-target lookup no longer creates projects (creation only on the START commit); zero-capacity sites and the whiteboard are not offered as targets; RuleBrain `new_structure_needed` feature.
- `3b5cf51`: test-only fix for the leaked fake Matplotlib in `tests/test_analysis_gui_replay.py` (it caused the modal "Plot initialization failed" and hung pytest). `interface.py` itself opens fine in `.venv`.
- `b68446e` (LOCAL, NOT PUSHED): Mars resource exhaustion is non-terminal. `task_manifest.json` `terminal_policy.end_on_resources_exhausted=false`; `_check_terminal_state` logs a one-time `resources_exhausted` event and sets `resources_exhausted_time` (also in final measures `run`); mission still ends at the summed phase durations with `time_expired`; engine default keeps early termination. Tests: `tests/test_mission_timing_semantics.py` (+2 updated assertions in `test_session_artifacts.py`). 18 focused tests passed.

## Task semantics decided with the user (source-grounded)
- Phase 1 (30 min) and Phase 2 (20 min) are clock-driven; the bridge gates Site C / second pile only, not Phase 2. Mission ends at configured total time. Outcome = number of arriving colonists the final colony supports, not a binary pass/fail. Phase 2 population is additive.
- DISPUTED, do NOT silently change: Phase 2 = +30 civ +20 VIP (Experimenter Protocol) vs +40 civ +20 VIP (earlier Task Design and current package); housing headroom 1 vs 2 bricks; connector endpoint spacing.
- Brick budget (100 + 100) provenance: introduced in commit `45e7793` (2026-03-23, "Rebuild construction as site/resource/bridge model"); `resource_nodes.csv` still says 999. No source doc found. Do not tune it to make the baseline succeed. A clearly labelled abundant-resources override (`SimulationState(construction_parameters={"pile_a_quantity":..,"pile_c_quantity":..})`) is acceptable for architecture-certification runs only.
- Do NOT implement a civilian/VIP allocation optimizer or scoring policy yet (human materials do not specify it).

## IN PROGRESS: phantom carried bricks (found, not yet fixed)
Diagnostic (seed 1, to t=1700 s) showed engineer_1 and botanist_1 repeating `transport_resources` blocked `construction_delivery_rejected` (12 identical failures each) from about t=1100 s, so the `resources_exhausted` event never fired for that seed (carried>0) and the agents loop. Seed 2 exhausts at 855 s and then coherently selects only `reassess_plan` (no impossible loops).
Root cause (world-truth bug, `modules/agent.py` ~11742 and ~11781, `modules/construction.py` `deliver_resource` ~1000 / `_consume_resource_for_site` ~888): pickup only sets `transport_state["carrying"]`; the pile is not decremented until delivery, where `_consume_resource_for_site` pulls from site inventory. Two agents can "pick up" the same last brick; the loser holds a phantom brick whose delivery is rejected forever.
Planned fix (not written):
1. `ConstructionManager.take_from_pile(pile_id, qty=1)` — decrement `site_resource_inventory[site]` and mirror `pile.quantity`; False if empty.
2. Agent pickup calls it; on failure stay in pickup stage (next tick the candidate list is empty and the existing `no_pickup_resource_source` terminal block applies).
3. `deliver_resource(..., from_carried=True)` skips the second consume.
4. `_reset_transport_episode` / the rebind block at ~11695 must return a held brick to its pile (`return_to_pile`) instead of discarding it, since bricks are now real.
5. Bridge delivery (`build_bridge_bc`, agent ~11612) does not use the carry state at all — separate quirk, note it, leave for later.
6. Tests: two agents racing for the last brick; delivery of a carried brick after the pile is empty; reset returns the brick; exhaustion event fires once piles and hands are empty. Then re-run the seed-1 diagnostic (~1700 s) and a seed-2 check.

## Remaining queue (user-approved order)
1. Finish the phantom-brick fix above, commit locally.
2. Post-exhaustion Pilot behaviour: add a grounded `productive_taskwork_available` feature to RuleBrain (from legal affordances/readiness), so WAIT/monitor is chosen from observation, not by middleware; tests that remaining validation/repair/delivery work is still chosen and that work resumes if circumstances change.
3. Support-accounting primitives (structure state only, no allocation): validated housing/food/water capacity from the actual connector graph. Known finding: the simulator lets ONE water generator connect to 14 houses (seed 2: 14 houses, 1 greenhouse, 1 generator, 15 connections each) although the rule is max 2 directly connected structures; the old `modules/metrics.py` proxy (`_colony_support_capacity_proxy`, hard-coded atoms 3/2/5/2/60/2/3/max-2) ignores the graph. Plan: task-package config for the atoms (with DIK provenance: D_ARCH_CAPACITY_PER_2X2, D_BOT_SOIL_SUPPORT_CAPACITY, D_ENG_WATERGEN_60_UNITS, D_ENG_WATERGEN_MAX_2_STRUCTURES, ...) + pure evaluator module; metrics should delegate to it. Greenhouse water supply is not representable by `attach_connector` (only house<->generator water, house<->greenhouse food).
4. Architecture-integrity audit: query/probe mutation, project creation during lookup, ambient repair, hidden world-truth into shared knowledge, RuleBrain bypass of PilotAdapter, action families mutating via proper subsystems. Fix only demonstrated violations, each with a test.
5. Phase 1 snapshot at t=1800 and a `mission` section with supported-population measures (after #3), then re-run seeds 0-5 on the default task and a labelled certification run.

## Known test state
- Focused regression set (21 files) = 143 passed / 25 failed, identical failures on the committed baseline (mostly `tests/test_rule_brain_repair_patch.py`). Full suite only ~65% exercised (whole-simulation tests are very slow); no failure seen only in the current tree. Other pre-existing failures: `test_dik_communication_flow.py` (28), `test_brain_scaffolding.py` (10), GUI `test_update_agent_table_renders_canvas_and_text` (fails alone; unrelated to the Matplotlib fix).
- Run tests with `.venv\Scripts\python.exe -m pytest ...` (system `py` has no pytest).

## Practical notes
- The Edit/Write tools are intermittently blocked by auto mode ("no usable answer"); the user authorized writing repo-local files through PowerShell/Python in that case (not destructive ops, not outside the repo, no force-push). Write multi-line commit messages to a file and use `git commit -F` (PowerShell mangles multi-line `-m`; avoid a BOM).
- Simulator prints "Agent logs saved" per step to stdout; redirect/filter in diagnostics. Diagnostic scripts live only in the session scratchpad (not in the repo): a 1700 s headless run takes ~3.5 minutes.
- Seed results before the exhaustion change (r0-r5, default task): every seed ended with 20 structures validated, `bridge_bc` complete, both piles at 0, site_c_house_011 unfinished; none had a defined success.
