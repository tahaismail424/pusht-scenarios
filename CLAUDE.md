# pusht-scenarios

A package for producing **expert Push-T datasets under varied environment conditions**: physics (mass, friction, damping, PD gains, center of gravity), shape, color, scale, and goal pose. The output must load directly into stable-worldmodel (swm), LeWM and SMWM training and eval pipelines.

Pipeline: **configure condition → record human expert demos (gamepad teleop) → augment (noisy replay + SE(2) transforms) → simulate/filter/render → export swm-format dataset.**

The design discussion that led here is in `2026-10-24-134931-okay-few-questions-for-our-benchmark-analysis-w.txt`, an exported Claude Code session. The end of that file (from "okay, i think we're converging on a plan!") is the plan of record. This file distills it.

## Why this exists (research context)

- The lab is moving toward **adaptive/generalizable JEPA world models**: T2-z (arXiv 2610.22364, a trainable per-environment latent z) and RMA (arXiv 2107.04034). The modeling paradigm may change. The dataset must serve any adaptive method: T2-z-style z, RMA-style supervised factors, history/context encoders, and AdaJEPA-style test-time adaptation (arXiv 2606.32026).
- No public Push-T dataset has real expert demos across varied conditions. DINO-WM PushObj uses random actions. AdaJEPA reuses the default 18.5k set. swm's `collect_pusht_fov.py` uses `WeakPolicy`, which takes random actions near the block. Clean varied-condition expert data is a real contribution.
- **Hidden factors** (physics) are what a latent z is meant to infer. **Visible factors** (color, shape) mostly test the encoder. Keep the two kinds of shift separable in condition specs so experiments can report them separately.
- Datasets must store **ground-truth factor values per episode**, plus an `env_id` (T2-z needs a finite set of environment ids).

## Established facts (measured in the earlier session, don't re-derive)

About LeWM's `pusht_expert_train.h5` (HF `quentinll/lewm-pusht`, a zstd-compressed .h5; on the Grace cluster at `/scratch/user/tahaismail424/.stable_wm/datasets/pusht_expert_train.h5`, 46 GB):
- 18,685 episodes, 2.34M frames, episode lengths 49–246 steps (median about 123, so about 12 s at 10 Hz).
- Columns: `pixels` (N,224,224,3) uint8, `action` (N,2), `state` (N,7) = agent xy, block xy, block angle, agent velocity xy, `proprio` (N,4), `episode_idx`, `step_idx`, `ep_len`, `ep_offset`.
- `pixels` are stored in (100,224,224,3) chunks with Blosc-LZ4 compression, level 5, shuffle (HDF5 filter 32001).
- There are no `variation.*` columns. Every episode uses the default gray T, default colors, default physics, and a fixed goal (center, 45°).
- **It's 185 human demos × 101 copies each**, stored back to back. The copies are open-loop replays of the same-length action sequence, with i.i.d. Gaussian noise added to the actions (not the pixels):

  | copies | noise σ (action units) | share that still reach the goal |
  |---|---|---|
  | 11 | ~0 | — |
  | 10 each | 0.003, 0.005, 0.01, 0.015 | 88% (σ<0.02) |
  | 10 each | 0.02, 0.025, 0.03 | 49% |
  | 10 | 0.07 | 9% |
  | 10 | 0.15 | 0% |

  Overall, 64% of episodes end within 20 px and 20° of the goal. Failed replays are kept on purpose: off-expert states are good for world-model coverage.
- Replay is exact. Resetting swm's env to an episode's start state and replaying its actions reproduces the logged states (to about 1e-5 px) and the frames (to about 0.5/255 per pixel).
- Original provenance: Diffusion Policy's (about 200) human mouse demos, made with `demo_pusht.py`. DINO-WM then did noisy replay, but its generation code was never released.

## swm Push-T physics as actually implemented (verified with pymunk 7.3.1, 2026-10-24)

These results came from scripted pushes in a bare pymunk copy of swm's setup, not from reading the code alone:
- **Contact friction is 0.** swm sets `body.friction = 1`, but `friction` is a *Shape* property in pymunk, so the line is a no-op. Shape friction defaults to 0, which makes agent–block contact frictionless. Setting the shape friction to 1 measurably changes push outcomes.
- **Block mass has exactly zero effect.** The agent is a `KINEMATIC` body (infinite mass) driven by a PD velocity update, so it cannot be slowed by anything. Block mass 0.1, 1, 10 and 100 give bit-identical trajectories. **Agent mass doesn't exist** until the agent is made dynamic.
- **`space.damping = 0` means maximum damping, not none.** In pymunk, damping is the fraction of velocity kept per second, so 0 zeroes the block's velocity every substep. The block is quasi-static: it moves only while being pushed and has no momentum. Damping of 0.5 to 1 gives it momentum (it slides). Even then, mass still has no effect, because damping scales velocity regardless of mass.
- **Center of gravity works.** It changes how the block rotates under a push. `body.position` stays the shape's geometric origin when the CoG moves, so the state layout and the goal-outline rendering are unaffected.
- **Making mass matter requires a dynamic agent.** With a dynamic agent of mass 1, a block of mass 1 vs 10 gives very different motion. The standard pymunk approach (used in its "tank" demo) is:
  - a kinematic *control body* that follows the PD target, joined to a dynamic agent by a `PivotJoint` with a `max_force` (the agent's strength);
  - floor friction on the block from a `PivotJoint` to the static body with `max_bias=0, max_force=μ·m·g`;
  - rotational floor friction from a `GearJoint` with its own `max_force`.

  This gives real Coulomb floor friction, independent angular friction, and meaningful agent and block masses.
- **There are two separate goal notions in swm.**
  - `goal_pose` (the variation's `goal.position` and `goal.angle`) is the **drawn green outline**.
  - `goal_state` is an agent+block state, either sampled randomly in `reset` or passed in as `options['goal_state']`. **`eval_state` checks against `goal_state`, and that check includes the agent's xy.**

  So swm's "success" is "match this goal image's state", not "block on the outline". The teleop recorder needs its own task-success check: block pose vs `goal_pose`, with shape symmetry, and no agent term.
- The `goal.scale` variation is unused: the outline is drawn from the block's own shapes.
- `DEFAULT_VARIATIONS` samples only the agent start, block start and block angle, so by default the goal stays fixed at (256, 256, π/4). Nothing prevents the agent and block from starting overlapping (there's a TODO in swm).
- `envs/utils.py` uses shapely only in `pymunk_to_shapely`, which Push-T never calls.
- Importing `stable_worldmodel.envs.pusht.env` is lightweight: the swm package import is lazy, and `envs/__init__` only registers ids. A parity test therefore needs only our core deps plus swm installed.

## Decisions made

1. **Simulator base:** fork swm's Push-T env (`../stable-worldmodel/stable_worldmodel/envs/pusht/env.py`, MIT), so our env matches swm's state layout, rendering and variation columns, and is a drop-in for swm's `World` and eval. Register it under our own gym id (planned: `pts/PushT-v0`).
2. **Recorder:** modeled on Diffusion Policy's `../diffusion_policy/demo_pusht.py` (MIT): record seeds in order, R = retry, Space = pause, auto-stop on success, 10 Hz. Additions:
   - **gamepad** input via `pygame.joystick`: the stick's [-1,1]² deflection *is* the env's relative action, with a dead zone and a precision-mode button. Mouse is a fallback.
   - **auto-advance** to the next seed, with about 0.5 s of "get ready" freeze; recording starts at the first input past the dead zone.
   - success must **hold for N steps**.
   - keys for retry / skip / undo-last / pause / quit.
   - **save each demo immediately** after it ends.
   - condition blocks (about 25 demos each) in randomized order, with the condition name shown on screen.
   - per-demo quality signals: duration, retries, push count.
   - an optional display rate faster than 10 Hz (recorded data is unchanged).
3. **Augmentation:** DINO-WM noisy replay (the levels above are the defaults) **plus SE(2) transforms**: rotate and shift the agent, block, goal and action vectors together. Mirroring is allowed only for shapes with a symmetry axis. Transforms can be composed with noise. **MimicGen is deferred.** Pushing is non-prehensile and the goal is fixed in the world rather than relative to the object, so MimicGen-style stitching amounts to building a push-primitive planner: a research project.
4. **Data rules:**
   - Raw demos store **state + exact actions passed to `step()`, never pixels**. Pixels are rendered at export.
   - **Every augmenter goes through the same simulate → filter → render stage**, so every saved episode was actually run in its own condition's physics.
   - Train/val splits are made **by `demo_id`, never by episode**, so noisy copies of one demo don't leak across splits.
5. **Repo:** `pusht-scenarios`, on GitHub at tahaismail424/pusht-scenarios, managed by uv. MIT license, keeping the swm and Diffusion Policy copyright notices on code copied from them.
6. Classical control or MPC with privileged state stays a **fallback** expert for conditions that are unplayable by hand. It would plug into the same augment and export path.
7. **Varied goals are supported** (user decision, 2026-10-24). Fixed-goal datasets remain possible as one condition.
8. **Target factors** (user, 2026-10-24): floor friction, block (and other) colors, block shape, initial positions of the goal and block, block mass, block center of mass, angular friction. **Agent mass: dropped** (user, 2026-10-24). Anything else reasonable is welcome too. Adding env functionality is fine, but default settings must keep swm parity.

## Open decisions (ask the user; don't assume)

1. Copy swm's `spaces.py` (leaning yes, for exact behavioral parity) or replace it with a lighter config class that still writes `variation.*` names.
2. (resolved 2026-10-25) Agent strength is implemented as knob #6; block mass matters only with it.
3. Use the measured DINO-WM noise levels as-is, or tune them.
4. The condition list (factors, values, train / held-out in-range / held-out out-of-range, demos per condition). **This comes from the doctoral-student collaborator.**
5. (resolved 2026-10-25) Export format: HDF5 in LeWM's layout; adaptive-wm reads .h5.
6. (resolved 2026-10-25) Raw-demo file format: one `.npz` per demo with JSON meta.

## Roadmap (each phase ends with a check that must pass)

| Phase | Work | Exit check |
|---|---|---|
| 0. Skeleton | deps, package layout, .gitignore, LICENSE, test setup | `uv run pytest` passes on an empty suite |
| 1. Forked env | vendor the swm PushT env + `spaces.py` + `DrawOptions`; register the gym id; make the hardcoded physics settable (block mass with inertia derived from it, agent–block and block–wall friction, linear and angular damping, block CoG, PD gains `k_p`/`k_v`, action scale, agent radius, optional constant force field); goal pose as a variation; wall-contact detection; all settings in `info` | **parity with swm**: same seed, variations and actions give identical states and frames to `swm/PushT-v1` under default settings; deterministic; on the cluster, replaying the 18k dataset reproduces its stored states |
| 1c. Sandbox (agreed 2026-10-24) | playable, non-recording pygame loop: `teleop/input.py` (gamepad + mouse → action) and `teleop/sandbox.py` (`render_mode='human'` loop at 10 Hz; keys to reset and to pick a shape or knob preset); script entry `pts-sandbox`. Built before the knobs so each knob gets a feel check; becomes the base of the Phase 3 recorder | headless scripted-input test passes; the user confirms the default env feels like Push-T on their Mac and gamepad |
| 2. Condition specs | one YAML per named condition (visual + physics + goal + start seeds + success rule, hold N, split tag); validation; defaults; a stable `env_id` hash; "suite" files for sessions and datasets | load → env → identical physics, every time; round-trip test |
| 3. Teleop recorder | the recorder described under Decisions, item 2 | replaying a saved demo reproduces its states exactly; a headless scripted-input run (`SDL_VIDEODRIVER=dummy`) completes end to end |
| 4. Augmenters | `augment(demo, condition, rng) -> candidates` (start state + goal + actions + metadata); `noisy`, `se2`, composition | a transformed wall-free demo, re-simulated, matches the transformed original; success vs σ roughly matches the table above |
| 5. Simulate/filter/render | run candidates; filter on wall contact (transforms), leaving the board, minimum block movement; keep failures, flagged; render 224 px frames in parallel (CPU Slurm job on the cluster) | throughput and filter-rate report on a pilot batch |
| 6. Export | swm-format dataset matching the columns and chunking above, plus `variation.*` (swm naming; `evaluate_from_dataset` restores these at eval time), `env_id`, `demo_id`, `aug_type`, `noise_std`, `success`; splits by `demo_id`; a manifest (condition specs, package version, git hash, counts); a LeWM eval config for the env | swm's dataset class loads it; a 1-epoch LeWM smoke train; a LeWM random-policy eval runs |
| 7. Pilot → collection | 2–3 conditions × about 20 demos through the whole pipeline; finalize conditions; record; CLI (`pts record / augment / export / view`); README | — |

The earlier estimate was about 2–3 weeks part-time before full collection. Recording throughput with auto-advance should be about 150–250 demos per hour.

## Planned layout

```
src/pusht_scenarios/
  env/        forked swm PushT + physics knobs (+ vendored spaces.py, DrawOptions)
  conditions/ spec schema, loader, env_id hashing, suites
  teleop/     pygame recorder (gamepad / mouse)
  augment/    noisy.py, se2.py, (mimicgen.py later, maybe)
  pipeline/   simulate → filter → render
  export/     swm-format writer, manifest, eval config
  tools/      replay viewer, per-condition stats, swm parity check
conditions/   YAML condition specs and suites (data, not code)
tests/
```

## Source repos (cloned in `..`)

- `../stable-worldmodel`: galilai-group, MIT, v0.1.1.
  - Push-T env: `stable_worldmodel/envs/pusht/env.py` (926 lines). The variation system is in `stable_worldmodel/spaces.py`, and `DrawOptions` is in `stable_worldmodel/envs/utils.py`. The gym id `swm/PushT-v1` is registered in `stable_worldmodel/envs/__init__.py`. Tests are in `tests/envs/test_pusht_env.py`.
  - `envs/pusht/expert_policy.py` contains **only `WeakPolicy`** (random actions near the block). There is no Push-T expert.
- `../diffusion_policy`: real-stanford, MIT.
  - `demo_pusht.py` (120 lines) is the original recorder.
  - The env is in `diffusion_policy/env/pusht/` and uses the old pymunk API, so don't copy env code from here.

## Gotchas

- **Keep the core package torch-free.** Teleop runs on the user's Mac laptop, which needs a display. swm's base install pulls in torch. Core deps are numpy, pymunk, pygame, gymnasium and opencv-python-headless (the env uses cv2), plus pyyaml later. `tests/test_imports.py` checks in a fresh interpreter that importing and using the core package doesn't load `stable_worldmodel`, `torch` or `shapely`.
- **swm is a dev-only dependency pinned by git commit** (`[tool.uv.sources]` in pyproject.toml, rev `21446f1`). It is never a path to the local clone: the user wants installs reproducible for others. PyPI `stable-worldmodel` 0.1.1 is **older than that commit**: it has a different default block scale, a different Z-shape geometry, and no symmetric-shape success check. Don't switch to PyPI without re-checking parity. swm's base install also needs `imageio` and `shapely` to import, so they're in the dev group too.
- When re-syncing env code with a newer swm, bump the git rev, then re-copy and re-run parity. Our `env.py` and `spaces.py` differ from upstream only in their import lines, and `get_in` lives in our `env/utils.py`. Keep that diff minimal.
- swm uses the **pymunk 7 API** (`space.on_collision(...)`), while Diffusion Policy uses the old `add_collision_handler`. Follow swm.
- **swm now defaults to Lance** (`World.collect(format='lance')`). HDF5 is behind the `format` extra (h5py, hdf5plugin). LeWM's existing Push-T file is HDF5. Check which format the LeWM/SMWM versions in use actually read before locking in the export format; their swm versions (0.0.6 vs 0.1.1) differ.
- Preserve swm's quirks for parity, and flag them rather than silently fixing them. For example, `add_tee` computes `inertia2` from `vertices1`, and mass is hardcoded to 1 in several helpers. Changing default behavior breaks frame and state parity with existing datasets.
- swm's PushT already takes a `damping` constructor argument. Build the physics knobs consistently with it.
- Grace cluster: the scratch file-count quota is 250k (most of it is the uv cache and venvs). Datasets as single files are fine. Disk space matters more than file count for pixel datasets.
- `zstd` won't read through HF-cache symlinks. Point it at the resolved blob path.

- **iCloud Desktop sync breaks the venv.** The repo lives under `~/Desktop`, which is iCloud-synced. iCloud sets the macOS `hidden` flag on files in `.venv` a few seconds after they're written, and Python 3.13 silently skips hidden `.pth` files. Editable installs then vanish: you get `ModuleNotFoundError: pusht_scenarios`, or `stable_worldmodel.envs` loads as an empty namespace package and `swm/` ids aren't found. The quick fix is `chflags -R nohidden .venv`. Real fixes are moving the repo out of `~/Desktop`, or keeping the venv in a `.nosync` path (e.g. `UV_PROJECT_ENVIRONMENT`). That decision is pending with the user.
- **Rendering draws shapes only: no joints, no contact points** (2026-10-25). pymunk's `debug_draw` draws joints and contact points by default.
  - Our knobs' joints (agent strength leash, floor-friction pivot) showed up as purple dots (RGB 142, 68, 173), which would leak hidden factors into training pixels.
  - Contacts showed up as red dots wherever the agent touched the block.
  - `env/utils.DrawOptions` now draws `DRAW_SHAPES` only. **swm does draw contact dots**, so swm-identical frames need `gym.make('pts/PushT-v0', render_contacts=True)`, which the parity test uses. Consequence: LeWM's original `pusht_expert_train.h5` frames contain red contact dots and ours don't, which is a small visual difference to remember when comparing against or fine-tuning from LeWM's Push-T checkpoints.
  - `tests/test_render.py` (mutation-checked) asserts:
    - hidden physics renders pixel-identical to default physics;
    - default frames never contain contact-dot pixels, while `render_contacts=True` frames do;
    - the HUD never reaches `render()` output.
  - Recordings store no pixels, so demos recorded before the fix are fine; only exports re-render.
- **pymunk contact friction is the *product* of the two shapes' frictions.** Setting friction on only the block (with the agent at 0) is a no-op. Any friction knob must set both shapes.
- **SDL can't see the user's 8BitDo Ultimate 2C** (2dc8:301b, Bluetooth LE) on this macOS. The count is 0 in pygame (SDL 2.28) and pygame-ce (SDL 2.32), with or without a window, and with every SDL_JOYSTICK_* hint. The GameController framework also sees 0. IOKit and `hidapi` do see it: it's a standard HID gamepad, and no Input Monitoring permission is needed. The fix (user decision, 2026-10-24) is `teleop/hid_gamepad.py`: `HIDGamepad` decodes reports from the device's own HID descriptor. `InputDevice` tries SDL first, then hidapi (`use_hid=True`), then the mouse. Tests pass `use_hid=False` so a real connected pad doesn't change their results. The real descriptor is the fixture in `tests/test_hid_gamepad.py`. Verified live: the stick reads correctly and rests at about -0.004.
  - The controller can also connect in its **Switch mode**, which SDL sees natively as "Nintendo Switch Pro Controller" (6 axes). The SDL path is then used, at the cost of about 3 s of SDL handshake at startup.
  - Tests are hermetic: `tests/conftest.py` hides real pads (patches `pygame.joystick.get_count` and `open_first_gamepad`) and sets `SDL_JOYSTICK_HIDAPI=0` and `SDL_JOYSTICK_MFI=0`. Without that, each human-mode env init cost a roughly 2 s handshake (38 s suite instead of 7 s).
- opencv-python-headless 5 on macOS bundles its own `libSDL2`, alongside pygame's (objc "implemented in both" warning). This is harmless for `rgb_array` rendering. If the Phase 3 teleop window or gamepad misbehaves, suspect this first. Don't swap `cv2.resize` for a pygame resize to work around it: that changes pixels and breaks parity.

## Status / how to run

- Done:
  - Phase 0: skeleton, dependencies, pytest.
  - Phase 1a: copied env, registered as `pts/PushT-v0` in `src/pusht_scenarios/env/__init__.py`.
  - Parity suite (2026-10-24): `uv run pytest -q -p no:warnings`, 22 tests, about 12 s.
    - `tests/test_parity.py` checks bit-identical obs, info, reward, variation values and rendered frames vs `swm/PushT-v1`, over 8 seeds × {default, `variation: ['all']`} × 150 steps. The scripted actions steer at the block, and the test asserts that contacts happened.
    - Mutation-checked: contact friction 0 → 0.25 fails the test at step 4.
  - Phase 1c, in progress: `teleop/input.py` `InputDevice` is done and tested (`tests/test_input.py`, with a fake joystick and mouse).
    - `read(env)` returns a **relative action** in [-1, 1]², never a target position. The env's relative mode adds `action × 100 px` to the agent position itself.
    - Stick: radial dead zone, rescaled; full tilt gives `gain`; optional precision button.
    - Mouse: `(cursor - agent) / 100`, clipped. Window pixels equal world coordinates (512, y down), so there's no flipping.
  - `teleop/sandbox.py` is done and tested headless (`tests/test_sandbox.py`, with scripted events and a fake stick). Run it with `uv run pts-sandbox [-c configs/sandbox.yaml]`.
    - The YAML has session settings plus `conditions`: each has a `name`, an optional `variation` (keys resampled per seed; omitted means swm's defaults) and `variation_values` (dotted keys). Values are cast to each space's dtype.
    - Round i of every condition uses seed `seed + i`.
    - Keys: R retry, N skip, Space pause, Q/Esc quit, **P queues a pause before the next round** (shown as "PAUSE QUEUED"; P again unqueues it; on the pause screen P continues and Q quits). The round starts at the first input: stick out of the dead zone, or the mouse within 30 px of the agent.
    - Success comes from `env/task.py` `block_on_goal`: block vs `goal_pose`, symmetry-aware, swm's 20 px / 20° tolerances, held `hold_steps`. A block at `goal_pose` fully covers the outline (checked by pixel count).
    - Precision mode (user, 2026-10-24): `precision_button` defaults to `'rightshoulder'`, given as an SDL GameController name or a raw index.
      - SDL path: names come from SDL's controller database (`sdl_button_names`, via `pygame._sdl2.controller`). Verified on the real pad in Switch mode: `rightshoulder` is button 10.
      - hidapi path: names come from `ANDROID_HID_LAYOUT` in `hid_gamepad.py`, where right shoulder is index 7. **That layout is a guess and unverified on the 8BitDo**; check with `uv run pts-buttons` in its non-Switch mode.
      - An unresolvable name disables precision with a startup message instead of crashing.
    - Full suite: 227 tests, about 21 s (with augmentation).
    - Stick gain default is 0.6 (user, 2026-10-24); 0.3 felt slow next to the mouse. Agent speed is linear in the action at about 410 px/s per action unit: 0.3 gives about 120 px/s, 0.6 about 245 px/s, and 1.0 (the mouse's max) about 410 px/s. Not yet compared against the original dataset's pace (which needs the cluster file).
  - The user played with the mouse and it "feels good". The gamepad works through the hidapi fallback. Still to report: whether stick direction, gain and the precision button feel right.
- 2026-10-25: the user recorded `data/raw/20261025-014136-demo` (75 demos, all successes, no undos). The augmentation sample is in `data/aug/sample`.
- Next (2026-10-25): finalize the v0 design with the collaborator. Then record `configs/generated/v0/record.yaml`, augment (`configs/augment/v0.yaml`), curate, and export (`configs/datasets/v0.yaml`). Then adaptive-wm integration: the env latent (encoder vs predictor is open) and the eval harness.

## Keeping the agent on the board (2026-10-25)

`env/bounds.py` `KeepAgentOnBoard` wrapper. It bounds the **action** so the PD target `agent + action·100` stays inside the walls (inner faces at 7 and 504, minus the agent radius). The PD gains are critically damped (k_v = 2√k_p), so the agent never overshoots its target, and this works under any physics. The env itself stays identical to swm. `info['action']` is the applied, bounded action: **record that one**. Bounded actions also keep the agent on the board when replayed in plain swm (tested). The sandbox uses the wrapper, and the recorder and augmenters must too.

## Recording (implemented 2026-10-25)

Run with `uv run pts-record -c configs/demo.yaml`, then watch with `uv run pts-replay data/raw/<session>`. User decisions (2026-10-25):
- raw format: `.npz`;
- conditions inline in the record config;
- `shuffle_conditions` flag;
- `save_failures` defaults to true.

- **Config:** one YAML drives both sandbox and recorder. The sandbox ignores the `output:` block (`sandbox.RECORDER_KEYS`). `configs/demo.yaml` is the showcase suite: 15 conditions, each changing one factor (heavy, slippery, sticky_floor, no_spin, grippy, two CoG shifts, colors, L and + shapes, two fixed goals, random goal, combo), 5 rounds each, shuffled.
- **Hooks:** `teleop/sandbox.py` `RoundHooks` (`on_round_start` at first input, `on_step`, `on_round_end`, `on_event`, `status`). `teleop/record.py` `DemoRecorder` implements them.
  - Success, and timeouts if `save_failures`, are saved immediately. Retry, skip and quit are logged and discarded; retries bump `meta.attempt`. U undoes the last saved demo.
  - With shuffle and no seed, a random seed is pinned, and the order is recorded in the session snapshot.
- **`io/demo.py`:** `Demo`/`DemoMeta` and `StepBuffer`.
  - Arrays: `state` (T+1,7), `sim_state` (T+1,14 = `SIM_STATE_FIELDS`), applied `action` (bounded), `raw_action`, `on_goal`, `t_wall`.
  - The meta holds the flattened `variation` and **`reset_options`**.
  - `.npz` with JSON meta (no pickle), atomic write, `format_version` checked on load.
- **`io/session.py`:** `<root>/<YYYYmmdd-HHMMSS>-<name>/` containing `session.yaml`, `events.jsonl` and `demos/NNNN_<cond>_seed<k>_<id8>.npz`. `rejected_ids()` reads undo events.
- **`env/variations.py`:** `flatten` / `cast_values` / `exact_reset_options`.
- **`gen/replay.py`:** `replay(demo, actions=None, start=0)` returns a `Rollout`. `reset_to` raises `ReplayMismatch` if the rebuilt variation differs from the recording. `divergence()`. Replay steps a **plain** env with the stored, already bounded actions.
- **`teleop/viewer.py` (`pts-replay`):**
  - Keys: Space, Left/Right (or N/P), R, +/−, Q.
  - The HUD shows the max stored-vs-replayed error, which should be 0. Undone demos are labelled REJECTED.
- **Exactness findings (tested):**
  1. **Replay needs the same `reset()` call as the recording.** swm's reset teleports to a randomly drawn goal state and steps the physics. That draw depends on how much randomness the options consumed, and pymunk carries joint/contact impulses between steps. So setting every value explicitly gave a different start state with a physical agent (measured 0.5 px on `mix` seed 1). Hence `meta.reset_options`. With it, every condition replays bit-exactly.
  2. **Restoring a mid-episode state is exact, but continuing from it is not a replay.** Warm-started contact impulses aren't stored, and divergence reached hundreds of px. **Augmenters must simulate from reset (step 0).** End-trims are safe; start-trims must be verified via the parent.
  3. Set a block's angle **before** its position: pymunk rotates about the CoG, which isn't the body origin.
- Still stubs: `io/collection.py` (index, merge, trim, reject) and `io/cli.py` (`pts-data`). The user deferred those.

## Augmentation (implemented 2026-10-25)

`uv run pts-augment -c configs/augment/sample.yaml` writes `data/aug/sample` (about 260 demos, for eyeballing). `configs/augment/full.yaml` writes `demo_v1`: about 30k demos, 404 per parent. Watch with `uv run pts-replay data/aug/<name>`; labels show the base and noise level.

User decisions (2026-10-25):
- copy DINO-WM;
- transforms on for all conditions (curation can exclude later);
- place start poses directly with a geometric validity check;
- **include mirroring**;
- explicit palettes, not random colors.

- **`gen/augment.py`:** augmenters only *propose* a `Candidate` (seed, actions, full variation, start state, chain). `BOARD_CENTER = (255.5, 255.5)`, since the walls run 5..506.
  - **Untransformed candidates keep the parent's exact reset call,** so a σ=0 copy is bit-identical to its parent. Transforms switch to explicit mode: every variation value set, plus `reset(options={'state': ...})`.
    - In explicit mode `variation.*start_position` keeps the parent's values; **`state[0]` is the true start.**
  - `noise(c, sigma, rng)`: i.i.d. Gaussian, clipped. `DINO_WM_NOISE_LEVELS` gives 101 copies.
  - `se2(c, theta, shift)`: rotates about the center, then shifts. Transforms agent, block, goal and actions.
  - `mirror(c, axis_deg, local_axis_deg)`: the new block angle comes from `R(a') = M R(a) S`, and the CoG offset maps to `S·offset`.
  - `local_mirror_axes(env)`: tests the real shapes on a point grid and requires the shape's own CoG to lie on the axis. Results: T and small_tee [90]; I [0, 90]; square and + [0, 45, 90, 135]; **swm's L and Z: none**, so mirror is skipped for them. swm's "L" is two offset bars.
  - `recolor(c, palette, name)`: keys from `COLOR_KEYS`.
- **`gen/simulate.py`:** `simulate(env, candidate)` returns a Demo.
  - Always from reset. Actions are bounded, and the proposed ones are stored as `raw_action`.
  - Full length, outcome judged at the final step (success/failure), failures kept.
  - Deterministic `demo_id` via uuid5 of (parent, chain, copy).
  - `derived_from` holds parent, recipe, chain, copy, base and metrics (`ever_on_goal`, final errors, `min_wall_margin`).
  - **Validity (`start_problems`) is checked only for transformed candidates:** block and goal outline inside the walls (margin ≥ 0), agent inside the walls, agent not overlapping the block, values within the variation ranges.
  - Untransformed copies skip it, because swm's own starts graze walls at 0.4 px: 12 of the 75 recorded parents, all seed 0, plus one `random_goal` whose outline crosses a wall by 1.4 px and was still solved. pymunk's `shapes_collide` asserts when shapes don't touch, so use `space.shape_query`.
- **`gen/recipe.py`:** a `Recipe` YAML has sources, filters (`conditions`, `parents_per_condition`, `include_rejected`, `include_failures`), bases (`original`, `rotations_deg`, `se2{count, rotation_deg, shift_px, max_tries}`, `mirror_axes_deg`), palettes (`apply_to: original|all`) and noise (`dino_wm` or `[[sigma, copies]]`).
  - Output: `<root>/<name>/` with `recipe.yaml`, `summary.yaml` (per-condition copies, outcomes, skipped with reasons) and `demos/`.
  - Per-parent RNG comes from (seed, demo_id), so the output is independent of worker count (tested). A `ProcessPoolExecutor` parallelizes, and derived demos are never re-augmented.
- **Measured on the 75 recorded demos:**
  - noise success: σ 0.003 → 83%, 0.01 → 73%, 0.03 → 38%, 0.07 → 12%, 0.15 → 4%, the same shape as DINO-WM's original data;
  - noise-free bases: original 15/15, recolor 30/30, mirror 12/12, rot180 13/14, random se2 9/15;
  - throughput about 26 simulations/s per core.
- Tests: `tests/test_augment.py` (53).
  - rigid-motion geometry of block, goal and actions;
  - mirror point-set reflection (mutation-checked);
  - early-step equivariance of the dynamics;
  - every validity reason;
  - σ=0 copy equals the parent; recolor leaves physics identical;
  - all outputs replay exactly;
  - recipe parsing, filters, counts, cross-worker determinism.

## Suites, curation, eval hooks, export (implemented 2026-10-25)

User decisions (2026-10-25):
- **Factorial training envs plus single-factor test envs.**
- **Holdout middle vs extreme is a knob**, not a fixed choice.
- **One agent strength shared by all v0 envs,** with mass varied on top.
- Shapes are my call.

- **`suite.py` (`pts-suite show|build`):** a YAML of factors → levels (variation values) plus `base`.
  - **Holdout knob:** a per-factor `holdout` (a level name or low/middle/high for `ordered` factors; categorical factors need a name), or a suite-wide `holdout_policy`. CLI: `--holdout mass=high` or `--holdout high`. A suite-wide policy only touches ordered factors.
  - Train envs: the factorial over train levels.
  - Test envs: one factor held out, the others at `reference` (`test_context: reference`) or at every train combination (`all`). Optional `include_all_heldout`.
  - `env_id`: train envs first (0..N−1, the latent-embedding index), then test envs. `env_hash` is a content hash.
  - `build` writes `suite.yaml` and `record.yaml`. Each condition gets `rounds`, `seed` (test envs default to 1000, so they get their own layouts) and `info`, which the recorder stores as `meta.condition_info`.
- **`configs/suites/v0.yaml`:**
  - mass {light 0.5, normal 1, heavy 3} (holds out normal: interpolation);
  - friction {low = damping 1 + floor 0.4, high = damping 1 + floor 2, stop = swm} (holds out high);
  - color {default, warm, night} (holds out night);
  - shape {T, I, + (2, 5, 7)} (holds out +). "X" is just a rotated +, and swm's L isn't a true L.
  - Base `sim.agent_max_force` 10000. 16 train + 4 test envs, 10/12 rounds each = 208 demos. Built into `configs/generated/v0/`.
  - **Finding:** mass is clearly visible under `stop` (pushes light 138 / normal 86 / heavy 31 px) but nearly invisible under `low` friction (135 vs 131): a sliding block needs little force once moving. Lower strength doesn't fix it (and makes heavy+high immovable). That's a real physics interaction, flagged to the user.
- **Curation, `io/collection.py` + `pts-data ls|reject|unreject|trim|merge|verify`:**
  - `DemoIndex.scan` reads metadata only (`load_meta`).
  - Rejects come from `<collection>/curation.yaml` plus session undo events, and **augmented copies inherit their parent's rejection**.
  - `merge` copies kept demos (idempotent, writes `collection.yaml`). `trim` does end-trims only; start-trims are refused (not exact).
  - **End-trims count as recordings** (`is_recorded`): they replace their source, so they're split by their own id, are augmentable, and never inherit a reject.
  - `pts-replay` X key toggles a reject.
- **Eval hooks:**
  - `PushT.reset_options_from_dataset(init_row, goal_row)` rebuilds an episode from the `variation.*` columns, skipping `START_POSE_KEYS` (the `state` column is the true start). It also takes `goal_state` from the goal row.
  - `pts/PushTBounded-v0` = `KeepAgentOnBoard(PushT)`, for planners.
  - **The wrapper's info key is `applied_action`:** swm's wrappers assert they own `info['action']`.
  - Tested: swm `World.evaluate(dataset=...)` with RandomPolicy restores mass, color, goal and seeds.
- **Export, `export/hdf5.py` (`pts-export -c configs/datasets/v0.yaml`):** writes `<name>_{train,val,test_adapt,test_eval}.h5`, `.episodes.jsonl` and `<name>.manifest.yaml`.
  - Splits are by parent. Test envs get only clean successful recordings: the first `adapt_per_env` by seed go to adapt, the rest to eval.
  - Rows: T+1 per episode, with a NaN last action (swm's collect convention).
  - Pixels are replayed from reset, and an inexact replay raises. Blosc-LZ4 chunks of 100.
  - Extra columns: seed, env_id, demo_idx, parent_idx, is_original, clean, success, on_goal, noise_std, base_idx, and `variation.*` with dtypes taken from the variation space.
  - Loads in swm `HDF5Dataset`.
- `configs/augment/v0.yaml`: train envs only (`suite_splits`), 3 bases × 25 noise copies, giving about 12k episodes.
- **CLI exit codes:** every module has `run(argv)` (returns results, used by tests) and `main()` (returns None). The entry points do `sys.exit(main())`, so returning an object used to print it and exit 1.
- Tests: `test_suite.py`, `test_collection.py`, `test_eval_hooks.py`, `test_export.py`, and the session-scoped `mini` fixture in `conftest.py` (mini suite → recordings → augmentation). 291 total, about 26 s.

## Facts for packaging and eval (checked 2026-10-25)

- **swm dataset-driven eval** (`World.evaluate(dataset=..., episodes_idx, start_steps, goal_offset, eval_budget)`, in `world/world.py` at the pinned rev):
  - It loads `[start, start+goal_offset]` of held-out dataset episodes. The goal is the last frame of that window: `goal` for pixels, `goal_<col>` for the other columns.
  - It uses a per-episode `seed` column as the reset seed if present.
  - **If the env has `reset_options_from_dataset(init_row, goal_row)`, swm calls it to build each env's reset options.** No swm env implements it yet, so our PushT can add it, restoring `variation.*`, `state` and `goal_state` without callables.
  - Otherwise it falls back to the legacy callables.
  - Success = `terminated`, from the env's `eval_state(goal_state, state)`, which **includes the agent's xy** (an image-goal match, not "block on the outline").
- **User's fork `github.com/tahaismail424/adaptive-wm`** (LeWM fork; holds the modeling and eval code) loads **HDF5** (`$STABLEWM_HOME/<name>.h5`), with Hydra configs under `config/train/data/` and `config/eval/`. It pins no swm version. So the export format is HDF5 in LeWM's layout.
- **T2-z (arXiv 2610.22364):**
  - z: a 32-D latent per environment, U(0,1) init, clipped to [0,1]. It's injected as an extra conditioning token plus an added modulation vector.
  - Training: 300 model steps, then alternating 200 latent / 200 model steps. Latent LR 0.03, model LR 1e-5.
  - Test-time adaptation: frozen model, a new z, 40 steps with LR 3.0/1.5/0.5/0.15 (10 steps each), on **1–2 trajectories** from the new environment.
  - Scale: **many environments with few episodes each** (12–84 envs × 9–50 episodes).
  - Baselines: no latent, LoRA test-time adaptation, a DINOv2+GRU context encoder, TTT. The paper doesn't describe its held-out protocol.
- **swm shape names:** `'+'` is a cross, so an "X" is just a `'+'` at 45°. Block angle is part of the state, so X adds nothing over `'+'`. swm's `'L'` is built from two offset bars and has no mirror axis.

## Phase 1b plan: physics knobs

Where the knobs live: `PushT.__init__` in `src/pusht_scenarios/env/env.py` builds `self.variation_space`, a nested `swm_spaces.Dict` whose top-level groups are `agent`, `block`, `goal`, `background` and `rendering`. The new top-level group is named **`sim`**. Two things keep swm's seeded draws unchanged: it is last in `sampling_order`, and its name sorts alphabetically after `rendering`. gymnasium hands each group a seed in sorted key order, so a key named `physics` shifted `rendering.render_goal`'s draw (user chose the rename, 2026-10-24). Any future top-level group must also sort after `rendering`. `_setup()` (and the shape builders) read the values and apply them to the pymunk bodies, shapes and joints at every `reset`.

Living in the variation space means each knob can be randomized (`options={'variation': [...]}`), set explicitly (`options={'variation_values': {...}}`), and exported as a `variation.sim.*` column, which swm's eval reads back to rebuild the env.

Procedure per knob:
1. Add it with a default that reproduces swm, so parity stays green.
2. Add an effect test showing that turning it on changes the dynamics in the expected direction.
3. Playtest it in the sandbox to find playable ranges.

Before the first knob, change the parity test's "all variations" case to request swm's variation names explicitly, since swm has no `sim` group.

| # | Knob | Implementation | Default (= swm) | Status |
|---|---|---|---|---|
| 0 | `sim` group scaffold (all knobs below declared, none wired) + parity-test tweak | — | — | done |
| 1 | Contact friction: agent–block, block–wall | agent = walls = 1, block = μ (pymunk multiplies them) | 0 | **done** (effect is modest: a glancing push turns about 10° more at μ=1) |
| 2 | Block center of mass | `center_of_gravity += offset`; **keeps swm's moment of inertia** (only the pivot moves). The `block_cog` constructor argument still works and is applied first | the shape's own CoG | **done** (strong: ±25 px along the bar turns a straight stem push by ±61°) |
| 3 | Block mass | `mass = m`, `moment *= m` | 1 | **done, but no visible effect until #6**: it exactly cancels with an unlimited-strength agent. `test_mass_has_no_effect_with_unlimited_agent` pins this; flip it when #6 lands |
| 4 | Floor (sliding) friction | `PivotJoint(static_body, block, (0,0), cog)` with `max_bias=0`, `max_force=μ·m·FLOOR_G` (`FLOOR_G=1000` px/s²). **Also brakes spin** through the gear joint below: a single friction point at the CoG can't resist rotation, and without it the block spun forever | off | **done** |
| 5 | Angular friction | `GearJoint(static_body, block)` with `max_bias=0` and a torque cap of `(μ_floor + μ_rot)·m·FLOOR_G·block.scale`; angular friction is extra spin resistance on top of the floor's | off | **done** |
| 6 | Agent strength (`sim.agent_max_force`) | Agent → `DYNAMIC` (mass `AGENT_MASS = 1`, infinite moment, ignores space damping via `_undamped_velocity`). The PD controller drives `self.agent_driver`, an invisible kinematic control body, joined to the agent by a `PivotJoint` with `max_force`. `step()` and `_set_state()` use `agent_driver`, which is the agent itself at default | 0 = off (kinematic agent) | **done**. At damping 0, max push speed is about `max_force·dt/m`. Measured over an 8-step push with strength 20000: block masses 1, 3 and 10 move 138, 59 and 19 px (kinematic: 142). Free-space cruise speed is unchanged at all strengths. With floor friction there is a Coulomb threshold: μ·m·g > strength means the block won't move |
| 0b | `sim.damping` | `space.damping` if ≠ 0. Independent of floor friction (no automatic coupling; user-facing configs pair them explicitly). **Floor and angular friction barely matter at damping 0**: the block has no momentum to brake. Use damping 1 for the friction conditions | 0 | **done** |
| — | Agent mass | — | — | dropped (user) |

Implementation notes (2026-10-25):
- All of this is in `PushT._apply_sim()`, called in `reset()` right after `_setup()` and swm's `block_cog`/`damping` overrides. `_setup()` stays byte-identical to upstream.
- `_set_state()` now zeroes the block's velocity and spin, since a set state is at rest. That's a no-op at damping 0, so parity holds.
- Measured: with damping 1, the block slides 66 px after release; floor friction 0.5 cuts that to 44 px and 2.0 to 14 px. Angular friction 2 cuts the turn from 78° to 7°.
- Tests: `tests/test_physics.py`. `configs/physics_check.yaml` holds distinguishable sandbox conditions for the user's feel check, and every `configs/*.yaml` is load-tested.
- **swm quirk, relevant to teleop and recording:** the default kinematic agent passes straight through walls and can leave the board (measured x = 1203 on the 512 board). A dynamic agent (strength > 0) stops at walls. Decide whether the recorder should clip or forbid this under default physics.
- Open: record block velocity and spin per step at export (the 7-number state lacks them once there's momentum).

Visual and geometric factors (colors, shape, scale, goal pose, start poses) already exist in the variation space. They need no env work, apart from the teleop success check (block vs `goal_pose`, with shape symmetry).

## Working style

- The user wants to work **deliberately and step by step**: research and explain options, agree on a decision, then implement. Don't race ahead into building later phases.
- Use uv for everything: `uv add`, `uv run`, `uv run pytest`. Python 3.13.
- Never launch large cluster jobs. Write the sbatch file or command and hand it to the user.
