# Re-simulate a recorded action sequence in its own condition.
#
# Used to verify recordings (replaying from the start reproduces the stored
# states exactly) and, later, as the shared "simulate" stage every augmenter
# goes through (CLAUDE.md decision 4).
#
# Exactness needs the *same reset call* as the recording: same seed and the
# same reset options (meta.reset_options). swm's reset briefly teleports the
# bodies to a randomly drawn goal state and steps the physics; that draw depends
# on how much randomness the options consumed, and pymunk carries joint/contact
# impulses between steps, so a different draw leaves a different start state
# (seen with a physical agent). Setting every value explicitly is NOT the same.
#
# Stored actions are the bounded ones actually applied, so replay steps a
# plain env: re-bounding could change the last float bit and break exactness.
# Augmenters producing *new* actions must bound them (env/bounds.py) and store
# the applied ones, exactly like the recorder.
from dataclasses import dataclass

import gymnasium as gym
import numpy as np

import pusht_scenarios.env  # noqa: F401  (registers pts/PushT-v0)
from pusht_scenarios.env import task, variations
from pusht_scenarios.io.demo import (
    Demo, capture_sim_state, current_state, restore_sim_state,
)


@dataclass
class Rollout:
    state: np.ndarray  # (T+1, 7)
    sim_state: np.ndarray  # (T+1, K)
    on_goal: np.ndarray  # (T,)


def make_env(demo: Demo, render_mode=None):
    return gym.make(demo.meta.env_id, render_mode=render_mode)


def demo_reset_options(env, demo: Demo) -> dict:
    """The reset options the demo was recorded with (cast for the env). Falls
    back to setting every stored variation value if none were recorded."""
    opts = demo.meta.reset_options
    if not opts:
        return variations.exact_reset_options(env, demo.meta.variation)
    out = {'variation_values': variations.cast_values(env, opts.get('variation_values') or {})}
    if opts.get('variation') is not None:
        out['variation'] = list(opts['variation'])
    if 'state' in opts:  # derived demos place their start pose directly
        out['state'] = np.asarray(opts['state'], dtype=np.float64)
    return out


class ReplayMismatch(RuntimeError):
    pass


def reset_to(env, demo: Demo, start: int = 0) -> None:
    """Rebuild the demo's env with its seed and reset options, check the
    variation values match the recording; for start > 0 also restore the full
    physical state at that step."""
    env.reset(seed=demo.meta.seed, options=demo_reset_options(env, demo))
    rebuilt = variations.flatten(env)
    if rebuilt != demo.meta.variation:
        diff = sorted(k for k in rebuilt if rebuilt[k] != demo.meta.variation.get(k))
        raise ReplayMismatch(f'variation differs from recording: {diff}')
    if start:
        restore_sim_state(env, demo.sim_state[start])


def iter_replay(env, demo: Demo, actions=None, start: int = 0):
    """Reset, then yield (t, action, info) after each step of `actions`
    (default: the demo's own from `start`)."""
    reset_to(env, demo, start)
    actions = demo.action[start:] if actions is None else np.asarray(actions, np.float32)
    for t, action in enumerate(actions, start=start):
        *_, info = env.step(action)
        yield t, action, info


def replay(demo: Demo, actions=None, start: int = 0, env=None, pos_tol=20.0, angle_tol_deg=20.0) -> Rollout:
    """Re-simulate and return the rollout's state, sim_state, on_goal arrays."""
    own = env is None
    env = env or make_env(demo)
    try:
        reset_to(env, demo, start)
        states, sims, goals = [current_state(env)], [capture_sim_state(env)], []
        acts = demo.action[start:] if actions is None else np.asarray(actions, np.float32)
        for action in acts:
            env.step(action)
            states.append(current_state(env))
            sims.append(capture_sim_state(env))
            goals.append(task.block_on_goal(env, pos_tol, np.deg2rad(angle_tol_deg)))
        return Rollout(np.array(states), np.array(sims), np.array(goals, dtype=bool))
    finally:
        if own:
            env.close()


def divergence(demo: Demo, rollout: Rollout, start: int = 0) -> float:
    """Max abs difference between stored and replayed swm states."""
    stored = demo.state[start:start + len(rollout.state)]
    return float(np.max(np.abs(stored - rollout.state)))
