# The one shared simulate stage: Candidate -> Demo (CLAUDE.md decision 4).
#
# Always from reset (step 0): a rollout continued from a restored mid-episode
# state is not a reliable replay (io/demo.restore_sim_state). Actions are
# bounded like the recorder's (env/bounds.py) and the applied ones stored, so
# the result replays exactly with gen/replay.py.
#
# DINO-WM protocol: run the full action sequence, no early stop; the outcome is
# judged at the final step; failures are kept (outcome='failure') for coverage.
import uuid

import numpy as np

from pusht_scenarios.env import task, variations
from pusht_scenarios.env.bounds import WALL_INNER, agent_radius, bound_action
from pusht_scenarios.gen.augment import Candidate
from pusht_scenarios.io.demo import Demo, DemoMeta, StepBuffer
from pusht_scenarios.io.session import now_iso

# deterministic demo ids: same parent + chain + copy index -> same id
ID_NAMESPACE = uuid.UUID('6f1c2a0e-4b7d-4c1e-9a52-2d3f6a8b9c10')


def reset_candidate(env, c: Candidate) -> None:
    opts = c.reset_options()
    options = {'variation_values': variations.cast_values(env, opts.get('variation_values') or {})}
    if opts.get('variation') is not None:
        options['variation'] = list(opts['variation'])
    if 'state' in opts:
        options['state'] = np.asarray(opts['state'], dtype=np.float64)
    env.reset(seed=c.seed, options=options)


def _shape_points(shape, pose=None):
    """World points bounding a shape (polygon vertices or circle extremes),
    optionally for its body placed at pose = (x, y, angle)."""
    body = shape.body
    if pose is None:
        to_world = body.local_to_world
    else:
        R = np.array([[np.cos(pose[2]), -np.sin(pose[2])], [np.sin(pose[2]), np.cos(pose[2])]])
        def to_world(p):  # noqa: E306
            return R @ np.asarray(p, float) + np.asarray(pose[:2], float)
    if hasattr(shape, 'get_vertices'):
        pts = np.array([to_world(v) for v in shape.get_vertices()])
        return pts, shape.radius
    center = np.asarray(to_world(shape.offset), float)
    return center[None, :], shape.radius


def wall_margin(shapes, pose=None) -> float:
    """Smallest distance from the shapes to the walls' inner faces (<0: through)."""
    lo, hi = WALL_INNER
    margin = np.inf
    for shape in shapes:
        pts, r = _shape_points(shape, pose)
        margin = min(margin, float(np.min(pts - lo)) - r, float(np.min(hi - pts)) - r)
    return margin


def start_problems(env, min_margin: float = 0.0) -> list[str]:
    """Why a transformed start is not a valid episode; [] if fine.

    min_margin 0: nothing may start inside a wall. (swm's own start
    distribution puts blocks as close as 0.4 px, so don't demand more.)"""
    base = env.unwrapped
    problems = []
    if wall_margin(base.block.shapes) < min_margin:
        problems.append('block_outside_walls')
    lo, hi = WALL_INNER
    r = agent_radius(base)
    if not all(lo + r <= x <= hi - r for x in base.agent.position):
        problems.append('agent_outside_walls')
    # (pymunk's Shape.shapes_collide asserts when shapes don't touch; use a query)
    block_shapes = set(base.block.shapes)
    if any(hit.shape in block_shapes for a in base.agent.shapes for hit in base.space.shape_query(a)):
        problems.append('agent_overlaps_block')
    if wall_margin(base.block.shapes, pose=base.goal_pose) < min_margin:
        problems.append('goal_outside_walls')
    return problems


class InvalidCandidate(ValueError):
    def __init__(self, problems):
        super().__init__(', '.join(problems))
        self.problems = problems


def check_candidate(env, c: Candidate) -> list[str]:
    """Reset to the candidate and report start problems (out-of-range values too).

    Only transformed (explicit) candidates are checked: an untransformed copy
    starts exactly where its parent did, which was played, so it's valid by
    construction (some swm starts/goals graze a wall and are still solvable)."""
    try:
        reset_candidate(env, c)
    except (AssertionError, ValueError):
        return ['variation_out_of_range']
    return start_problems(env) if c.explicit else []


def simulate(env, c: Candidate, copy_index: int = 0, recipe: str = '',
             pos_tol=20.0, angle_tol_deg=20.0) -> Demo:
    """Run the candidate from reset; raise InvalidCandidate if its start is invalid."""
    problems = check_candidate(env, c)
    if problems:
        raise InvalidCandidate(problems)

    base = env.unwrapped
    angle_tol = np.deg2rad(angle_tol_deg)
    buf = StepBuffer()
    buf.start(env)
    margin = wall_margin(base.block.shapes)
    for t, proposed in enumerate(c.actions):
        applied = bound_action(base, proposed)
        env.step(applied)
        on_goal = task.block_on_goal(env, pos_tol, angle_tol)
        buf.add(env, proposed, {'applied_action': applied}, on_goal, c.parent.t_wall[t])
        margin = min(margin, wall_margin(base.block.shapes))

    parent = c.parent.meta
    pos_err, angle_err = task.block_goal_error(env)
    chain_key = repr([parent.demo_id, c.chain, copy_index])
    meta = DemoMeta(
        demo_id=uuid.uuid5(ID_NAMESPACE, chain_key).hex,
        session_id=recipe or parent.session_id,
        condition=parent.condition,
        seed=c.seed,
        variation={},  # filled from the env by StepBuffer.to_demo
        outcome='success' if buf.on_goal and buf.on_goal[-1] else 'failure',
        steps=0,
        seconds=parent.seconds,
        attempt=parent.attempt,
        device=parent.device,
        env_id=parent.env_id,
        operator=parent.operator,
        created_at=now_iso(),
        package_version=parent.package_version,
        git_commit=parent.git_commit,
        reset_options=c.reset_options(),
        derived_from={
            'parent': parent.demo_id,
            'parent_session': parent.session_id,
            'recipe': recipe,
            'chain': c.chain,
            'copy': copy_index,
            'metrics': {
                'ever_on_goal': bool(any(buf.on_goal)),
                'final_pos_err': round(pos_err, 3),
                'final_angle_err_deg': round(float(np.degrees(angle_err)), 3),
                'min_wall_margin': round(margin, 3),
            },
        },
    )
    return buf.to_demo(meta)
