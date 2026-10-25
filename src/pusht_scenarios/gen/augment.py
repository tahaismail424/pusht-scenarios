# Augmenters: propose new episodes from a recorded demo. They never simulate.
#
# Every augmenter maps a Candidate to a Candidate: (reset options, actions,
# provenance chain). gen/simulate.py runs each candidate from reset in its own
# physics, bounds the actions, and produces an ordinary Demo (CLAUDE.md
# decision 4), so recorded and augmented demos replay/export the same way.
#
#   noise    DINO-WM-style noisy replay: i.i.d. Gaussian noise on the actions,
#            same length, open loop.
#   se2      rotate (about the board center) and shift the whole episode:
#            agent, block, goal and action vectors together.
#   mirror   reflect the whole episode across a line through the board center.
#            Only for blocks with a mirror axis through their origin (T, I, +,
#            square, ...; not Z); the center-of-mass offset is mirrored too.
#   recolor  apply an explicit color palette (no effect on physics).
#
# The board (walls at 5..506) is symmetric about BOARD_CENTER, so rotations by
# multiples of 90 deg and mirrors map the walls onto themselves.
import copy
from dataclasses import dataclass, field

import numpy as np

from pusht_scenarios.io.demo import Demo

BOARD_CENTER = np.array([255.5, 255.5])

# DINO-WM's noisy-replay recipe, measured from LeWM's pusht_expert_train.h5:
# 101 copies per demo (CLAUDE.md "Established facts").
DINO_WM_NOISE_LEVELS = (
    (0.0, 11), (0.003, 10), (0.005, 10), (0.01, 10), (0.015, 10),
    (0.02, 10), (0.025, 10), (0.03, 10), (0.07, 10), (0.15, 10),
)

COLOR_KEYS = ('block.color', 'agent.color', 'goal.color', 'background.color')


def wrap_angle(a: float) -> float:
    return float(np.arctan2(np.sin(a), np.cos(a)))


def rotation(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


def reflection(axis_angle: float) -> np.ndarray:
    """Reflection across a line through the origin at axis_angle (rad)."""
    c, s = np.cos(2 * axis_angle), np.sin(2 * axis_angle)
    return np.array([[c, s], [s, -c]])


def angle_of(matrix: np.ndarray) -> float:
    return float(np.arctan2(matrix[1, 0], matrix[0, 0]))


@dataclass
class Candidate:
    """A proposed episode. Untransformed candidates keep the parent's exact
    reset call (so a noise-free copy reproduces the parent bit for bit);
    geometric transforms switch to explicit mode: every variation value set,
    and the start pose placed directly via reset(options={'state': ...})."""

    parent: Demo
    seed: int
    actions: np.ndarray  # (T, 2) proposed, before bounding
    variation: dict  # full flattened variation values
    start_state: np.ndarray  # (7,) agent xy, block xy, block angle, agent vel xy
    explicit: bool = False
    overrides: dict = field(default_factory=dict)  # recolor etc. in non-explicit mode
    chain: list = field(default_factory=list)  # [{'op': 'noise', 'sigma': 0.01}, ...]

    def reset_options(self) -> dict:
        """Plain-data reset options (JSON-safe); stored as meta.reset_options."""
        if self.explicit:
            return {
                'variation': [],
                'variation_values': copy.deepcopy(self.variation),
                'state': np.asarray(self.start_state, dtype=float).tolist(),
            }
        opts = copy.deepcopy(self.parent.meta.reset_options) or {
            'variation': [], 'variation_values': copy.deepcopy(self.parent.meta.variation),
        }
        opts.setdefault('variation_values', {}).update(copy.deepcopy(self.overrides))
        return opts

    def derive(self, op: dict, **changes) -> 'Candidate':
        new = copy.copy(self)
        new.variation = copy.deepcopy(self.variation)
        new.overrides = copy.deepcopy(self.overrides)
        new.start_state = np.array(self.start_state, dtype=float)
        new.chain = [*self.chain, op]
        for k, v in changes.items():
            setattr(new, k, v)
        return new


def from_demo(demo: Demo) -> Candidate:
    return Candidate(
        parent=demo,
        seed=demo.meta.seed,
        actions=np.array(demo.action, dtype=np.float32),
        variation=copy.deepcopy(demo.meta.variation),
        start_state=np.array(demo.state[0], dtype=float),
    )


# --- noise -----------------------------------------------------------------


def noise(c: Candidate, sigma: float, rng: np.random.Generator) -> Candidate:
    """Add i.i.d. Gaussian noise (std sigma, action units) to every action."""
    noisy = c.actions + rng.normal(0.0, sigma, c.actions.shape) if sigma else c.actions.copy()
    return c.derive({'op': 'noise', 'sigma': float(sigma)},
                    actions=np.clip(noisy, -1, 1).astype(np.float32))


# --- geometric transforms --------------------------------------------------


def _transform_episode(c: Candidate, linear: np.ndarray, shift, op: dict,
                       angle_map, cog_map=None) -> Candidate:
    """Apply x -> linear @ (x - C) + C + shift to agent, block and goal
    positions; angle_map to block/goal angles; linear to actions/velocities."""
    shift = np.asarray(shift, dtype=float)
    s = np.array(c.start_state, dtype=float)
    v = copy.deepcopy(c.variation)

    def point(p):
        return linear @ (np.asarray(p, dtype=float) - BOARD_CENTER) + BOARD_CENTER + shift

    s[0:2] = point(s[0:2])
    s[2:4] = point(s[2:4])
    s[4] = wrap_angle(angle_map(s[4]))
    s[5:7] = linear @ s[5:7]
    v['goal.position'] = point(v['goal.position']).tolist()
    v['goal.angle'] = wrap_angle(angle_map(v['goal.angle']))
    if cog_map is not None:
        v['sim.block_cog_offset'] = cog_map(np.asarray(v['sim.block_cog_offset'], float)).tolist()
    actions = (linear @ c.actions.T).T.astype(np.float32)
    return c.derive(op, start_state=s, variation=v, actions=actions, explicit=True)


def se2(c: Candidate, theta: float, shift=(0.0, 0.0)) -> Candidate:
    """Rotate the episode by theta (rad) about the board center, then shift."""
    R = rotation(theta)
    op = {'op': 'se2', 'theta': float(theta), 'shift': [float(x) for x in shift]}
    return _transform_episode(c, R, shift, op, lambda a: a + theta)


def mirror(c: Candidate, axis_deg: float, local_axis_deg: float) -> Candidate:
    """Reflect the episode across the world line through the board center at
    axis_deg. local_axis_deg: a mirror axis of the block in its own frame (see
    local_mirror_axes); the reflected block is the same shape, at a new angle."""
    M = reflection(np.radians(axis_deg))
    S = reflection(np.radians(local_axis_deg))
    op = {'op': 'mirror', 'axis_deg': float(axis_deg), 'local_axis_deg': float(local_axis_deg)}
    # block frame R(a) -> M R(a) S: a rotation (det +1) since det M = det S = -1
    return _transform_episode(
        c, M, (0.0, 0.0), op,
        angle_map=lambda a: angle_of(M @ rotation(a) @ S),
        cog_map=lambda off: S @ off,
    )


def local_mirror_axes(env, candidates_deg=(0, 45, 90, 135), step=2.0) -> list[float]:
    """Mirror axes through the block's origin, in its own frame (degrees).

    Tested on the actual pymunk shapes: a grid of points around the block must
    be inside/outside identically before and after reflection. Also requires
    the block's default center of gravity to lie on the axis, so the reflected
    body has the same mass distribution."""
    block = env.unwrapped.block
    reach = 0.0
    for shape in block.shapes:
        if hasattr(shape, 'get_vertices'):
            reach = max(reach, max(p.length for p in shape.get_vertices()) + shape.radius)
        else:
            reach = max(reach, shape.offset.length + shape.radius)
    xs = np.arange(-reach, reach + step, step)
    grid = np.array([(x, y) for x in xs for y in xs])

    def inside(local_points):
        out = np.empty(len(local_points), dtype=float)
        for i, p in enumerate(local_points):
            world = block.local_to_world(tuple(p))
            out[i] = min(s.point_query(world).distance for s in block.shapes)
        return out

    here = inside(grid)
    clear = np.abs(here) > 0.75  # skip points on the boundary
    # the shape's own CoG: sim.block_cog_offset is mirrored separately
    offset = np.asarray(env.unwrapped.variation_space['sim']['block_cog_offset'].value, float)
    cog = np.asarray(block.center_of_gravity, dtype=float) - offset
    axes = []
    for deg in candidates_deg:
        S = reflection(np.radians(deg))
        there = inside(grid @ S.T)
        ok = np.all((here[clear] < 0) == (there[clear] < 0))
        if ok and np.allclose(S @ cog, cog, atol=1e-6):
            axes.append(float(deg))
    return axes


# --- appearance ------------------------------------------------------------


def recolor(c: Candidate, palette: dict, name: str = '') -> Candidate:
    """Set colors from palette ({'block.color': [r, g, b], ...}); physics unchanged."""
    unknown = set(palette) - set(COLOR_KEYS)
    if unknown:
        raise ValueError(f'unknown palette keys: {sorted(unknown)}')
    colors = {k: [int(x) for x in v] for k, v in palette.items()}
    v = copy.deepcopy(c.variation)
    v.update(colors)
    overrides = {**c.overrides, **colors}
    return c.derive({'op': 'recolor', 'palette': name, 'colors': colors},
                    variation=v, overrides=overrides)
