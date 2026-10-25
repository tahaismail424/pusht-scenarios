# One recorded demo: the raw, re-simulatable record of truth.
#
# Rules (CLAUDE.md, decisions 4 and 2):
# - Store state + the exact (bounded) actions passed to env.step(), never pixels.
#   Pixels are rendered at export, so any resolution/color can be re-rendered.
# - A demo file is immutable once written. Curation (reject/keep) lives outside
#   the file (session events, later io/collection.py).
# - Everything needed to rebuild the env exactly: the fully resolved variation
#   values (including randomly sampled ones), seed, and env id.
#
# Per-step arrays (T = number of steps):
#   state      (T+1, 7)  swm obs['state'] before each step, plus the final one
#   sim_state  (T+1, K)  full physical state (SIM_STATE_FIELDS), so a demo can be
#                        restored at any step even with momentum on, which the
#                        7-number swm state can't do
#   action     (T, 2)    bounded action actually applied (info['applied_action'])
#   raw_action (T, 2)    device output before bounding (quality diagnostics)
#   on_goal    (T,)      task.block_on_goal after each step
#   t_wall     (T,)      wall-clock seconds since round start (pace, fatigue)
#
# File format: one compressed .npz per demo; meta is a JSON string inside it
# (no pickle). Written to a temp name and renamed, so a crash never leaves a
# half-written demo.
import json
import os
import tempfile
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

import numpy as np

from pusht_scenarios.env import variations

SIM_STATE_FIELDS = (
    'agent_x', 'agent_y', 'agent_vx', 'agent_vy',
    # the body the PD controller drives: the agent itself when kinematic (swm),
    # its control body with sim.agent_max_force
    'control_x', 'control_y', 'control_vx', 'control_vy',
    'block_x', 'block_y', 'block_angle',
    'block_vx', 'block_vy', 'block_omega',
)
ARRAYS = ('state', 'sim_state', 'action', 'raw_action', 'on_goal', 't_wall')
FORMAT_VERSION = 1


@dataclass
class DemoMeta:
    demo_id: str  # uuid4 hex; stable across merges/renames
    session_id: str
    condition: str  # condition name from the config
    seed: int
    variation: dict  # fully resolved, flattened variation values
    outcome: str  # success | timeout
    steps: int
    seconds: float
    attempt: int = 1  # 1 + number of retries of this seed before this demo
    # The exact reset(options=...) used, as plain data: {'variation': [...] or
    # None, 'variation_values': {...}}. Replay must call reset with these same
    # options and seed: reset's internal random draws depend on them, and those
    # draws leave traces in the start state (see gen/replay.py).
    reset_options: dict = field(default_factory=dict)
    device: dict = field(default_factory=dict)  # kind, name, gain, ...
    env_id: str = 'pts/PushT-v0'
    operator: str = ''
    created_at: str = ''  # ISO 8601
    package_version: str = ''
    git_commit: str = ''
    format_version: int = FORMAT_VERSION
    # provenance for derived demos (trim, augmentation): {'parent': demo_id, ...}
    derived_from: dict | None = None
    # suite environment this demo was played in (suite.py): {'suite', 'env_id',
    # 'env_name', 'split', 'levels', 'env_hash'}; {} for ad-hoc conditions
    condition_info: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict) -> 'DemoMeta':
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass
class Demo:
    meta: DemoMeta
    state: np.ndarray
    sim_state: np.ndarray
    action: np.ndarray
    raw_action: np.ndarray
    on_goal: np.ndarray
    t_wall: np.ndarray

    def __len__(self) -> int:
        return len(self.action)

    def validate(self) -> None:
        """Raise ValueError if shapes/values are inconsistent."""
        T = len(self.action)
        expected = {
            'state': (T + 1, 7),
            'sim_state': (T + 1, len(SIM_STATE_FIELDS)),
            'action': (T, 2),
            'raw_action': (T, 2),
            'on_goal': (T,),
            't_wall': (T,),
        }
        for name, shape in expected.items():
            got = getattr(self, name).shape
            if got != shape:
                raise ValueError(f'{name}: shape {got}, expected {shape}')
        for name in ('state', 'sim_state', 'action', 'raw_action', 't_wall'):
            if not np.all(np.isfinite(getattr(self, name))):
                raise ValueError(f'{name}: non-finite values')
        if T and np.abs(self.action).max() > 1:
            raise ValueError('action outside [-1, 1]')
        if self.meta.steps != T:
            raise ValueError(f'meta.steps={self.meta.steps} but {T} actions')


def capture_sim_state(env) -> np.ndarray:
    base = env.unwrapped
    a, c, b = base.agent, base.agent_driver, base.block
    return np.array(
        [*a.position, *a.velocity, *c.position, *c.velocity,
         *b.position, b.angle, *b.velocity, b.angular_velocity],
        dtype=np.float64,
    )


def restore_sim_state(env, sim_state) -> None:
    """Set agent, control body and block (incl. velocities) to sim_state.

    Unlike env._set_state this restores block velocity/spin and takes no
    physics step. The env must already be reset to the demo's variation.

    The restored state is exact, but a rollout continued from it is NOT a
    reliable replay: pymunk's cached contact/joint impulses (warm starting)
    aren't part of any state, and pushing is contact-chaotic, so continuations
    can diverge by hundreds of px within a few steps. Exact re-simulation must
    start from reset (gen/replay.py with start=0).
    """
    base = env.unwrapped
    s = dict(zip(SIM_STATE_FIELDS, np.asarray(sim_state, dtype=np.float64).tolist()))
    base.agent.position = (s['agent_x'], s['agent_y'])
    base.agent.velocity = (s['agent_vx'], s['agent_vy'])
    if base.agent_driver is not base.agent:
        base.agent_driver.position = (s['control_x'], s['control_y'])
        base.agent_driver.velocity = (s['control_vx'], s['control_vy'])
    # angle BEFORE position: pymunk rotates about the center of gravity, which
    # isn't the body origin for most shapes, so setting the angle second would
    # move the origin (swm's _set_state uses the same order)
    base.block.angle = s['block_angle']
    base.block.position = (s['block_x'], s['block_y'])
    base.block.velocity = (s['block_vx'], s['block_vy'])
    base.block.angular_velocity = s['block_omega']
    base.space.reindex_shapes_for_body(base.block)
    base.space.reindex_shapes_for_body(base.agent)


def current_state(env) -> np.ndarray:
    """swm's 7-number obs['state'] for the env as it is now."""
    return np.asarray(env.unwrapped._get_obs(), dtype=np.float64)


class StepBuffer:
    """Accumulates one round in memory; becomes a Demo when the round ends."""

    def __init__(self):
        self.state, self.sim_state = [], []
        self.action, self.raw_action, self.on_goal, self.t_wall = [], [], [], []
        self.variation: dict | None = None

    def start(self, env) -> None:
        """Capture the start state and resolved variation, right after reset."""
        self.variation = variations.flatten(env)
        self.state.append(current_state(env))
        self.sim_state.append(capture_sim_state(env))

    def add(self, env, raw_action, info, on_goal, t_wall) -> None:
        """Append one step; call after env.step (info['applied_action'] = applied action)."""
        if self.variation is None:
            raise RuntimeError('StepBuffer.add before start')
        applied = info.get('applied_action', raw_action)
        self.action.append(np.asarray(applied, dtype=np.float32))
        self.raw_action.append(np.asarray(raw_action, dtype=np.float32))
        self.on_goal.append(bool(on_goal))
        self.t_wall.append(float(t_wall))
        self.state.append(current_state(env))
        self.sim_state.append(capture_sim_state(env))

    def __len__(self) -> int:
        return len(self.action)

    def to_demo(self, meta: DemoMeta) -> Demo:
        meta.variation = self.variation
        meta.steps = len(self.action)

        def stack(rows, dtype, width):
            return np.asarray(rows, dtype=dtype).reshape(len(rows), *width)

        demo = Demo(
            meta=meta,
            state=stack(self.state, np.float64, (7,)),
            sim_state=stack(self.sim_state, np.float64, (len(SIM_STATE_FIELDS),)),
            action=stack(self.action, np.float32, (2,)),
            raw_action=stack(self.raw_action, np.float32, (2,)),
            on_goal=np.asarray(self.on_goal, dtype=bool),
            t_wall=np.asarray(self.t_wall, dtype=np.float64),
        )
        demo.validate()
        return demo


def save_demo(demo: Demo, path) -> Path:
    """Validate and atomically write demo to path (.npz)."""
    demo.validate()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {name: getattr(demo, name) for name in ARRAYS}
    arrays['meta'] = np.array(json.dumps(asdict(demo.meta)))
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix='.tmp-', suffix='.npz')
    try:
        with os.fdopen(fd, 'wb') as f:
            np.savez_compressed(f, **arrays)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise
    return path


def load_meta(path) -> DemoMeta:
    """Just the metadata (reads one small member of the .npz, no arrays)."""
    with np.load(path, allow_pickle=False) as data:
        meta = json.loads(str(data['meta']))
    if meta.get('format_version') != FORMAT_VERSION:
        raise ValueError(f'{path}: format_version {meta.get("format_version")}, expected {FORMAT_VERSION}')
    return DemoMeta.from_dict(meta)


def load_demo(path) -> Demo:
    with np.load(path, allow_pickle=False) as data:
        meta = json.loads(str(data['meta']))
        if meta.get('format_version') != FORMAT_VERSION:
            raise ValueError(
                f'{path}: format_version {meta.get("format_version")}, '
                f'expected {FORMAT_VERSION}'
            )
        demo = Demo(meta=DemoMeta.from_dict(meta), **{n: data[n] for n in ARRAYS})
    demo.validate()
    return demo


def find_demos(*paths) -> list[Path]:
    """Demo files from files and/or directories (searched recursively), sorted."""
    out = []
    for p in map(Path, paths):
        if p.is_dir():
            out.extend(q for q in p.rglob('*.npz') if not q.name.startswith('.tmp-'))
        elif p.suffix == '.npz':
            out.append(p)
        else:
            raise ValueError(f'not a demo file or directory: {p}')
    return sorted(set(out))
