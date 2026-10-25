# Export curated demos to stable-worldmodel / LeWM HDF5 datasets.
#
#   uv run pts-export -c configs/datasets/v0.yaml [--workers 8] [--limit 20]
#
# Writes <output_root>/:
#   <name>_train.h5        train-env episodes of train parents (+ their augmented copies)
#   <name>_val.h5          train-env episodes of held-out parents (split by parent demo)
#   <name>_test_adapt.h5   test envs: the first `adapt_per_env` clean recorded demos
#                          per env, for test-time adaptation (T2-z: 1-2 trajectories)
#   <name>_test_eval.h5    test envs: the remaining clean recorded demos; swm's
#                          dataset eval samples start/goal windows from these
#   <name>_<split>.episodes.jsonl   one line per episode: demo ids, env, provenance
#   <name>.manifest.yaml   spec, suite envs, counts, columns, versions
#
# Layout (swm HDF5Dataset; same as LeWM's pusht_expert_train.h5): one dataset
# per column with rows for every step of every episode, plus ep_len/ep_offset.
# An episode with T actions has T+1 rows: row t = observation at t and the
# action taken from it; the last row's action is NaN (swm's World.collect).
# Pixels are re-rendered by replaying each demo from reset, and the replay
# must match the stored states exactly (check_exact).
#
# Columns: pixels (uint8, H x W x 3, Blosc-LZ4 chunks of 100 frames), action,
# state, proprio (agent xy + agent velocity), episode_idx, step_idx, seed,
# env_id, demo_idx, parent_idx, is_original, clean, success, on_goal,
# noise_std, base_idx, and variation.<key> for every variation except the start
# pose (the `state` column is the true start). The env's
# reset_options_from_dataset reads these back for swm's dataset eval.
import argparse
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path

import gymnasium as gym
import h5py
import hdf5plugin
import numpy as np
import yaml

import pusht_scenarios.env  # noqa: F401
from pusht_scenarios.env import task, variations
from pusht_scenarios.env.env import PushT
from pusht_scenarios.gen import replay
from pusht_scenarios.io.collection import DemoIndex, DemoRecord
from pusht_scenarios.io.demo import current_state, load_demo
from pusht_scenarios.io.session import git_commit, now_iso, package_version

SPLITS = ('train', 'val', 'test_adapt', 'test_eval')


@dataclass
class SelectConfig:
    include_rejected: bool = False
    include_failures: bool = True  # augmented copies whose outcome is failure
    include_timeouts: bool = True  # recorded timeouts (train/val only, never test)
    max_noise: float | None = None  # drop augmented copies noisier than this
    conditions: list | None = None


@dataclass
class SplitConfig:
    val_fraction: float = 0.1  # of each train env's recorded demos (parents)
    adapt_per_env: int = 2
    test_include_augmented: bool = False  # augmented copies of test parents too


@dataclass
class ExportSpec:
    name: str
    sources: list = field(default_factory=lambda: ['data/raw'])
    suite: str | None = None  # built suite.yaml; None = every condition is a train env
    output_root: str = 'data/export'
    resolution: int = 224
    seed: int = 0
    pixel_chunk: int = 100
    check_exact: bool = True
    select: SelectConfig = field(default_factory=SelectConfig)
    splits: SplitConfig = field(default_factory=SplitConfig)


def spec_from_dict(raw: dict) -> ExportSpec:
    raw = dict(raw)
    for cls, key in ((ExportSpec, None), (SelectConfig, 'select'), (SplitConfig, 'splits')):
        part = raw if key is None else (raw.get(key) or {})
        unknown = set(part) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f'unknown {key or "export"} keys: {sorted(unknown)}')
    select = SelectConfig(**(raw.pop('select', None) or {}))
    splits = SplitConfig(**(raw.pop('splits', None) or {}))
    if not 0 <= splits.val_fraction < 1:
        raise ValueError('splits.val_fraction must be in [0, 1)')
    return ExportSpec(**raw, select=select, splits=splits)


def load_spec(path) -> ExportSpec:
    with open(path) as f:
        return spec_from_dict(yaml.safe_load(f) or {})


# --- environments and splits -------------------------------------------------


@dataclass
class EnvInfo:
    env_id: int
    name: str
    split: str  # train | test
    levels: dict = field(default_factory=dict)
    env_hash: str = ''


def resolve_envs(spec: ExportSpec, index: DemoIndex) -> dict[str, EnvInfo]:
    """condition name -> EnvInfo, from the suite (if any) or the conditions seen."""
    if spec.suite:
        from pusht_scenarios.suite import load_suite

        suite = load_suite(spec.suite)
        return {e.name: EnvInfo(e.env_id, e.name, e.split, e.levels, e.env_hash) for e in suite.envs}
    names = sorted({r.meta.condition for r in index})
    return {n: EnvInfo(i, n, 'train') for i, n in enumerate(names)}


def _unit(seed: int, key: str) -> float:
    h = hashlib.sha1(f'{seed}:{key}'.encode()).digest()
    return int.from_bytes(h[:8], 'big') / 2**64


def assign_splits(spec: ExportSpec, index: DemoIndex, envs: dict) -> dict[str, list[DemoRecord]]:
    """Every kept record -> one split (or none), by parent demo, never by episode."""
    sel = spec.select
    records = index.select(include_rejected=sel.include_rejected, conditions=sel.conditions)
    unknown = sorted({r.meta.condition for r in records} - set(envs))
    if unknown:
        raise ValueError(f'conditions not in the suite: {unknown}')

    parent_split: dict[str, str] = {}
    originals = [r for r in records if not r.is_derived]
    by_env: dict[str, list[DemoRecord]] = {}
    for r in originals:
        by_env.setdefault(r.meta.condition, []).append(r)
    for cond, rs in by_env.items():
        env = envs[cond]
        rs = sorted(rs, key=lambda r: (r.meta.seed, r.meta.demo_id))
        if env.split == 'train':
            for r in rs:
                if r.meta.outcome == 'timeout' and not sel.include_timeouts:
                    continue
                val = _unit(spec.seed, r.meta.demo_id) < spec.splits.val_fraction
                parent_split[r.meta.demo_id] = 'val' if val else 'train'
        else:  # test env: only clean successful recordings
            ok = [r for r in rs if r.meta.outcome == 'success']
            for k, r in enumerate(ok):
                parent_split[r.meta.demo_id] = 'test_adapt' if k < spec.splits.adapt_per_env else 'test_eval'

    out = {s: [] for s in SPLITS}
    for r in records:
        split = parent_split.get(r.parent_id)
        if split is None:
            continue
        if r.is_derived:
            if split.startswith('test') and not spec.splits.test_include_augmented:
                continue
            if r.meta.outcome == 'failure' and not sel.include_failures:
                continue
            if sel.max_noise is not None and r.noise_std > sel.max_noise:
                continue
        out[split].append(r)
    for s in out:
        out[s].sort(key=lambda r: (envs[r.meta.condition].env_id, r.parent_id,
                                   r.is_derived, r.path.name))
    return out


# --- rendering (worker processes) --------------------------------------------

_ENV = None


def _env(resolution):
    global _ENV
    if _ENV is None or _ENV.unwrapped.render_size != resolution:
        _ENV = gym.make('pts/PushT-v0', resolution=resolution)
    return _ENV


def render_episode(job) -> dict:
    """Replay one demo from reset; return its per-row arrays (T+1 rows)."""
    path, resolution, check_exact = job
    demo = load_demo(path)
    env = _env(resolution)
    replay.reset_to(env, demo)
    frames, on_goal, states = [env.render()], [task.block_on_goal(env)], [current_state(env)]
    for a in demo.action:
        env.step(a)
        frames.append(env.render())
        on_goal.append(task.block_on_goal(env))
        states.append(current_state(env))
    states = np.asarray(states)
    err = float(np.max(np.abs(states - demo.state)))
    if check_exact and err != 0.0:
        raise RuntimeError(f'{path}: replay is not exact (max state error {err:.3e})')
    T = len(demo)
    action = np.full((T + 1, 2), np.nan, dtype=np.float32)
    action[:T] = demo.action
    state = demo.state.astype(np.float64)
    var = {k: v for k, v in demo.meta.variation.items() if k not in PushT.START_POSE_KEYS}
    return {
        'pixels': np.asarray(frames, dtype=np.uint8),
        'action': action,
        'state': state,
        'proprio': state[:, [0, 1, 5, 6]].copy(),
        'on_goal': np.asarray(on_goal, dtype=np.uint8),
        '_variation': var,
        '_meta': demo.meta,
    }


# --- writing -----------------------------------------------------------------


def _compression(kind):
    shuffle = hdf5plugin.Blosc.SHUFFLE if kind == 'pixels' else hdf5plugin.Blosc.BITSHUFFLE
    return hdf5plugin.Blosc(cname='lz4', clevel=5, shuffle=shuffle)


class H5Writer:
    """Appends episodes to a swm-layout HDF5 file."""

    def __init__(self, path, pixel_chunk=100):
        self.f = h5py.File(path, 'w')
        self.pixel_chunk = pixel_chunk
        self.rows = 0
        self.f.create_dataset('ep_len', shape=(0,), maxshape=(None,), dtype=np.int32)
        self.f.create_dataset('ep_offset', shape=(0,), maxshape=(None,), dtype=np.int64)

    def _dataset(self, name, arr):
        if name in self.f:
            ds = self.f[name]
            if ds.shape[1:] != arr.shape[1:] or ds.dtype != arr.dtype:
                raise ValueError(f'column {name}: {arr.dtype}{arr.shape[1:]} vs {ds.dtype}{ds.shape[1:]}')
            return ds
        if self.rows:
            raise ValueError(f'column {name} missing from earlier episodes')
        chunk = self.pixel_chunk if name == 'pixels' else 4096
        return self.f.create_dataset(
            name, shape=(0, *arr.shape[1:]), maxshape=(None, *arr.shape[1:]), dtype=arr.dtype,
            chunks=(chunk, *arr.shape[1:]), **_compression('pixels' if name == 'pixels' else 'other'))

    def write(self, columns: dict) -> None:
        n = len(next(iter(columns.values())))
        if set(columns) != set(self.f) - {'ep_len', 'ep_offset'} and self.rows:
            missing = (set(self.f) - {'ep_len', 'ep_offset'}) ^ set(columns)
            raise ValueError(f'columns differ from earlier episodes: {sorted(missing)}')
        for name, arr in columns.items():
            if len(arr) != n:
                raise ValueError(f'column {name}: {len(arr)} rows, expected {n}')
            ds = self._dataset(name, arr)
            ds.resize(self.rows + n, axis=0)
            ds[self.rows:self.rows + n] = arr
        k = self.f['ep_len'].shape[0]
        for name, value in (('ep_len', n), ('ep_offset', self.rows)):
            self.f[name].resize(k + 1, axis=0)
            self.f[name][k] = value
        self.rows += n

    def close(self):
        self.f.close()


def episode_columns(ep: dict, record: DemoRecord, env: EnvInfo, episode_idx: int,
                    demo_idx: int, parent_idx: int, base_idx: int) -> dict:
    n = len(ep['state'])
    m = ep['_meta']

    def const(value, dtype):
        return np.full(n, value, dtype=dtype)

    cols = {
        'pixels': ep['pixels'],
        'action': ep['action'],
        'state': ep['state'],
        'proprio': ep['proprio'],
        'episode_idx': const(episode_idx, np.int64),
        'step_idx': np.arange(n, dtype=np.int64),
        'seed': const(m.seed, np.int64),
        'env_id': const(env.env_id, np.int64),
        'demo_idx': const(demo_idx, np.int64),
        'parent_idx': const(parent_idx, np.int64),
        'is_original': const(not record.is_derived, np.uint8),
        'clean': const(record.is_clean_original, np.uint8),
        'success': const(m.outcome == 'success', np.uint8),
        'on_goal': ep['on_goal'],
        'noise_std': const(record.noise_std, np.float32),
        'base_idx': const(base_idx, np.int16),
    }
    dtypes = variation_dtypes()
    for key, value in sorted(ep['_variation'].items()):
        arr = np.asarray(value, dtype=dtypes[key])  # the space's dtype, not the value's
        cols[f'variation.{key}'] = np.broadcast_to(arr, (n, *arr.shape)).copy()
    return cols


_DTYPES = None


def variation_dtypes() -> dict:
    """Column dtype per variation key, from the env's variation space (stored
    values don't carry it: an unsampled float can still be a Python int)."""
    global _DTYPES
    if _DTYPES is None:
        from pusht_scenarios.env.utils import get_in

        space = PushT().variation_space
        _DTYPES = {k: np.dtype(get_in(space, k.split('.')).dtype) for k in space.names()}
    return _DTYPES


def run_export(spec: ExportSpec, workers: int | None = None, limit: int | None = None,
               verbose: bool = True) -> Path:
    out = Path(spec.output_root)
    targets = [out / f'{spec.name}_{s}.h5' for s in SPLITS] + [out / f'{spec.name}.manifest.yaml']
    existing = [p for p in targets if p.exists()]
    if existing:
        raise FileExistsError(f'export outputs already exist: {[str(p) for p in existing]}')
    out.mkdir(parents=True, exist_ok=True)

    index = DemoIndex.scan(*spec.sources)
    envs = resolve_envs(spec, index)
    splits = assign_splits(spec, index, envs)
    if limit:
        splits = {s: rs[:limit] for s, rs in splits.items()}

    demo_ids = sorted({r.meta.demo_id for rs in splits.values() for r in rs}
                      | {r.parent_id for rs in splits.values() for r in rs})
    demo_idx = {d: i for i, d in enumerate(demo_ids)}
    bases = sorted({r.base for rs in splits.values() for r in rs})
    base_idx = {b: i for i, b in enumerate(bases)}

    counts = {}
    pool = ProcessPoolExecutor(max_workers=workers) if (workers or 0) > 1 else None
    try:
        for split in SPLITS:
            records = splits[split]
            counts[split] = {'episodes': 0, 'rows': 0}
            if not records:
                continue
            jobs = [(r.path, spec.resolution, spec.check_exact) for r in records]
            episodes = pool.map(render_episode, jobs, chunksize=4) if pool else map(render_episode, jobs)
            writer = H5Writer(out / f'{spec.name}_{split}.h5', spec.pixel_chunk)
            lines = []
            try:
                for k, (r, ep) in enumerate(zip(records, episodes)):
                    env = envs[r.meta.condition]
                    writer.write(episode_columns(ep, r, env, k, demo_idx[r.meta.demo_id],
                                                 demo_idx[r.parent_id], base_idx[r.base]))
                    lines.append({
                        'episode_idx': k, 'demo_idx': demo_idx[r.meta.demo_id],
                        'demo_id': r.meta.demo_id, 'parent_id': r.parent_id,
                        'condition': r.meta.condition, 'env_id': env.env_id,
                        'base': r.base, 'noise_std': r.noise_std, 'outcome': r.meta.outcome,
                        'steps': r.meta.steps, 'source': str(r.path),
                    })
                rows = writer.rows
            finally:
                writer.close()
            with open(out / f'{spec.name}_{split}.episodes.jsonl', 'w') as f:
                for line in lines:
                    f.write(json.dumps(line) + '\n')
            counts[split] = {'episodes': len(lines), 'rows': rows,
                             'envs': sorted({l['env_id'] for l in lines})}
            if verbose:
                print(f'  {split:<10} {len(lines):6d} episodes  {rows:8d} rows')
    finally:
        if pool:
            pool.shutdown()

    manifest = {
        'name': spec.name,
        'created_at': now_iso(),
        'package_version': package_version(),
        'git_commit': git_commit(),
        'spec': asdict(spec),
        'envs': [asdict(e) for e in sorted(envs.values(), key=lambda e: e.env_id)],
        'bases': bases,
        'splits': counts,
        'files': {s: f'{spec.name}_{s}.h5' for s in SPLITS if counts[s]['episodes']},
        'notes': {
            'rows': 'T+1 rows per episode; the last row action is NaN',
            'eval_env': 'pts/PushTBounded-v0 (reset_options_from_dataset restores each episode)',
            'start_pose': 'variation.* omits the start pose; the state column is the true start',
        },
    }
    with open(out / f'{spec.name}.manifest.yaml', 'w') as f:
        yaml.safe_dump(manifest, f, sort_keys=False)
    if verbose:
        print(f'wrote {out}')
    return out


def run(argv=None):
    parser = argparse.ArgumentParser(prog='pts-export', description='Export demos to swm/LeWM HDF5.')
    parser.add_argument('-c', '--config', required=True, help='dataset spec YAML')
    parser.add_argument('--workers', type=int, help='render processes (default: 1)')
    parser.add_argument('--limit', type=int, help='at most N episodes per split (smoke tests)')
    parser.add_argument('--name', help='override the dataset name')
    args = parser.parse_args(argv)
    spec = load_spec(args.config)
    if args.name:
        spec.name = args.name
    return run_export(spec, workers=args.workers, limit=args.limit)


def main(argv=None):
    """Console entry point (returns None so the exit status is 0)."""
    run(argv)


if __name__ == '__main__':
    main()
