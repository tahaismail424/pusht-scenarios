"""HDF5 export (export/hdf5.py): splits by parent, swm/LeWM row layout,
columns, pixels, manifests, and loading with swm's HDF5Dataset."""

import json
from pathlib import Path

import gymnasium as gym
import h5py
import numpy as np
import pytest
import yaml

import pusht_scenarios.env  # noqa: F401
from pusht_scenarios.env.env import PushT
from pusht_scenarios.export import hdf5 as X
from pusht_scenarios.gen import replay
from pusht_scenarios.io.collection import DemoIndex
from pusht_scenarios.io.demo import load_demo, save_demo

DATASETS = Path(__file__).parents[1] / 'configs' / 'datasets'


def mini_spec(mini, out, **changes):
    raw = {'name': 'mini', 'sources': [str(mini['raw']), str(mini['aug'])],
           'suite': str(mini['suite_path']), 'output_root': str(out),
           'splits': {'val_fraction': 0.34, 'adapt_per_env': 1}}
    for k, v in changes.items():
        if isinstance(v, dict) and isinstance(raw.get(k), dict):
            raw[k] = {**raw[k], **v}
        else:
            raw[k] = v
    return X.spec_from_dict(raw)


@pytest.fixture(scope='module')
def exported(mini, tmp_path_factory):
    out = tmp_path_factory.mktemp('export')
    X.run_export(mini_spec(mini, out), workers=1, verbose=False)
    return out


def episodes(out, split):
    return [json.loads(line) for line in (out / f'mini_{split}.episodes.jsonl').read_text().splitlines()]


# --- spec -------------------------------------------------------------------


@pytest.mark.parametrize('raw, match', [
    ({'name': 'x', 'colour': 1}, 'export'),
    ({'name': 'x', 'select': {'failures': True}}, 'select'),
    ({'name': 'x', 'splits': {'val': 0.1}}, 'splits'),
    ({'name': 'x', 'splits': {'val_fraction': 1.5}}, 'val_fraction'),
])
def test_spec_validation(raw, match):
    with pytest.raises(ValueError, match=match):
        X.spec_from_dict(raw)


@pytest.mark.parametrize('path', sorted(DATASETS.glob('*.yaml')), ids=lambda p: p.name)
def test_shipped_specs_parse(path):
    assert X.load_spec(path).name


# --- splits -----------------------------------------------------------------


@pytest.fixture(scope='module')
def splits(mini):
    spec = mini_spec(mini, '.')
    index = DemoIndex.scan(*spec.sources)
    envs = X.resolve_envs(spec, index)
    return X.assign_splits(spec, index, envs), envs


def test_splits_by_parent_never_by_episode(splits):
    s, _ = splits
    parents = {k: {r.parent_id for r in rs} for k, rs in s.items()}
    for a in X.SPLITS:
        for b in X.SPLITS:
            if a < b:
                assert not parents[a] & parents[b], (a, b)


def test_train_envs_go_to_train_val_test_envs_to_test(splits):
    s, envs = splits
    for split, rs in s.items():
        for r in rs:
            env_split = envs[r.meta.condition].split
            assert (env_split == 'train') == (split in ('train', 'val')), (split, r.meta.condition)
    assert s['train'] and s['val'] and s['test_adapt'] and s['test_eval']


def test_test_splits_hold_only_clean_successful_recordings(splits):
    s, envs = splits
    for split in ('test_adapt', 'test_eval'):
        for r in s[split]:
            assert not r.is_derived and r.meta.outcome == 'success'
    per_env = {}
    for r in s['test_adapt']:
        per_env.setdefault(r.meta.condition, []).append(r)
    assert all(len(v) == 1 for v in per_env.values())  # adapt_per_env: 1
    assert {r.meta.condition for r in s['test_eval']} == set(per_env)
    first = {c: min(rs + [r for r in s['test_eval'] if r.meta.condition == c],
                    key=lambda r: r.meta.seed) for c, rs in per_env.items()}
    assert all(per_env[c][0] is first[c] for c in per_env)  # adapt = lowest seeds


def test_train_includes_augmented_copies_of_its_parents(splits):
    s, _ = splits
    train_parents = {r.meta.demo_id for r in s['train'] if not r.is_derived}
    derived = [r for r in s['train'] if r.is_derived]
    assert derived and all(r.parent_id in train_parents for r in derived)


def test_select_options(mini):
    def split_with(**kw):
        spec = mini_spec(mini, '.', **kw)
        index = DemoIndex.scan(*spec.sources)
        return X.assign_splits(spec, index, X.resolve_envs(spec, index))

    base = split_with()
    no_fail = split_with(select={'include_failures': False})
    assert all(r.meta.outcome != 'failure' for r in no_fail['train'] if r.is_derived)
    no_noise = split_with(select={'max_noise': 0.0})
    assert all(r.noise_std == 0.0 for r in no_noise['train'])
    no_timeouts = split_with(select={'include_timeouts': False})
    assert not no_timeouts['train'] and not no_timeouts['val']  # every train recording timed out
    aug_test = split_with(splits={'test_include_augmented': True})
    assert len(aug_test['test_eval']) > len(base['test_eval'])


def test_without_suite_every_condition_is_a_train_env(mini):
    spec = mini_spec(mini, '.', suite=None)
    index = DemoIndex.scan(*spec.sources)
    envs = X.resolve_envs(spec, index)
    assert {e.split for e in envs.values()} == {'train'}
    assert sorted(e.env_id for e in envs.values()) == list(range(len(envs)))


# --- files --------------------------------------------------------------------


def test_layout_matches_swm(exported):
    with h5py.File(exported / 'mini_train.h5') as f:
        ep_len, ep_off = f['ep_len'][:], f['ep_offset'][:]
        assert f['ep_len'].dtype == np.int32 and f['ep_offset'].dtype == np.int64
        np.testing.assert_array_equal(ep_off, np.concatenate([[0], np.cumsum(ep_len)[:-1]]))
        n = int(ep_len.sum())
        for col in f:
            if col not in ('ep_len', 'ep_offset'):
                assert len(f[col]) == n, col
        assert f['pixels'].shape[1:] == (224, 224, 3) and f['pixels'].dtype == np.uint8
        assert f['pixels'].chunks[0] == 100


def test_rows_are_t_plus_one_with_nan_final_action(exported):
    lines = episodes(exported, 'train')
    with h5py.File(exported / 'mini_train.h5') as f:
        for line in lines[:6]:
            k = line['episode_idx']
            off, n = f['ep_offset'][k], f['ep_len'][k]
            demo = load_demo(line['source'])
            assert n == len(demo) + 1
            action = f['action'][off:off + n]
            np.testing.assert_array_equal(action[:-1], demo.action)
            assert np.isnan(action[-1]).all()
            np.testing.assert_array_equal(f['state'][off:off + n], demo.state)
            np.testing.assert_array_equal(f['proprio'][off:off + n], demo.state[:, [0, 1, 5, 6]])
            np.testing.assert_array_equal(f['step_idx'][off:off + n], np.arange(n))
            assert (f['episode_idx'][off:off + n] == k).all()


def test_pixels_are_the_replayed_frames(exported):
    line = episodes(exported, 'test_eval')[0]
    demo = load_demo(line['source'])
    env = gym.make('pts/PushT-v0', resolution=224)
    replay.reset_to(env, demo)
    first = env.render()
    for a in demo.action:
        env.step(a)
    last = env.render()
    with h5py.File(exported / 'mini_test_eval.h5') as f:
        off, n = f['ep_offset'][line['episode_idx']], f['ep_len'][line['episode_idx']]
        np.testing.assert_array_equal(f['pixels'][off], first)
        np.testing.assert_array_equal(f['pixels'][off + n - 1], last)


def test_metadata_columns(exported, mini):
    suite = mini['suite'].by_name()
    lines = episodes(exported, 'train')
    with h5py.File(exported / 'mini_train.h5') as f:
        variation_cols = [c for c in f if c.startswith('variation.')]
        assert not {f'variation.{k}' for k in PushT.START_POSE_KEYS} & set(variation_cols)
        assert 'variation.sim.block_mass' in variation_cols and 'variation.block.color' in variation_cols
        assert f['variation.block.color'].dtype == np.uint8
        for line in lines:
            off = f['ep_offset'][line['episode_idx']]
            env = suite[line['condition']]
            assert f['env_id'][off] == env.env_id == line['env_id']
            assert f['variation.sim.block_mass'][off] == env.variation_values['sim.block_mass']
            assert f['is_original'][off] == (line['base'] == 'original' and line['demo_id'] == line['parent_id'])
            assert f['noise_std'][off] == np.float32(line['noise_std'])
            assert f['success'][off] == (line['outcome'] == 'success')
            assert not any(c.startswith('goal') for c in f)  # swm's eval skips goal* columns


def test_parent_and_demo_indices_link_copies(exported):
    with h5py.File(exported / 'mini_train.h5') as f:
        off = f['ep_offset'][:]
        demo_idx, parent_idx, orig = f['demo_idx'][:][off], f['parent_idx'][:][off], f['is_original'][:][off]
    originals = set(demo_idx[orig == 1])
    assert set(parent_idx) <= originals
    assert (demo_idx[orig == 1] == parent_idx[orig == 1]).all()


def test_manifest_and_episode_lists(exported):
    m = yaml.safe_load((exported / 'mini.manifest.yaml').read_text())
    assert [e['env_id'] for e in m['envs']] == list(range(6))
    for split in X.SPLITS:
        lines = episodes(exported, split)
        assert m['splits'][split]['episodes'] == len(lines)
        with h5py.File(exported / f'mini_{split}.h5') as f:
            assert m['splits'][split]['rows'] == int(f['ep_len'][:].sum())
    assert m['notes']['eval_env'].startswith('pts/PushTBounded-v0')


def test_loads_in_swm(exported):
    from stable_worldmodel.data.formats.hdf5 import HDF5Dataset

    ds = HDF5Dataset(path=exported / 'mini_train.h5', num_steps=4)
    sample = ds[0]
    assert tuple(sample['pixels'].shape) == (4, 3, 224, 224)
    assert tuple(sample['action'].shape) == (4, 2) and tuple(sample['state'].shape) == (4, 7)
    assert 'variation.sim.block_mass' in sample and 'env_id' in sample


def test_refuses_existing_outputs(mini, exported):
    with pytest.raises(FileExistsError):
        X.run_export(mini_spec(mini, exported), verbose=False)


def test_parallel_export_matches_serial(mini, exported, tmp_path):
    X.run_export(mini_spec(mini, tmp_path), workers=2, verbose=False)
    for split in X.SPLITS:
        with h5py.File(exported / f'mini_{split}.h5') as a, h5py.File(tmp_path / f'mini_{split}.h5') as b:
            assert set(a) == set(b)
            for col in ('pixels', 'state', 'env_id', 'demo_idx'):
                np.testing.assert_array_equal(a[col][:], b[col][:])


def test_limit(mini, tmp_path):
    X.run_export(mini_spec(mini, tmp_path), limit=2, verbose=False)
    m = yaml.safe_load((tmp_path / 'mini.manifest.yaml').read_text())
    assert all(v['episodes'] <= 2 for v in m['splits'].values())


def test_inexact_replay_is_refused(mini, tmp_path):
    src = tmp_path / 'src' / 'demos'
    demo = load_demo(next(iter(sorted(mini['raw'].rglob('*.npz')))))
    demo.state[3, 2] += 1.0  # corrupt a stored state
    save_demo(demo, src / 'bad.npz')
    spec = X.spec_from_dict({'name': 'bad', 'sources': [str(src.parent)], 'output_root': str(tmp_path / 'out')})
    with pytest.raises(RuntimeError, match='not exact'):
        X.run_export(spec, verbose=False)


# --- full-factorial recordings, re-split at export ---------------------------


@pytest.fixture(scope='module')
def factorial(tmp_path_factory):
    """Mini suite with record_all: all 3x3 combinations recorded (2 successes each)."""
    import pygame

    from pusht_scenarios import suite as S
    from pusht_scenarios.env.bounds import KeepAgentOnBoard
    from pusht_scenarios.teleop import record
    from pusht_scenarios.teleop.input import InputDevice

    from conftest import MINI_SUITE, ON_GOAL, IdleStick

    root = tmp_path_factory.mktemp('factorial')
    raw = {**MINI_SUITE, 'record_all': True,
           'record': {**MINI_SUITE['record'], 'rounds': {'all': 2}}}
    built = S.build(S.spec_from_dict(raw))
    cfg = record.record_config_from_dict(S.record_config(built))
    cfg.output.root = str(root / 'raw')
    for c in cfg.sandbox.conditions:
        c.variation, c.variation_values = [], {**c.variation_values, **ON_GOAL}
    pygame.init()
    env = KeepAgentOnBoard(gym.make('pts/PushT-v0', render_mode='human'))
    record.record(cfg, env=env, device=InputDevice(joystick=IdleStick()), get_events=lambda: [], verbose=False)
    env.close()
    pygame.quit()
    suites = {}
    for name, overrides in (('middle', []), ('high', ['mass=heavy'])):
        out = root / name
        S.write(S.build(S.apply_overrides(S.spec_from_dict(raw), overrides)), out)
        suites[name] = out / 'suite.yaml'
    return {'raw': root / 'raw', 'suites': suites, 'root': root}


def _split_names(spec):
    index = DemoIndex.scan(*spec.sources)
    s = X.assign_splits(spec, index, X.resolve_envs(spec, index))
    return {k: {r.meta.condition for r in v} for k, v in s.items()}


def test_same_recordings_resplit_by_holdout(factorial):
    base = {'name': 'f', 'sources': [str(factorial['raw'])], 'output_root': '.',
            'splits': {'val_fraction': 0.0, 'adapt_per_env': 1}}
    mid = _split_names(X.spec_from_dict({**base, 'suite': str(factorial['suites']['middle'])}))
    high = _split_names(X.spec_from_dict({**base, 'suite': str(factorial['suites']['high'])}))
    assert all('mass-normal' not in n for n in mid['train'])
    assert all('mass-heavy' not in n for n in high['train'])
    assert any('mass-normal' in n for n in high['train'])
    # 9 recorded combos: 4 train + 2 test used, 3 unused (not exported)
    assert len(mid['train']) == 4 and len(mid['test_adapt']) == 2
    used = mid['train'] | mid['test_adapt'] | mid['test_eval']
    assert len(used) == 6


def test_resplit_export_via_cli_suite_override(factorial, tmp_path):
    spec_path = tmp_path / 'spec.yaml'
    spec_path.write_text(yaml.safe_dump({'name': 'f', 'sources': [str(factorial['raw'])],
                                         'suite': str(factorial['suites']['middle']),
                                         'output_root': str(tmp_path / 'out'),
                                         'splits': {'val_fraction': 0.0, 'adapt_per_env': 1}}))
    X.run(['-c', str(spec_path), '--suite', str(factorial['suites']['high'])])
    m = yaml.safe_load((tmp_path / 'out' / 'f.manifest.yaml').read_text())
    assert m['spec']['suite'] == str(factorial['suites']['high'])
    lines = [json.loads(l) for l in (tmp_path / 'out' / 'f_train.episodes.jsonl').read_text().splitlines()]
    assert lines and all('mass-heavy' not in l['condition'] for l in lines)


def test_changed_level_values_are_refused(factorial, tmp_path):
    from pusht_scenarios import suite as S

    from conftest import MINI_SUITE

    raw = {**MINI_SUITE, 'record_all': True}
    raw['factors'] = {**MINI_SUITE['factors'],
                      'mass': {**MINI_SUITE['factors']['mass'],
                               'levels': {**MINI_SUITE['factors']['mass']['levels'],
                                          'heavy': {'sim.block_mass': 4.0}}}}
    S.write(S.build(S.spec_from_dict(raw)), tmp_path)
    spec = X.spec_from_dict({'name': 'f', 'sources': [str(factorial['raw'])],
                             'suite': str(tmp_path / 'suite.yaml'), 'output_root': '.'})
    index = DemoIndex.scan(*spec.sources)
    with pytest.raises(ValueError, match='env_hash mismatch'):
        X.assign_splits(spec, index, X.resolve_envs(spec, index))


def test_conditions_outside_the_suite_are_skipped(mini, tmp_path):
    # e.g. a demo.yaml session living in the same data/raw
    import shutil

    src = tmp_path / 'raw'
    shutil.copytree(mini['raw'], src)
    other = load_demo(next(iter(sorted(src.rglob('*.npz')))))
    other.meta.condition, other.meta.demo_id, other.meta.condition_info = 'not_in_suite', 'f' * 32, {}
    save_demo(other, src / 'extra' / 'demos' / 'x.npz')
    spec = mini_spec(mini, '.', sources=[str(src)])
    index = DemoIndex.scan(*spec.sources)
    s = X.assign_splits(spec, index, X.resolve_envs(spec, index))
    assert all(r.meta.condition != 'not_in_suite' for rs in s.values() for r in rs)


def test_export_with_a_joint_holdout(factorial, tmp_path):
    from pusht_scenarios import suite as S

    from conftest import MINI_SUITE

    raw = {**MINI_SUITE, 'record_all': True}
    built = S.build(S.apply_overrides(S.spec_from_dict(raw), [], 'all', combos=['mass=heavy,color=warm']))
    S.write(built, tmp_path)
    names = _split_names(X.spec_from_dict({
        'name': 'j', 'sources': [str(factorial['raw'])], 'suite': str(tmp_path / 'suite.yaml'),
        'output_root': '.', 'splits': {'val_fraction': 0.0, 'adapt_per_env': 1}}))
    assert names['test_adapt'] == names['test_eval'] == {'mass-heavy.color-warm'}
    assert len(names['train']) == 8
    assert 'mass-heavy.color-default' in names['train'] and 'mass-light.color-warm' in names['train']
