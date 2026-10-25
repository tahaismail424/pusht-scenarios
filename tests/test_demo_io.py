"""Demo data model, .npz format, sim-state capture, and session directories."""

import json

import gymnasium as gym
import numpy as np
import pytest
import yaml

import pusht_scenarios.env  # noqa: F401
from pusht_scenarios.env import variations
from pusht_scenarios.env.bounds import KeepAgentOnBoard
from pusht_scenarios.io import demo as demo_io
from pusht_scenarios.io.demo import (
    SIM_STATE_FIELDS, Demo, DemoMeta, StepBuffer, capture_sim_state, find_demos,
    load_demo, restore_sim_state, save_demo,
)
from pusht_scenarios.io.session import (
    OutputConfig, SessionWriter, read_events, rejected_ids, session_dir_of,
)


@pytest.fixture
def env():
    env = KeepAgentOnBoard(gym.make('pts/PushT-v0'))
    env.reset(seed=3)
    yield env
    env.close()


def meta(**kw):
    base = dict(demo_id='ab' * 16, session_id='s', condition='default', seed=3,
                variation={}, outcome='success', steps=0, seconds=1.0)
    base.update(kw)
    return DemoMeta(**base)


def make_demo(env, n=5, **meta_kw):
    buf = StepBuffer()
    buf.start(env)
    rng = np.random.default_rng(0)
    for t in range(n):
        raw = rng.uniform(-1, 1, 2).astype(np.float32)
        *_, info = env.step(raw)
        buf.add(env, raw, info, on_goal=False, t_wall=0.1 * t)
    return buf.to_demo(meta(**meta_kw))


# variations


def test_flatten_covers_every_variation_and_is_json(env):
    flat = variations.flatten(env)
    assert set(flat) == set(env.unwrapped.variation_space.names())
    assert json.loads(json.dumps(flat)) == flat


def test_flatten_roundtrips_through_exact_reset(env):
    env.reset(seed=5, options={'variation': ['all']})
    flat = variations.flatten(env)
    other = gym.make('pts/PushT-v0')
    other.reset(seed=0, options=variations.exact_reset_options(other, json.loads(json.dumps(flat))))
    assert variations.flatten(other) == flat


def test_cast_values_rejects_unknown_key(env):
    with pytest.raises(ValueError, match='block.colour'):
        variations.cast_values(env, {'block.colour': [1, 2, 3]})


# sim state


def test_sim_state_layout(env):
    s = capture_sim_state(env)
    assert s.shape == (len(SIM_STATE_FIELDS),)
    base = env.unwrapped
    np.testing.assert_allclose(s[:2], base.agent.position)
    np.testing.assert_allclose(s[8:10], base.block.position)
    # kinematic agent: the control columns are the agent itself
    np.testing.assert_allclose(s[4:6], s[:2])


def test_restore_sim_state_roundtrip(env):
    env.reset(seed=1, options={'variation_values': {'sim.damping': np.float64(1.0),
                                                    'sim.agent_max_force': np.float64(8000.0)}})
    for _ in range(5):
        env.step(np.array([0.5, 0.2], np.float32))
    snap = capture_sim_state(env)
    for _ in range(5):
        env.step(np.array([-0.5, 0.4], np.float32))
    restore_sim_state(env, snap)
    np.testing.assert_allclose(capture_sim_state(env), snap)


# StepBuffer / Demo


def test_step_buffer_shapes(env):
    d = make_demo(env, n=7)
    assert len(d) == 7 and d.meta.steps == 7
    assert d.state.shape == (8, 7) and d.sim_state.shape == (8, len(SIM_STATE_FIELDS))
    assert d.action.dtype == np.float32 and d.on_goal.dtype == bool
    assert d.meta.variation == variations.flatten(env)


def test_buffer_records_bounded_action(env):
    base = env.unwrapped
    base.agent.position = (495.0, 256.0)  # at the right wall
    buf = StepBuffer()
    buf.start(env)
    raw = np.array([1.0, 0.0], np.float32)
    *_, info = env.step(raw)
    buf.add(env, raw, info, False, 0.0)
    d = buf.to_demo(meta())
    assert d.raw_action[0, 0] == 1.0
    assert d.action[0, 0] < 0  # bounded: target pulled back inside the walls


def test_add_before_start_raises(env):
    with pytest.raises(RuntimeError):
        StepBuffer().add(env, np.zeros(2), {}, False, 0.0)


@pytest.mark.parametrize('field, bad', [
    ('state', np.zeros((3, 7))),
    ('action', np.full((5, 2), 1.5, np.float32)),
    ('t_wall', np.full(5, np.nan)),
])
def test_validate_catches_bad_arrays(env, field, bad):
    d = make_demo(env)
    setattr(d, field, bad)
    with pytest.raises(ValueError, match=field.split('_')[0]):
        d.validate()


def test_validate_checks_meta_steps(env):
    d = make_demo(env)
    d.meta.steps = 99
    with pytest.raises(ValueError, match='steps'):
        d.validate()


# files


def test_save_load_roundtrip(env, tmp_path):
    d = make_demo(env, reset_options={'variation': None, 'variation_values': {'block.shape': 1}})
    path = save_demo(d, tmp_path / 'sub' / 'x.npz')
    back = load_demo(path)
    assert back.meta == d.meta
    for name in demo_io.ARRAYS:
        np.testing.assert_array_equal(getattr(back, name), getattr(d, name))
        assert getattr(back, name).dtype == getattr(d, name).dtype


def test_save_is_atomic_and_leaves_no_temp(env, tmp_path):
    save_demo(make_demo(env), tmp_path / 'x.npz')
    assert [p.name for p in tmp_path.iterdir()] == ['x.npz']


def test_failed_save_leaves_nothing(env, tmp_path, monkeypatch):
    def boom(*a, **k):
        raise OSError('disk full')

    monkeypatch.setattr(np, 'savez_compressed', boom)
    with pytest.raises(OSError):
        save_demo(make_demo(env), tmp_path / 'x.npz')
    assert list(tmp_path.iterdir()) == []


def test_load_rejects_other_format_version(env, tmp_path):
    d = make_demo(env)
    d.meta.format_version = 99
    save_demo(d, tmp_path / 'x.npz')
    with pytest.raises(ValueError, match='format_version'):
        load_demo(tmp_path / 'x.npz')


def test_npz_has_no_pickles(env, tmp_path):
    save_demo(make_demo(env), tmp_path / 'x.npz')
    with np.load(tmp_path / 'x.npz', allow_pickle=False) as data:
        assert set(data.files) == set(demo_io.ARRAYS) | {'meta'}


def test_find_demos(env, tmp_path):
    for name in ('b/2.npz', 'a/1.npz', 'a/.tmp-x.npz'):
        save_demo(make_demo(env), tmp_path / name) if 'tmp' not in name else (tmp_path / name).touch()
    found = find_demos(tmp_path)
    assert [p.relative_to(tmp_path).as_posix() for p in found] == ['a/1.npz', 'b/2.npz']
    assert find_demos(found[0]) == [found[0]]
    with pytest.raises(ValueError):
        find_demos(tmp_path / 'notes.txt')


# sessions


def test_session_layout_and_events(env, tmp_path):
    out = OutputConfig(root=str(tmp_path), session_name='pilot', operator='me')
    with SessionWriter(out, {'seed': 0}) as s:
        p1 = s.write_demo(make_demo(env, demo_id='1' * 32, condition='heavy', seed=2))
        s.write_demo(make_demo(env, demo_id='2' * 32))
        assert s.undo_last() == '2' * 32
        assert s.undo_last() == '1' * 32
        assert s.undo_last() is None
    assert s.session_id.endswith('-pilot')
    assert p1.name == '0001_heavy_seed2_11111111.npz'
    info = yaml.safe_load((s.path / 'session.yaml').read_text())
    assert info['config'] == {'seed': 0} and info['operator'] == 'me'
    events = [e['event'] for e in read_events(s.path)]
    assert events == ['session_start', 'demo_saved', 'demo_saved', 'undo', 'undo', 'session_end']
    assert rejected_ids(s.path) == {'1' * 32, '2' * 32}
    assert session_dir_of(p1) == s.path.resolve()


def test_session_refuses_existing_dir(tmp_path):
    out = OutputConfig(root=str(tmp_path))
    with SessionWriter(out, {}, session_id='x'):
        pass
    with pytest.raises(FileExistsError):
        SessionWriter(out, {}, session_id='x').__enter__()


def test_session_end_logged_on_error(env, tmp_path):
    out = OutputConfig(root=str(tmp_path))
    with pytest.raises(KeyError):
        with SessionWriter(out, {}, session_id='x') as s:
            raise KeyError('crash')
    last = read_events(s.path)[-1]
    assert last['event'] == 'session_end' and 'crash' in last['error']


def test_sessions_started_in_the_same_second_get_distinct_dirs(tmp_path):
    out = OutputConfig(root=str(tmp_path), session_name='x')
    with SessionWriter(out, {}) as a, SessionWriter(out, {}) as b:
        assert a.path != b.path and a.path.exists() and b.path.exists()
