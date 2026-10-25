"""Env hooks for evaluation: reset_options_from_dataset, the bounded env id,
and swm's dataset-driven eval running on an exported dataset."""

import gymnasium as gym
import h5py
import numpy as np
import pytest

import pusht_scenarios.env  # noqa: F401
from pusht_scenarios.env import variations
from pusht_scenarios.env.bounds import WALL_INNER, agent_radius
from pusht_scenarios.export import hdf5 as X

from test_export import mini_spec


@pytest.fixture(scope='module')
def exported(mini, tmp_path_factory):
    out = tmp_path_factory.mktemp('export_eval')
    X.run_export(mini_spec(mini, out), verbose=False)
    return out


def rows(path, ep, start=0, offset=5):
    with h5py.File(path) as f:
        o = f['ep_offset'][ep]
        init = {c: f[c][o + start] for c in f if c not in ('ep_len', 'ep_offset', 'pixels')}
        goal = {f'goal_{c}': f[c][o + start + offset] for c in ('state', 'proprio')}
    return init, goal


def test_reset_options_from_dataset_rebuilds_the_episode(exported, mini):
    path = exported / 'mini_test_eval.h5'
    env = gym.make('pts/PushT-v0')
    init, goal = rows(path, 0, start=1)
    env.reset(seed=int(init['seed']), options=env.unwrapped.reset_options_from_dataset(init, goal))
    base = env.unwrapped
    np.testing.assert_allclose(base._get_obs(), init['state'])
    np.testing.assert_allclose(base.goal_state, goal['goal_state'])
    flat = variations.flatten(env)
    for key, value in init.items():
        if key.startswith('variation.'):
            np.testing.assert_allclose(flat[key[len('variation.'):]], value)
    # night palette / normal mass etc. actually applied
    cond = next(e for e in mini['suite'].envs if e.env_id == int(init['env_id']))
    for key, value in cond.variation_values.items():
        if not key.endswith('start_position') and key != 'block.angle':
            np.testing.assert_allclose(flat[key], value)


def test_start_pose_comes_from_state_not_variation(exported):
    env = gym.make('pts/PushT-v0')
    init, goal = rows(exported / 'mini_train.h5', 2, start=4)
    init['variation.block.start_position'] = np.array([111.0, 111.0])  # must be ignored
    env.reset(seed=0, options=env.unwrapped.reset_options_from_dataset(init, goal))
    # swm's _set_state takes one physics step after placing the bodies, so a
    # mid-episode state in contact can move a hair; (111, 111) must not be used
    np.testing.assert_allclose(env.unwrapped._get_obs()[2:4], init['state'][2:4], atol=0.5)


def test_bounded_env_id(mini):
    env = gym.make('pts/PushTBounded-v0')
    env.reset(seed=0, options={'variation_values': {'sim.agent_max_force': np.float64(0.0)}})
    for _ in range(40):
        *_, info = env.step(np.array([1.0, 1.0], np.float32))
    r = agent_radius(env.unwrapped)
    assert max(env.unwrapped.agent.position) <= WALL_INNER[1] - r + 0.5
    assert 'applied_action' in info and 'action' not in info  # swm's wrappers own info['action']
    plain = gym.make('pts/PushT-v0')
    plain.reset(seed=0)
    for _ in range(40):
        plain.step(np.array([1.0, 1.0], np.float32))
    assert max(plain.unwrapped.agent.position) > 600  # swm's agent leaves the board


def test_swm_dataset_eval_runs_on_export(exported, mini):
    import stable_worldmodel as swm
    from stable_worldmodel.data.formats.hdf5 import HDF5Dataset
    from stable_worldmodel.policy import RandomPolicy

    path = exported / 'mini_test_eval.h5'
    ds = HDF5Dataset(path=path, num_steps=1)
    n = min(3, len(ds.lengths))
    world = swm.World('pts/PushTBounded-v0', num_envs=n, image_shape=(224, 224), max_episode_steps=30)
    world.set_policy(RandomPolicy(seed=0))
    result = world.evaluate(dataset=ds, episodes_idx=list(range(n)), start_steps=[0] * n,
                            goal_offset=3, eval_budget=3)
    assert 'success_rate' in result and len(result['episode_successes']) == n
    with h5py.File(path) as f:
        off = f['ep_offset'][:]
        for i in range(n):
            env = world.envs.envs[i].unwrapped
            flat = variations.flatten(env)
            assert flat['sim.block_mass'] == f['variation.sim.block_mass'][off[i]]
            assert flat['block.color'] == list(f['variation.block.color'][off[i]])
            np.testing.assert_allclose(env.goal_state, f['state'][off[i] + 3])
    assert result['seeds'] == [int(s) for s in h5py.File(path)['seed'][:][off[:n]]]
