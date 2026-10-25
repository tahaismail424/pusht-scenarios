"""Parity between our forked env (pts/PushT-v0) and stable-worldmodel's (swm/PushT-v1).

The reference is the local ../stable-worldmodel clone (dev dependency), which is
the code env.py and spaces.py were copied from. Under identical seeds, reset
options and actions, both envs must produce bit-identical observations, info,
variation values and rendered frames. Any physics knob we add must leave this
passing when the knob is at its default.
"""

import gymnasium as gym
import numpy as np
import pytest

import pusht_scenarios.env  # noqa: F401  (registers pts/PushT-v0)

swm_envs = pytest.importorskip('stable_worldmodel.envs')

N_STEPS = 150
SEEDS = range(8)

# 'all_swm_variations' resamples every variation swm has: colors, shapes,
# scales, goal pose, start poses. Exercises the copied spaces.py as well as the
# env. Our 'sim' (physics) group isn't in swm, so it stays at its defaults here.
RESET_OPTIONS = ['default', 'all_swm_variations']


def reset_options(name, ref):
    if name == 'default':
        return None
    return {'variation': ref.unwrapped.variation_space.names()}


def make_pair():
    ours = gym.make('pts/PushT-v0')
    ref = gym.make('swm/PushT-v1')
    return ours, ref


def pushing_actions(env, rng, noise=0.3):
    """Steer the agent at the block (plus noise) so contacts actually happen."""
    base = env.unwrapped
    delta = np.asarray(base.block.position) - np.asarray(base.agent.position)
    action = delta / base.action_scale + rng.normal(0, noise, size=2)
    return np.clip(action, -1, 1).astype(np.float32)


def flatten(d, prefix=''):
    out = {}
    for k, v in d.items():
        key = f'{prefix}.{k}' if prefix else k
        if isinstance(v, dict):
            out.update(flatten(v, key))
        else:
            out[key] = v
    return out


def assert_same(a, b, what):
    if isinstance(a, dict):
        assert a.keys() == b.keys(), f'{what}: keys differ'
        for k in a:
            assert_same(a[k], b[k], f'{what}.{k}')
    else:
        np.testing.assert_array_equal(
            np.asarray(a), np.asarray(b), err_msg=what
        )


def assert_variations_same(ours, ref):
    ours_vals = flatten(ours.unwrapped.variation_space.value)
    ref_vals = flatten(ref.unwrapped.variation_space.value)
    shared = {k: ours_vals[k] for k in ref_vals}
    assert_same(shared, ref_vals, 'variation')


@pytest.mark.parametrize('options_name', RESET_OPTIONS)
@pytest.mark.parametrize('seed', SEEDS)
def test_rollout_parity(seed, options_name):
    ours, ref = make_pair()
    options = reset_options(options_name, ref)

    obs_a, info_a = ours.reset(seed=seed, options=options)
    obs_b, info_b = ref.reset(seed=seed, options=options)
    assert_same(obs_a, obs_b, 'reset obs')
    assert_same(info_a, info_b, 'reset info')
    assert_variations_same(ours, ref)
    assert_same(ours.render(), ref.render(), 'reset frame')

    rng = np.random.default_rng(1000 + seed)
    total_contacts = 0
    for t in range(N_STEPS):
        action = pushing_actions(ours, rng)
        out_a = ours.step(action)
        out_b = ref.step(action)
        # obs, reward, terminated, truncated, info
        for name, a, b in zip(
            ('obs', 'reward', 'terminated', 'truncated', 'info'), out_a, out_b
        ):
            assert_same(a, b, f'step {t} {name}')
        assert_same(ours.render(), ref.render(), f'step {t} frame')
        total_contacts += out_a[4]['n_contacts']

    # Guard against a vacuous test where the agent never touches the block.
    assert total_contacts > 0, 'rollout never made contact with the block'
    ours.close()
    ref.close()


def test_spaces_match():
    ours, ref = make_pair()
    assert ours.observation_space == ref.observation_space
    assert ours.action_space == ref.action_space
    ours_names = ours.unwrapped.variation_space.names()
    ref_names = ref.unwrapped.variation_space.names()
    # ours = swm's variations + our sim (physics) group, nothing else
    assert [n for n in ours_names if not n.startswith('sim.')] == ref_names


def test_physics_defaults_are_swm():
    """Default physics values must encode swm's hardcoded physics."""
    env = gym.make('pts/PushT-v0')
    env.reset(seed=0)
    physics = env.unwrapped.variation_space['sim']
    expected = {
        'damping': 0.0,
        'contact_friction': 0.0,
        'block_mass': 1.0,
        'block_cog_offset': [0.0, 0.0],
        'floor_friction': 0.0,
        'angular_friction': 0.0,
        'agent_max_force': 0.0,
    }
    assert set(physics.spaces) == set(expected)
    for name, value in expected.items():
        np.testing.assert_array_equal(physics[name].value, value, err_msg=name)


@pytest.mark.parametrize('seed', [0, 1])
def test_deterministic(seed):
    """Same seed and actions twice in our env -> identical trajectories."""
    runs = []
    for _ in range(2):
        env = gym.make('pts/PushT-v0')
        env.reset(seed=seed, options={'variation': ['all']})  # incl. sim (physics)
        rng = np.random.default_rng(seed)
        states = [env.step(pushing_actions(env, rng))[0]['state'] for _ in range(N_STEPS)]
        runs.append(np.stack(states))
        env.close()
    np.testing.assert_array_equal(runs[0], runs[1])
