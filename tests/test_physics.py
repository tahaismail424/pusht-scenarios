"""Effect tests for the sim.* physics knobs (env.PushT._apply_sim).

Each test runs one scripted push: the agent starts left of a T and pushes
right, slightly below center so the block also turns (8 steps), then backs off
(12 steps). Defaults are covered by tests/test_parity.py; here each knob must
change the dynamics in the expected direction.
"""

import gymnasium as gym
import numpy as np
import pytest

import pusht_scenarios.env  # noqa: F401

PUSH, RELEASE = 8, 12


@pytest.fixture(scope='module')
def env():
    env = gym.make('pts/PushT-v0')
    yield env
    env.close()


def reset(env, **sim):
    values = {
        'agent.start_position': np.array([150.0, 270.0]),
        'block.start_position': np.array([256.0, 256.0]),
        'block.angle': 0.0,
    }
    for k, v in sim.items():
        values[f'sim.{k}'] = np.array(v, dtype=np.float64) if np.ndim(v) else float(v)
    env.reset(seed=0, options={'variation': [], 'variation_values': values})
    return env.unwrapped


def push(env, **sim):
    """-> dict with block pose after the push and after release."""
    base = reset(env, **sim)
    b = base.block
    for _ in range(PUSH):
        env.step(np.array([0.6, 0.0], np.float32))
    pushed = np.array([*b.position, b.angle])
    for _ in range(RELEASE):
        env.step(np.array([-0.6, 0.0], np.float32))
    end = np.array([*b.position, b.angle])
    return {
        'pushed': pushed,
        'end': end,
        'turn': np.degrees(pushed[2]),  # degrees turned while pushed
        'drift': float(np.linalg.norm(end[:2] - pushed[:2])),  # px moved after release
        'spin': float(b.angular_velocity),
    }


@pytest.fixture(scope='module')
def default(env):
    return push(env)


# structure


def test_defaults_add_nothing(env):
    base = reset(env)
    assert len(base.space.constraints) == 0
    assert base.space.damping == 0
    assert all(s.friction == 0 for s in base.block.shapes)


def test_contact_friction_sets_both_sides(env):
    base = reset(env, contact_friction=0.7)
    assert all(s.friction == 0.7 for s in base.block.shapes)
    assert all(s.friction == 1.0 for s in base.agent.shapes)
    assert all(s.friction == 1.0 for s in base.space.static_body.shapes)  # walls


def test_mass_scales_inertia(env):
    m1 = reset(env).block.moment
    block = reset(env, block_mass=4.0).block
    assert block.mass == 4.0
    assert block.moment == pytest.approx(4.0 * m1)


def test_cog_offset_is_relative_to_shape_cog(env):
    cog = np.array(reset(env).block.center_of_gravity)
    moved = np.array(reset(env, block_cog_offset=[10.0, -5.0]).block.center_of_gravity)
    np.testing.assert_allclose(moved - cog, [10.0, -5.0])


def test_friction_joints_added(env):
    assert len(reset(env, floor_friction=1.0).space.constraints) == 2  # slide + spin
    assert len(reset(env, angular_friction=1.0).space.constraints) == 1  # spin only


def test_set_state_leaves_block_at_rest(env):
    # with momentum on, reset's teleports must not leave the block moving
    base = reset(env, damping=1.0)
    assert base.block.velocity.length == 0 and base.block.angular_velocity == 0


# effects


def test_default_has_no_momentum(default):
    assert default['drift'] < 5


def test_contact_friction_changes_turning(env, default):
    assert abs(push(env, contact_friction=1.0)['turn'] - default['turn']) > 5


def test_cog_offset_changes_turning_by_direction(env, default):
    # the push contact is below center: moving the CoG toward it turns the T
    # more, away from it less
    toward = push(env, block_cog_offset=[0.0, 25.0])['turn']
    away = push(env, block_cog_offset=[0.0, -25.0])['turn']
    assert toward > default['turn'] + 10
    assert away < default['turn'] - 10


def test_damping_one_gives_momentum(env, default):
    assert push(env, damping=1.0)['drift'] > 10 * default['drift']


def test_floor_friction_shortens_slide(env):
    slides = [push(env, damping=1.0, floor_friction=mu)['drift'] for mu in (0.0, 0.5, 2.0)]
    assert slides[0] > slides[1] > slides[2]


def free_spin(env, **sim):
    """Spin the block at 3 rad/s with the agent far away; -> rad turned in 1 s."""
    base = reset(env, **sim)
    base.agent.position = (60.0, 450.0)  # out of the way
    base.block.angular_velocity = 3.0
    start = base.block.angle
    for _ in range(10):
        env.step(np.zeros(2, np.float32))
    return base.block.angle - start


def test_floor_friction_brakes_spin(env):
    # one friction point at the CoG can't stop spinning; the paired gear joint must
    free = free_spin(env, damping=1.0)
    braked = free_spin(env, damping=1.0, floor_friction=1.0)
    assert free > 2.5  # undamped: keeps spinning (~3 rad in 1 s)
    assert braked < 0.5 * free


def test_angular_friction_resists_turning(env):
    free = push(env, damping=1.0)['turn']
    resisted = push(env, damping=1.0, angular_friction=2.0)['turn']
    assert abs(resisted) < 0.5 * abs(free)


def test_mass_has_no_effect_with_unlimited_agent(env):
    # Expected physics, not a bug: swm's kinematic agent has infinite strength,
    # and friction caps scale with mass, so mass cancels out. Mass only shows
    # with finite agent strength (test_heavier_block_pushes_slower).
    for sim in ({}, {'damping': 1.0, 'floor_friction': 1.0}):
        light = push(env, **sim)['end']
        heavy = push(env, block_mass=10.0, **sim)['end']
        np.testing.assert_allclose(light, heavy, atol=1e-6)


def test_deterministic_with_all_knobs(env):
    sim = dict(damping=1.0, contact_friction=0.5, block_cog_offset=[5.0, 5.0],
               block_mass=2.0, floor_friction=0.5, angular_friction=0.5,
               agent_max_force=8000.0)
    np.testing.assert_array_equal(push(env, **sim)['end'], push(env, **sim)['end'])


# agent strength (sim.agent_max_force)


def pushed_dx(env, **sim):
    return push(env, **sim)['pushed'][0] - 256.0


def test_strength_structure(env):
    base = reset(env)
    assert base.agent.body_type == base.agent.KINEMATIC and base.agent_driver is base.agent

    base = reset(env, agent_max_force=5000.0)
    assert base.agent.body_type == base.agent.DYNAMIC
    control = base.agent_driver
    assert control is not base.agent and control.body_type == control.KINEMATIC
    assert control.shapes == set()  # invisible: nothing to render
    (leash,) = base.space.constraints
    assert leash.max_force == 5000.0 and {leash.a, leash.b} == {control, base.agent}


def test_set_state_moves_control_body(env):
    base = reset(env, agent_max_force=5000.0)
    np.testing.assert_allclose(base.agent_driver.position, base.agent.position)
    np.testing.assert_allclose(base.agent.position, [150.0, 270.0])


def test_strength_free_motion_matches_swm(env):
    # in open space a weak agent moves like swm's: block feels heavy, the agent
    # doesn't feel sluggish (only the first step from rest lags slightly)
    def xs(**sim):
        base = reset(env, **sim)
        base.block.position = (256.0, 450.0)  # out of the way
        out = []
        for _ in range(8):
            env.step(np.array([0.6, 0.0], np.float32))
            out.append(base.agent.position[0])
        return np.diff(out)[2:]  # per-step cruise displacement

    np.testing.assert_allclose(xs(agent_max_force=1000.0), xs(), atol=1.0)


def test_strong_agent_approaches_swm(env):
    assert pushed_dx(env, agent_max_force=1e5) == pytest.approx(pushed_dx(env), rel=0.05)


def test_heavier_block_pushes_slower(env):
    dx = [pushed_dx(env, agent_max_force=20000.0, block_mass=m) for m in (1.0, 3.0, 10.0)]
    assert dx[0] > 1.5 * dx[1] > 1.5 * 1.5 * dx[2]


def test_weaker_agent_pushes_slower(env):
    dx = [pushed_dx(env, agent_max_force=f, block_mass=3.0) for f in (1e5, 20000.0, 5000.0)]
    assert dx[0] > 1.5 * dx[1] > 1.5 * 1.5 * dx[2]


def test_too_heavy_to_move_against_floor_friction(env):
    # Coulomb threshold: mu * m * g = 1 * 10 * 1000 = 10000 > 5000 of strength
    sim = dict(agent_max_force=5000.0, damping=1.0, floor_friction=1.0)
    assert pushed_dx(env, block_mass=3.0, **sim) > 100  # 3000 < 5000: moves
    assert abs(pushed_dx(env, block_mass=10.0, **sim)) < 2  # can't budge


def test_dynamic_agent_stopped_by_walls(env):
    base = reset(env, agent_max_force=5000.0)
    base.block.position = (256.0, 450.0)
    for _ in range(20):
        env.step(np.array([1.0, 0.0], np.float32))
    # right wall at x=506 (2 px thick), agent radius 15
    assert base.agent.position[0] < 506 - 15 + 1
