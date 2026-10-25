"""The agent must stay on the board under any physics (env/bounds.py)."""

import gymnasium as gym
import numpy as np
import pytest

import pusht_scenarios.env  # noqa: F401
from pusht_scenarios.env.bounds import WALL_INNER, KeepAgentOnBoard, agent_radius, bound_action

CORNERS = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1)]
PHYSICS = {
    'swm': {},
    'strength': {'sim.agent_max_force': 5000.0},
    'ice': {'sim.damping': 1.0},
}


@pytest.fixture
def env():
    env = KeepAgentOnBoard(gym.make('pts/PushT-v0'))
    yield env
    env.close()


def reset(env, agent_scale=40.0, **sim):
    values = {
        'agent.start_position': np.array([256.0, 256.0]),
        'block.start_position': np.array([120.0, 120.0]),
        'agent.scale': agent_scale,
    }
    values.update({k: float(v) for k, v in sim.items()})
    env.reset(seed=0, options={'variation': [], 'variation_values': values})
    return env.unwrapped


@pytest.mark.parametrize('physics', list(PHYSICS))
@pytest.mark.parametrize('direction', CORNERS)
def test_agent_never_leaves_board(env, physics, direction):
    base = reset(env, **PHYSICS[physics])
    r = agent_radius(base)
    for _ in range(40):
        env.step(np.array(direction, np.float32))
        x, y = base.agent.position
        assert WALL_INNER[0] + r - 0.5 <= x <= WALL_INNER[1] - r + 0.5
        assert WALL_INNER[0] + r - 0.5 <= y <= WALL_INNER[1] - r + 0.5


def test_larger_agent_gets_larger_margin(env):
    small = agent_radius(reset(env, agent_scale=20.0))
    large = agent_radius(reset(env, agent_scale=60.0))
    assert large == pytest.approx(3 * small)


def test_interior_actions_unchanged(env):
    base = reset(env)
    action = np.array([0.6, -0.3], np.float32)
    np.testing.assert_allclose(bound_action(base, action), action, atol=1e-6)


def test_info_reports_applied_action(env):
    base = reset(env)
    base.agent.position = (490.0, 256.0)
    *_, info = env.step(np.array([1.0, 0.0], np.float32))
    assert info['applied_action'][0] < 0.1  # shrunk: target clamped at the right wall
    assert env.action_space.contains(info['applied_action'])


def test_bounded_actions_replay_in_plain_swm_env():
    # the recorded (bounded) actions keep the agent on the board even when
    # replayed in an unwrapped env, which is what swm/LeWM eval would use
    wrapped, plain = KeepAgentOnBoard(gym.make('pts/PushT-v0')), gym.make('pts/PushT-v0')
    opts = {'variation': [], 'variation_values': {'agent.start_position': np.array([256.0, 256.0]),
                                                  'block.start_position': np.array([120.0, 120.0])}}
    wrapped.reset(seed=0, options=opts)
    plain.reset(seed=0, options=opts)
    for _ in range(30):
        *_, info = wrapped.step(np.array([1.0, 1.0], np.float32))
        plain.step(info['applied_action'])
    np.testing.assert_allclose(plain.unwrapped.agent.position, wrapped.unwrapped.agent.position)
    assert plain.unwrapped.agent.position[0] < WALL_INNER[1]
