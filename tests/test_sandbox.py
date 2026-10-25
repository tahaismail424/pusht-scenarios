"""Headless sandbox tests: config loading, success check, and the round loop
driven by scripted events and simulated input (SDL dummy driver, no window)."""

from pathlib import Path

import gymnasium as gym
import numpy as np
import pygame
import pytest

import pusht_scenarios.env  # noqa: F401
from pusht_scenarios.env import task
from pusht_scenarios.teleop import sandbox
from pusht_scenarios.teleop.input import InputDevice
from pusht_scenarios.teleop.sandbox import Condition, SandboxConfig

EXAMPLE = Path(__file__).parents[1] / 'configs' / 'sandbox.yaml'
GOAL = [256.0, 256.0]  # swm's default goal pose
GOAL_ANGLE = np.pi / 4


class FakeJoystick:
    def __init__(self, x=0.0, y=0.0):
        self.axes = [x, y]

    def get_axis(self, i):
        return self.axes[i]

    def get_button(self, i):
        return False


def scripted(*frames):
    """get_events stand-in: frame k returns frames[k], then nothing forever."""
    frames = list(frames)

    def get_events():
        return frames.pop(0) if frames else []

    return get_events


def key(k):
    return pygame.event.Event(pygame.KEYDOWN, key=k)


@pytest.fixture(scope='module', autouse=True)
def pygame_init():
    pygame.init()
    yield
    pygame.quit()


@pytest.fixture
def env():
    env = gym.make('pts/PushT-v0', render_mode='human')
    yield env
    env.close()


def fast(**kw):
    return SandboxConfig(fps=0, wait_for_input=False, **kw)


ON_GOAL = Condition(
    'on_goal',
    variation=[],
    variation_values={'block.start_position': GOAL, 'block.angle': GOAL_ANGLE},
)


# config


def test_example_config_loads():
    config = sandbox.load_config(EXAMPLE)
    assert [c.name for c in config.conditions] == ['default', 'red_L', 'random_goal']
    assert config.device['gain'] == 0.6


def test_default_config():
    config = sandbox.load_config(None)
    assert [c.name for c in config.conditions] == ['default']


def test_unknown_key_rejected(tmp_path):
    path = tmp_path / 'bad.yaml'
    path.write_text('rounds: 3\n')  # typo for rounds_per_condition
    with pytest.raises(ValueError, match='rounds'):
        sandbox.load_config(path)


def test_example_conditions_reset(env):
    # every example condition must produce valid reset options for the env
    for condition in sandbox.load_config(EXAMPLE).conditions:
        env.reset(seed=0, options=sandbox.reset_options(env, condition))


def test_reset_options_cast_yaml_values(env):
    condition = Condition(
        variation_values={'block.shape': 1, 'block.color': [220, 60, 60]}
    )
    env.reset(seed=0, options=sandbox.reset_options(env, condition))
    block = env.unwrapped.variation_space['block']
    assert block['shape'].value == 1
    np.testing.assert_array_equal(block['color'].value, [220, 60, 60])


def test_unknown_variation_key_rejected(env):
    with pytest.raises(ValueError, match='block.colour'):
        sandbox.reset_options(env, Condition(variation_values={'block.colour': [1, 2, 3]}))


# success check


def place_block(env, shape, xy, angle):
    env.reset(
        seed=0,
        options={
            'variation': [],
            'variation_values': {
                'block.shape': shape,
                'block.start_position': np.array(xy, dtype=np.float64),
                'block.angle': angle,
            },
        },
    )


def test_block_on_goal(env):
    place_block(env, 2, GOAL, GOAL_ANGLE)
    assert task.block_on_goal(env)
    place_block(env, 2, [GOAL[0] + 30, GOAL[1]], GOAL_ANGLE)
    assert not task.block_on_goal(env)


def test_block_on_goal_uses_shape_symmetry(env):
    # a square rotated 90 degrees still covers its outline; a T does not
    place_block(env, 4, GOAL, GOAL_ANGLE + np.pi / 2)
    assert task.block_on_goal(env)
    place_block(env, 2, GOAL, GOAL_ANGLE + np.pi / 2)
    assert not task.block_on_goal(env)


# round loop


def zero_device():
    return InputDevice(joystick=FakeJoystick())


def test_success_after_hold_steps(env):
    config = fast(hold_steps=5)
    r = sandbox.run_round(env, zero_device(), ON_GOAL, 0, config)
    assert (r.outcome, r.steps) == ('success', 5)


def test_timeout(env):
    r = sandbox.run_round(env, zero_device(), Condition(), 0, fast(max_steps=7))
    assert (r.outcome, r.steps) == ('timeout', 7)


@pytest.mark.parametrize(
    'event, outcome',
    [
        (key(pygame.K_r), 'retry'),
        (key(pygame.K_n), 'skip'),
        (key(pygame.K_q), 'quit'),
        (key(pygame.K_ESCAPE), 'quit'),
        (pygame.event.Event(pygame.QUIT), 'quit'),
    ],
)
def test_keys(env, event, outcome):
    r = sandbox.run_round(
        env, zero_device(), Condition(), 0, fast(), get_events=scripted([], [], [event])
    )
    assert (r.outcome, r.steps) == (outcome, 2)


def test_pause_stops_stepping(env):
    events = scripted([key(pygame.K_SPACE)], [], [], [key(pygame.K_q)])
    r = sandbox.run_round(env, zero_device(), Condition(), 0, fast(), get_events=events)
    assert (r.outcome, r.steps) == ('quit', 0)


def test_waits_for_input(env):
    # idle stick: the round never starts, so no steps are taken
    config = SandboxConfig(fps=0, wait_for_input=True)
    events = scripted(*([[]] * 10), [key(pygame.K_q)])
    r = sandbox.run_round(env, zero_device(), Condition(), 0, config, get_events=events)
    assert (r.outcome, r.steps) == ('quit', 0)


def test_stick_input_moves_agent_in_round(env):
    device = InputDevice(joystick=FakeJoystick(1.0, 0.0))
    config = SandboxConfig(fps=0, wait_for_input=True, max_steps=5)
    r = sandbox.run_round(env, device, Condition(), 0, config)
    assert (r.outcome, r.steps) == ('timeout', 5)
    env.reset(seed=0)
    start_x = env.unwrapped.agent.position[0]
    sandbox.run_round(env, device, Condition(), 0, config)
    assert env.unwrapped.agent.position[0] > start_x + 10


def test_session_order_and_retry(env):
    # condition a: seed 0 retried once, then seeds 0, 1; condition b: seeds 0, 1
    config = fast(
        conditions=[ON_GOAL, Condition('b', variation=[], variation_values=ON_GOAL.variation_values)],
        rounds_per_condition=2,
    )
    events = scripted([key(pygame.K_r)])
    results = sandbox.run_session(config, env=env, device=zero_device(), get_events=events)
    assert [(r.condition, r.seed, r.outcome) for r in results] == [
        ('on_goal', 0, 'retry'),
        ('on_goal', 0, 'success'),
        ('on_goal', 1, 'success'),
        ('b', 0, 'success'),
        ('b', 1, 'success'),
    ]
    assert 'success 4/4' in sandbox.summarize(results)


def test_main_runs_headless(tmp_path, capsys):
    path = tmp_path / 'quick.yaml'
    path.write_text(
        'rounds_per_condition: 1\nmax_steps: 3\nfps: 0\nwait_for_input: false\n'
        'device: {use_hid: false}\n'
        'conditions:\n  - name: default\n  - name: square\n'
        '    variation_values: {block.shape: 4}\n'
    )
    results = sandbox.main(['-c', str(path)])
    assert [r.outcome for r in results] == ['timeout', 'timeout']
    assert 'input: mouse' in capsys.readouterr().out
