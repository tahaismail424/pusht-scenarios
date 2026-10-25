"""InputDevice tests with simulated joystick and mouse (no hardware needed)."""

import gymnasium as gym
import numpy as np
import pytest

import pusht_scenarios.env  # noqa: F401  (registers pts/PushT-v0)
from pusht_scenarios.teleop.input import InputDevice


class FakeJoystick:
    def __init__(self, x=0.0, y=0.0, buttons=()):
        self.axes = [x, y]
        self.buttons = set(buttons)

    def get_axis(self, i):
        return self.axes[i]

    def get_button(self, i):
        return i in self.buttons


@pytest.fixture
def env():
    env = gym.make('pts/PushT-v0')
    env.reset(seed=0)
    yield env
    env.close()


def stick(x, y, **kwargs):
    return InputDevice(joystick=FakeJoystick(x, y, kwargs.pop('buttons', ())), **kwargs)


def test_no_joystick_falls_back_to_mouse():
    # SDL_VIDEODRIVER=dummy (conftest) exposes no joysticks; use_hid=False so a
    # real HID gamepad plugged into this machine doesn't change the result
    assert InputDevice(use_hid=False).kind == 'mouse'
    assert stick(0, 0).kind == 'gamepad'


def test_action_dtype_and_shape(env):
    for device in (stick(0.5, -0.5), InputDevice(use_hid=False, mouse_pos=lambda: (10, 10))):
        action = device.read(env)
        assert action.shape == (2,) and action.dtype == np.float32
        assert env.action_space.contains(action)


@pytest.mark.parametrize('x, y', [(0.0, 0.0), (0.05, -0.05), (0.07, 0.07)])
def test_dead_zone(env, x, y):
    np.testing.assert_array_equal(stick(x, y, stick_thresh=0.1).read(env), [0, 0])


def test_dead_zone_is_radial(env):
    # per-axis thresholds would zero the y component here
    action = stick(0.8, 0.09, stick_thresh=0.1).read(env)
    assert action[1] > 0


def test_full_tilt_equals_gain(env):
    action = stick(1.0, 0.0, gain=0.3).read(env)
    np.testing.assert_allclose(action, [0.3, 0.0], atol=1e-6)


def test_corner_magnitude_capped_at_gain(env):
    # a square-gated stick reports (1, 1) in the corner, magnitude sqrt(2)
    action = stick(1.0, 1.0, gain=0.3).read(env)
    np.testing.assert_allclose(np.linalg.norm(action), 0.3, atol=1e-6)


def test_output_continuous_at_dead_zone_edge(env):
    just_outside = stick(0.1001, 0.0, stick_thresh=0.1).read(env)
    assert np.linalg.norm(just_outside) < 1e-3


def test_precision_button(env):
    normal = stick(1.0, 0.0, gain=0.3, precision_gain=0.1, precision_button=4)
    held = stick(1.0, 0.0, gain=0.3, precision_gain=0.1, precision_button=4, buttons={4})
    np.testing.assert_allclose(normal.read(env), [0.3, 0.0], atol=1e-6)
    np.testing.assert_allclose(held.read(env), [0.1, 0.0], atol=1e-6)


def test_mouse_is_relative_to_agent(env):
    agent = np.array(env.unwrapped.agent.position)
    device = InputDevice(use_hid=False, mouse_pos=lambda: tuple(agent + [50, -20]))
    np.testing.assert_allclose(device.read(env), [0.5, -0.2], atol=1e-6)


def test_mouse_far_away_is_clipped(env):
    device = InputDevice(use_hid=False, mouse_pos=lambda: (10_000, -10_000))
    np.testing.assert_array_equal(device.read(env), [1, -1])


# End to end: the action actually moves the agent the intended way in the env.


@pytest.mark.parametrize(
    'x, y, axis, sign',
    [(1, 0, 0, +1), (-1, 0, 0, -1), (0, 1, 1, +1), (0, -1, 1, -1)],
)
def test_stick_moves_agent(env, x, y, axis, sign):
    # stick down (+y in pygame) must move the agent down the screen (+y world)
    device = stick(x, y)
    start = np.array(env.unwrapped.agent.position)
    for _ in range(3):
        env.step(device.read(env))
    moved = np.array(env.unwrapped.agent.position) - start
    assert sign * moved[axis] > 10
    assert abs(moved[1 - axis]) < 1


def test_mouse_agent_converges_to_cursor(env):
    target = (150.0, 150.0)
    device = InputDevice(use_hid=False, mouse_pos=lambda: target)
    for _ in range(30):
        env.step(device.read(env))
    # the block may get in the way, but the agent should end up near the cursor
    assert np.linalg.norm(np.array(env.unwrapped.agent.position) - target) < 30
