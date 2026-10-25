"""Rendered pixels must not leak hidden factors or UI: hidden physics renders
exactly like default physics, and the HUD never reaches render() output."""

import gymnasium as gym
import numpy as np
import pygame
import pytest

import pusht_scenarios.env  # noqa: F401
from pusht_scenarios.teleop.sandbox import draw_hud

HIDDEN = {
    'strength': {'sim.agent_max_force': 8000.0},
    'mass': {'sim.agent_max_force': 8000.0, 'sim.block_mass': 3.0},
    'floor': {'sim.damping': 1.0, 'sim.floor_friction': 1.0},
    'angular': {'sim.damping': 1.0, 'sim.angular_friction': 2.0},
    'contact': {'sim.contact_friction': 1.5},
    'cog': {'sim.block_cog_offset': [20.0, -10.0]},
    'everything': {'sim.agent_max_force': 8000.0, 'sim.block_mass': 3.0, 'sim.damping': 1.0,
                   'sim.floor_friction': 1.0, 'sim.angular_friction': 1.0,
                   'sim.contact_friction': 1.0, 'sim.block_cog_offset': [10.0, 10.0]},
}
STATE = np.array([150.0, 380.0, 300.0, 220.0, 0.6, 0.0, 0.0])


def frame(env, values, state=STATE, seed=3):
    cast = {k: (np.asarray(v, np.float64) if isinstance(v, list) else np.float64(v)) for k, v in values.items()}
    env.reset(seed=seed, options={'variation': [], 'variation_values': cast, 'state': state})
    return env.render()


@pytest.mark.parametrize('name', list(HIDDEN))
def test_hidden_physics_renders_like_default(name):
    env = gym.make('pts/PushT-v0')
    np.testing.assert_array_equal(frame(env, HIDDEN[name]), frame(env, {}))


@pytest.mark.parametrize('name', ['strength', 'floor', 'everything'])
def test_no_joint_markers_mid_episode(name):
    # same pose reached under different physics -> same pixels (no joint dots)
    env = gym.make('pts/PushT-v0')
    frame(env, HIDDEN[name])
    for _ in range(5):
        env.step(np.array([0.6, -0.4], np.float32))
    pose = env.unwrapped._get_obs()
    np.testing.assert_array_equal(frame(env, HIDDEN[name], state=pose), frame(env, {}, state=pose))


def test_hud_is_not_in_rendered_frames():
    pygame.init()
    human = gym.make('pts/PushT-v0', render_mode='human')
    plain = gym.make('pts/PushT-v0')
    a, b = frame(human, {}), frame(plain, {})
    draw_hud(human, 'mass-heavy.friction-low  step 3/400  PAUSE QUEUED')
    np.testing.assert_array_equal(human.render(), b)
    np.testing.assert_array_equal(a, b)
    human.close()
    pygame.quit()


def contact_dot_pixels(frame) -> int:
    """Pixels close to pymunk's contact-point color (red, unlike any scene color)."""
    import pygame
    from pusht_scenarios.env.utils import DrawOptions

    c = DrawOptions(pygame.Surface((1, 1))).collision_point_color
    target = np.array([c.r, c.g, c.b], dtype=float)
    return int((np.linalg.norm(frame.astype(float) - target, axis=-1) < 60).sum())


def test_no_contact_dots_by_default():
    # during a push, swm-style rendering (render_contacts=True) draws red
    # contact dots; the default rendering must never contain them
    env = gym.make('pts/PushT-v0')
    swm_like = gym.make('pts/PushT-v0', render_contacts=True)
    values = {'agent.start_position': np.array([150.0, 270.0]),
              'block.start_position': np.array([256.0, 256.0]), 'block.angle': 0.0}
    for e in (env, swm_like):
        e.reset(seed=0, options={'variation': [], 'variation_values': values})
    dotted = clean = 0
    for _ in range(20):
        env.step(np.array([0.6, 0.0], np.float32))
        swm_like.step(np.array([0.6, 0.0], np.float32))
        clean += contact_dot_pixels(env.render())
        dotted += contact_dot_pixels(swm_like.render())
    assert dotted > 0  # the dots really are drawn when enabled
    assert clean == 0
