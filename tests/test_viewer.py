"""pts-replay viewer, headless with scripted keys."""

import gymnasium as gym
import pygame
import pytest

import pusht_scenarios.env  # noqa: F401
from pusht_scenarios.env.bounds import KeepAgentOnBoard
from pusht_scenarios.io.demo import load_demo
from pusht_scenarios.teleop import record, viewer
from pusht_scenarios.teleop.input import InputDevice


class Stick:
    def __init__(self, x, y):
        self.axes = [x, y]

    def get_axis(self, i):
        return self.axes[i]

    def get_button(self, i):
        return False


def key(k):
    return pygame.event.Event(pygame.KEYDOWN, key=k)


@pytest.fixture(scope='module', autouse=True)
def pygame_init():
    pygame.init()
    yield
    pygame.quit()


@pytest.fixture(scope='module')
def session(tmp_path_factory):
    root = tmp_path_factory.mktemp('rec')
    env = KeepAgentOnBoard(gym.make('pts/PushT-v0', render_mode='human'))
    config = record.record_config_from_dict({
        'rounds_per_condition': 1, 'max_steps': 12, 'fps': 0, 'wait_for_input': False,
        'output': {'root': str(root)},
        'conditions': [{'name': 'a'}, {'name': 'b', 'variation_values': {'sim.damping': 1.0}},
                       {'name': 'c', 'variation_values': {'block.shape': 7}}],
    })
    frames = [[]] * 30 + [[pygame.event.Event(pygame.KEYDOWN, key=pygame.K_u)]]
    events = lambda: frames.pop(0) if frames else []  # noqa: E731
    s, _ = record.record(config, env=env, device=InputDevice(joystick=Stick(0.8, 0.3)),
                         get_events=events, verbose=False)
    env.close()
    return s


def fast_view(paths, frames=(), speed=0):
    frames = list(frames)
    return viewer.view(paths, speed=speed, verbose=False,
                       get_events=lambda: frames.pop(0) if frames else [])


def test_plays_every_demo_exactly(session):
    results = fast_view([session.path])
    assert len(results) == 3
    assert all(r.finished and r.max_error == 0.0 for r in results)


def test_single_file(session):
    (_, path), *_ = session.saved
    (r,) = fast_view([path])
    assert r.finished and r.steps_shown == len(load_demo(path))


def test_quit_and_skip_keys(session):
    # each frame: handle keys, then step. Demo 1: 2 steps then Right; demo 2: 1 step then Q
    frames = [[], [], [key(pygame.K_RIGHT)], [], [key(pygame.K_q)]]
    results = fast_view([session.path], frames=frames)
    assert len(results) == 2
    assert results[0].steps_shown == 2 and not results[0].finished
    assert results[1].steps_shown == 1


def test_previous_and_restart(session):
    frames = [[key(pygame.K_RIGHT)], [key(pygame.K_LEFT)], [key(pygame.K_r)], [key(pygame.K_q)]]
    results = fast_view([session.path], frames=frames)
    names = [r.path.rsplit('/', 1)[-1][:4] for r in results]
    assert names == ['0001', '0002', '0001', '0001']


def test_pause_holds_position(session):
    frames = [[], [key(pygame.K_SPACE)]] + [[]] * 5 + [[key(pygame.K_q)]]
    (r,) = fast_view([session.path], frames=frames)
    assert r.steps_shown == 1  # one step, then paused for 5 frames


def test_rejected_demo_is_labelled(session, monkeypatch):
    labels = []
    original = viewer.play_demo

    def spy(env, demo, label='', **kw):
        labels.append(label)
        return original(env, demo, label, **kw)

    monkeypatch.setattr(viewer, 'play_demo', spy)
    fast_view([session.path])
    assert sum('REJECTED' in label for label in labels) == 1


def test_no_demos_found(tmp_path):
    with pytest.raises(SystemExit):
        fast_view([tmp_path])


def test_main_runs(session, monkeypatch):
    monkeypatch.setattr(viewer, 'view', lambda paths, speed: (paths, speed))
    assert viewer.run([str(session.path), '--speed', '2']) == ([str(session.path)], 2.0)
