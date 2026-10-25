"""Recorder end to end (headless, scripted input) and exact replay from file."""

from pathlib import Path

import gymnasium as gym
import numpy as np
import pygame
import pytest

import pusht_scenarios.env  # noqa: F401
from pusht_scenarios.env.bounds import KeepAgentOnBoard
from pusht_scenarios.gen import replay
from pusht_scenarios.io.demo import find_demos, load_demo
from pusht_scenarios.io.session import read_events, rejected_ids
from pusht_scenarios.teleop import record
from pusht_scenarios.teleop.input import InputDevice

CONFIGS = Path(__file__).parents[1] / 'configs'
ON_GOAL = {'variation': [], 'variation_values': {'block.start_position': [256.0, 256.0],
                                                  'block.angle': float(np.pi / 4)}}


class SteeringStick:
    """Scripted stick: steers at the block, plus noise (makes contact)."""

    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)
        self.env = None

    def get_axis(self, i):
        b = self.env.unwrapped
        d = np.array(b.block.position) - np.array(b.agent.position)
        v = d / (np.linalg.norm(d) + 1e-9) + self.rng.normal(0, 0.4, 2)
        return float(np.clip(v[i], -1, 1))

    def get_button(self, i):
        return False


class IdleStick:
    def get_axis(self, i):
        return 0.0

    def get_button(self, i):
        return False


def scripted(*frames):
    frames = list(frames)
    return lambda: frames.pop(0) if frames else []


def key(k):
    return pygame.event.Event(pygame.KEYDOWN, key=k)


@pytest.fixture(scope='module', autouse=True)
def pygame_init():
    pygame.init()
    yield
    pygame.quit()


@pytest.fixture
def env():
    env = KeepAgentOnBoard(gym.make('pts/PushT-v0', render_mode='human'))
    yield env
    env.close()


def run(env, tmp_path, conditions, stick=None, events=None, **top):
    raw = {'rounds_per_condition': 1, 'max_steps': 40, 'fps': 0, 'wait_for_input': False,
           'output': {'root': str(tmp_path), 'session_name': 't', 'operator': 'tester'},
           'conditions': conditions, **top}
    config = record.record_config_from_dict(raw)
    stick = stick or SteeringStick()
    if isinstance(stick, SteeringStick):
        stick.env = env
    device = InputDevice(joystick=stick)
    return record.record(config, env=env, device=device,
                         get_events=events or (lambda: []), verbose=False)


PHYSICS = {
    'default': {},
    'ice': {'sim.damping': 1.0},
    'heavy': {'sim.agent_max_force': 8000.0, 'sim.block_mass': 3.0},
    'mix': {'sim.damping': 1.0, 'sim.floor_friction': 0.5, 'sim.contact_friction': 1.0,
            'sim.block_cog_offset': [10.0, -10.0], 'sim.agent_max_force': 9000.0,
            'block.shape': 1},
    'goal': None,  # random goals (resampled variations)
}


def conditions_for(*names):
    out = []
    for n in names:
        if PHYSICS[n] is None:
            out.append({'name': n, 'variation': ['agent.start_position', 'block.start_position',
                                                 'block.angle', 'goal.position', 'goal.angle']})
        else:
            out.append({'name': n, 'variation_values': PHYSICS[n]})
    return out


# config


def test_record_config_parses_output_and_sandbox():
    cfg = record.record_config_from_dict({'output': {'root': 'x', 'save_failures': False},
                                          'rounds_per_condition': 2})
    assert cfg.output.root == 'x' and not cfg.output.save_failures
    assert cfg.sandbox.rounds_per_condition == 2


def test_save_failures_defaults_true():
    assert record.record_config_from_dict({}).output.save_failures


def test_unknown_output_key_rejected():
    with pytest.raises(ValueError, match='folder'):
        record.record_config_from_dict({'output': {'folder': 'x'}})


# recording


def test_success_is_saved_with_meta(env, tmp_path):
    session, results = run(env, tmp_path, [{'name': 'on_goal', **ON_GOAL}], stick=IdleStick())
    assert [r.outcome for r in results] == ['success']
    (demo_id, path), = session.saved
    d = load_demo(path)
    m = d.meta
    assert (m.demo_id, m.condition, m.outcome, m.attempt, m.operator) == (demo_id, 'on_goal', 'success', 1, 'tester')
    assert m.steps == len(d) == 5  # hold_steps
    assert m.device['kind'] == 'gamepad' and m.device['gain'] == 0.6
    assert m.session_id == session.session_id and m.created_at
    assert m.reset_options == {'variation': [], 'variation_values': ON_GOAL['variation_values']}
    assert d.on_goal.all()


def test_timeout_saved_by_default(env, tmp_path):
    session, results = run(env, tmp_path, conditions_for('default'), max_steps=12)
    assert results[0].outcome == 'timeout'
    assert load_demo(session.saved[0][1]).meta.outcome == 'timeout'


def test_timeout_discarded_when_save_failures_false(env, tmp_path):
    raw_output = {'root': str(tmp_path), 'save_failures': False}
    config = record.record_config_from_dict({'max_steps': 5, 'fps': 0, 'wait_for_input': False,
                                             'rounds_per_condition': 1, 'output': raw_output})
    stick = SteeringStick()
    stick.env = env
    session, _ = record.record(config, env=env, device=InputDevice(joystick=stick),
                               get_events=lambda: [], verbose=False)
    assert session.saved == []
    assert 'round_end' in [e['event'] for e in read_events(session.path)]


def test_retry_not_saved_and_counts_attempts(env, tmp_path):
    # round 1: R on the 3rd frame (retry); round 2 of the same seed: success
    events = scripted([], [], [key(pygame.K_r)])
    session, results = run(env, tmp_path, [{'name': 'on_goal', **ON_GOAL}], stick=IdleStick(),
                           events=events)
    assert [r.outcome for r in results] == ['retry', 'success']
    (_, path), = session.saved
    assert load_demo(path).meta.attempt == 2


def test_skip_and_quit_not_saved(env, tmp_path):
    session, results = run(env, tmp_path, conditions_for('default', 'ice'),
                           events=scripted([], [key(pygame.K_n)], [], [key(pygame.K_q)]))
    assert [r.outcome for r in results] == ['skip', 'quit']
    assert session.saved == []


def test_undo_marks_last_demo_rejected(env, tmp_path):
    # 2 successful rounds, then U during round 3 (which is then quit)
    conds = [{'name': 'a', **ON_GOAL}, {'name': 'b', **ON_GOAL}, {'name': 'c', **ON_GOAL}]
    frames = [[]] * 10 + [[key(pygame.K_u)], [key(pygame.K_q)]]
    session, _ = run(env, tmp_path, conds, stick=IdleStick(), events=scripted(*frames),
                     hold_steps=5)
    assert len(session.saved) == 2
    assert rejected_ids(session.path) == {session.saved[1][0]}


def test_wait_for_input_excludes_idle_frames(env, tmp_path):
    # idle for 5 frames, then the stick moves: recording starts at the move
    class LateStick(SteeringStick):
        frames = 0

        def get_axis(self, i):
            if i == 0:
                LateStick.frames += 1
            return super().get_axis(i) if LateStick.frames > 5 else 0.0

    session, _ = run(env, tmp_path, conditions_for('default'), stick=LateStick(),
                     wait_for_input=True, max_steps=10)
    d = load_demo(session.saved[0][1])
    assert len(d) == 10 and np.all(np.abs(d.raw_action).sum(axis=1) > 0)


def test_shuffle_order_is_recorded_and_followed(env, tmp_path):
    conds = [{'name': n, **ON_GOAL} for n in 'abcdef']
    session, results = run(env, tmp_path, conds, stick=IdleStick(),
                           shuffle_conditions=True, shuffle_seed=7)
    import yaml
    order = yaml.safe_load((session.path / 'session.yaml').read_text())['config']['condition_order']
    assert [r.condition for r in results] == order
    assert sorted(order) == list('abcdef') and order != list('abcdef')


def test_shuffle_without_seed_pins_one(env, tmp_path):
    conds = [{'name': n, **ON_GOAL} for n in 'abc']
    session, _ = run(env, tmp_path, conds, stick=IdleStick(), shuffle_conditions=True)
    import yaml
    assert yaml.safe_load((session.path / 'session.yaml').read_text())['config']['shuffle_seed'] is not None


# replay from file


@pytest.mark.parametrize('name', list(PHYSICS))
def test_replay_from_file_is_exact(env, tmp_path, name):
    session, _ = run(env, tmp_path, conditions_for(name), rounds_per_condition=2, max_steps=50)
    for _, path in session.saved:
        d = load_demo(path)
        r = replay.replay(d)
        assert replay.divergence(d, r) == 0.0
        np.testing.assert_array_equal(r.sim_state, d.sim_state)
        np.testing.assert_array_equal(r.on_goal, d.on_goal)


def test_replay_needs_the_same_reset_call(env, tmp_path):
    # Setting every value explicitly is not the same reset: with a physical
    # agent, the start state differs (see gen/replay.py). Documents why
    # meta.reset_options exists.
    # 'mix' seed 1 is a known case where the start state depends on it
    session, _ = run(env, tmp_path, conditions_for('mix'), rounds_per_condition=2, max_steps=30)
    exact_all, via_values = [], []
    for _, path in session.saved:
        d = load_demo(path)
        exact_all.append(replay.divergence(d, replay.replay(d)))
        d.meta.reset_options = {}  # fall back to explicit values
        via_values.append(replay.divergence(d, replay.replay(d)))
    assert max(exact_all) == 0.0
    assert max(via_values) > 0.0


def test_replay_detects_wrong_variation(env, tmp_path):
    session, _ = run(env, tmp_path, conditions_for('heavy'))
    d = load_demo(session.saved[0][1])
    d.meta.variation['sim.block_mass'] = 1.0
    with pytest.raises(replay.ReplayMismatch, match='sim.block_mass'):
        replay.replay(d)


def test_replay_other_actions(env, tmp_path):
    # augmenters will replay modified actions through the same path
    session, _ = run(env, tmp_path, conditions_for('default'), max_steps=20)
    d = load_demo(session.saved[0][1])
    r = replay.replay(d, actions=np.zeros_like(d.action))
    assert r.state.shape == d.state.shape
    assert replay.divergence(d, r) > 0


def test_replay_from_mid_episode_restores_exact_state(env, tmp_path):
    # The restored state is exact, but the rollout after it is NOT guaranteed
    # to match: pymunk's cached contact/joint impulses aren't in any state, and
    # contact makes tiny differences grow (measured: up to hundreds of px).
    # So augmenters must simulate from reset (start=0), never from a restore.
    session, _ = run(env, tmp_path, conditions_for('ice'), max_steps=40)
    d = load_demo(session.saved[0][1])
    r = replay.replay(d, start=20)
    np.testing.assert_array_equal(r.state[0], d.state[20])
    np.testing.assert_array_equal(r.sim_state[0], d.sim_state[20])
    assert len(r.state) == len(d.state) - 20


def test_all_demos_in_session_found(env, tmp_path):
    session, _ = run(env, tmp_path, conditions_for('default', 'ice'), max_steps=10)
    assert find_demos(tmp_path) == sorted(p for _, p in session.saved)


def test_demo_yaml_conditions_record(env, tmp_path):
    # every demo.yaml condition records and replays exactly (short rounds)
    cfg = record.load_record_config(CONFIGS / 'demo.yaml')
    raw_conditions = [{'name': c.name, 'variation': c.variation, 'variation_values': c.variation_values}
                      for c in cfg.sandbox.conditions]
    session, _ = run(env, tmp_path, raw_conditions, max_steps=15)
    assert len(session.saved) == len(raw_conditions)
    for _, path in session.saved:
        d = load_demo(path)
        assert replay.divergence(d, replay.replay(d)) == 0.0


# resuming a long suite over several sessions


def test_resume_skips_recorded_rounds(env, tmp_path):
    conds = [{'name': n, **ON_GOAL} for n in ('a', 'b')]
    top = dict(rounds_per_condition=2, output={'root': str(tmp_path), 'resume': True})

    def session(events):
        raw = {'max_steps': 40, 'fps': 0, 'wait_for_input': False, 'conditions': conds, **top}
        cfg = record.record_config_from_dict(raw)
        return record.record(cfg, env=env, device=InputDevice(joystick=IdleStick()),
                             get_events=events, verbose=False)

    # session 1: one round (5 frames to succeed), then Q
    s1, r1 = session(scripted(*([[]] * 5), [key(pygame.K_q)]))
    assert [r.outcome for r in r1] == ['success', 'quit'] and len(s1.saved) == 1
    # session 2: only the 3 remaining rounds are played
    s2, r2 = session(lambda: [])
    assert [(r.condition, r.seed, r.outcome) for r in r2] == [('a', 1, 'success'), ('b', 0, 'success'),
                                                              ('b', 1, 'success')]
    # everything is recorded now; an undone demo makes its round playable again
    with open(s2.path / 'events.jsonl', 'a') as f:
        f.write('{"event": "undo", "demo_id": "%s"}\n' % s2.saved[-1][0])
    s3, r3 = session(lambda: [])
    assert [(r.condition, r.seed) for r in r3] == [('b', 1)]


def test_resume_off_records_everything(env, tmp_path):
    conds = [{'name': 'a', **ON_GOAL}]
    for _ in range(2):
        cfg = record.record_config_from_dict({'max_steps': 40, 'fps': 0, 'wait_for_input': False,
                                              'rounds_per_condition': 2, 'conditions': conds,
                                              'output': {'root': str(tmp_path)}})
        _, results = record.record(cfg, env=env, device=InputDevice(joystick=IdleStick()),
                                   get_events=lambda: [], verbose=False)
        assert len(results) == 2


def test_resume_counts_demos_from_an_earlier_seed_plan(env, tmp_path):
    # demos recorded under another seed plan still count toward the target
    def session(seed, rounds):
        cfg = record.record_config_from_dict({
            'max_steps': 40, 'fps': 0, 'wait_for_input': False, 'seed': seed,
            'rounds_per_condition': rounds, 'conditions': [{'name': 'a', **ON_GOAL}],
            'output': {'root': str(tmp_path), 'resume': True}})
        return record.record(cfg, env=env, device=InputDevice(joystick=IdleStick()),
                             get_events=lambda: [], verbose=False)

    _, first = session(seed=50, rounds=2)  # old plan: seeds 50, 51
    _, second = session(seed=0, rounds=3)  # new plan wants 3: only 1 more, on a fresh seed
    assert [r.seed for r in first] == [50, 51]
    assert [r.seed for r in second] == [0]
    _, third = session(seed=0, rounds=2)  # already have 3 >= 2: nothing to play
    assert third == []
