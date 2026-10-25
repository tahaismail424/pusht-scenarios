import os

import pytest

# Headless pygame: no window or audio device needed for tests.
os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
os.environ.setdefault('SDL_AUDIODRIVER', 'dummy')
# Keep SDL's controller drivers off: with a Switch Pro-style pad connected,
# every pygame.init() (each human-mode env) costs a ~2 s HIDAPI handshake.
os.environ.setdefault('SDL_JOYSTICK_HIDAPI', '0')
os.environ.setdefault('SDL_JOYSTICK_MFI', '0')


@pytest.fixture(autouse=True)
def no_real_gamepads(monkeypatch):
    """Hide any gamepad connected to the machine (via SDL or hidapi), so tests
    don't depend on hardware. Tests that need a gamepad inject a fake one."""
    import pygame

    from pusht_scenarios.teleop import hid_gamepad

    monkeypatch.setattr(pygame.joystick, 'get_count', lambda: 0)
    monkeypatch.setattr(hid_gamepad, 'open_first_gamepad', lambda: None)


# --- shared mini dataset: suite -> recordings -> augmentation ----------------

MINI_SUITE = {
    'name': 'mini',
    'base': {'sim.agent_max_force': 10000.0},
    'holdout_policy': 'middle',
    'factors': {
        'mass': {'ordered': True, 'levels': {'light': {'sim.block_mass': 0.5},
                                             'normal': {'sim.block_mass': 1.0},
                                             'heavy': {'sim.block_mass': 3.0}}},
        'color': {'ordered': False, 'holdout': 'night', 'levels': {
            'default': {},
            'warm': {'block.color': [205, 70, 70], 'background.color': [250, 245, 235]},
            'night': {'block.color': [190, 190, 210], 'background.color': [25, 25, 35]}}},
    },
    'record': {'rounds': {'train': 3, 'test': 4}, 'max_steps': 25, 'fps': 0,
               'wait_for_input': False},
}
ON_GOAL = {'block.start_position': [256.0, 256.0], 'block.angle': 0.7853981633974483,
           'agent.start_position': [120.0, 420.0]}


class SteeringStick:
    """Scripted stick: steers at the block, plus noise (makes contact)."""

    def __init__(self, seed=0):
        import numpy as np
        self.rng = np.random.default_rng(seed)
        self.env = None

    def get_axis(self, i):
        import numpy as np
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


@pytest.fixture(scope='session')
def mini(tmp_path_factory):
    """Built mini suite (4 train + 2 test envs), recorded demos, augmented copies.

    Train envs: 3 contact-rich scripted rounds each (timeouts). Test envs: 4
    rounds each that start on the goal and succeed (eval/adapt need successes).
    """
    import gymnasium as gym
    import pygame

    from pusht_scenarios import suite as S
    from pusht_scenarios.env.bounds import KeepAgentOnBoard
    from pusht_scenarios.gen import recipe as R
    from pusht_scenarios.teleop import record
    from pusht_scenarios.teleop.input import InputDevice

    root = tmp_path_factory.mktemp('mini')
    built = S.build(S.spec_from_dict(MINI_SUITE))
    S.write(built, root / 'suite')
    cfg = record.record_config_from_dict(S.record_config(built))
    raw = root / 'raw'
    cfg.output.root = str(raw)

    pygame.init()
    env = KeepAgentOnBoard(gym.make('pts/PushT-v0', render_mode='human'))
    train = [c for c in cfg.sandbox.conditions if c.info['split'] == 'train']
    test = [c for c in cfg.sandbox.conditions if c.info['split'] == 'test']
    for c in test:  # start on the goal, so idle play succeeds
        c.variation = []
        c.variation_values = {**c.variation_values, **ON_GOAL}

    stick = SteeringStick()
    stick.env = env
    cfg.sandbox.conditions = train
    cfg.output.session_name = 'train'
    record.record(cfg, env=env, device=InputDevice(joystick=stick), get_events=lambda: [], verbose=False)
    cfg.sandbox.conditions = test
    cfg.output.session_name = 'test'
    record.record(cfg, env=env, device=InputDevice(joystick=IdleStick()), get_events=lambda: [], verbose=False)
    env.close()
    pygame.quit()

    recipe = R.recipe_from_dict({
        'name': 'aug', 'sources': [str(raw)], 'output_root': str(root),
        'bases': {'original': True, 'rotations_deg': [180]},
        'noise': [[0.0, 1], [0.05, 1]],
    })
    aug = R.run_recipe(recipe, workers=1, verbose=False)
    return {'root': root, 'suite_path': root / 'suite' / 'suite.yaml', 'suite': built,
            'raw': raw, 'aug': aug}
