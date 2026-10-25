"""Import and registration checks for the core package."""

import subprocess
import sys

import gymnasium as gym


def test_modules_import():
    import pusht_scenarios.env  # noqa: F401
    from pusht_scenarios.env import env, spaces, utils  # noqa: F401
    from pusht_scenarios.env.env import PushT  # noqa: F401


def test_gym_id_registered():
    import pusht_scenarios.env  # noqa: F401

    env = gym.make('pts/PushT-v0')
    obs, info = env.reset(seed=0)
    assert set(obs) == {'proprio', 'state'}
    env.close()


def test_core_does_not_import_swm_or_torch():
    # Teleop runs on a laptop, so the core package must stay free of the
    # stable_worldmodel/torch stack. Run in a fresh interpreter so other
    # tests' imports don't leak into sys.modules.
    code = (
        'import sys, gymnasium as gym, pusht_scenarios.env\n'
        "gym.make('pts/PushT-v0').reset(seed=0)\n"
        "leaked = [m for m in ('stable_worldmodel', 'torch', 'shapely') if m in sys.modules]\n"
        'assert not leaked, leaked\n'
    )
    result = subprocess.run(
        [sys.executable, '-c', code], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
