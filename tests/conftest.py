import os

import pytest

# Headless pygame: no window or audio device needed for tests.
os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
os.environ.setdefault('SDL_AUDIODRIVER', 'dummy')


@pytest.fixture(autouse=True)
def no_real_gamepads(monkeypatch):
    """Hide any gamepad connected to the machine (via SDL or hidapi), so tests
    don't depend on hardware. Tests that need a gamepad inject a fake one."""
    import pygame

    from pusht_scenarios.teleop import hid_gamepad

    monkeypatch.setattr(pygame.joystick, 'get_count', lambda: 0)
    monkeypatch.setattr(hid_gamepad, 'open_first_gamepad', lambda: None)
