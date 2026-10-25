"""HIDGamepad tests against a real descriptor with synthetic reports (no hardware)."""

import gymnasium as gym
import numpy as np
import pytest

import pusht_scenarios.env  # noqa: F401
from pusht_scenarios.teleop.hid_gamepad import HIDGamepad, parse_report_descriptor
from pusht_scenarios.teleop.input import InputDevice

# 8BitDo Ultimate 2C Wireless (2dc8:301b) over Bluetooth LE, captured 2026-10-24.
# Report 1 payload: hat nibble | X Y Z Rz (0-255) | 2 triggers | 16 buttons | battery
ULTIMATE_2C = bytes.fromhex(
    '05010905a1018501050115002507463b0195017504651409398142750195048101150026ff00'
    '09300931093209359504750881020502150026ff0009c409c5950275088102050919012910150025'
    '017501951081020506092015002564750895018102050f0970850515002564750895049102c0'
)


def report(x=127, y=127, z=127, rz=127, buttons=0, hat=15):
    return [1, hat, x, y, z, rz, 0, 0, buttons & 0xFF, buttons >> 8, 90]


class FakeHID:
    """Stand-in for an opened, non-blocking hid.device."""

    def __init__(self, *reports):
        self.queue = list(reports)

    def read(self, n):
        return self.queue.pop(0) if self.queue else []


def pad(*reports):
    return HIDGamepad(FakeHID(*reports), ULTIMATE_2C, 'fake 2C')


def test_descriptor_fields():
    fields = parse_report_descriptor(ULTIMATE_2C)
    axes = {(f.usage, f.bit_offset, f.bit_size) for f in fields if f.usage_page == 1 and f.usage in (0x30, 0x31, 0x32, 0x35)}
    assert axes == {(0x30, 8, 8), (0x31, 16, 8), (0x32, 24, 8), (0x35, 32, 8)}
    buttons = [f for f in fields if f.usage_page == 9]
    assert [f.usage for f in buttons] == list(range(1, 17))
    assert buttons[0].bit_offset == 56
    assert all(f.logical_max == 255 for f in fields if f.usage == 0x30)


def test_axis_order_and_count():
    g = pad()
    assert g.get_numaxes() == 4
    assert [f.usage for f in g.axes] == [0x30, 0x31, 0x32, 0x35]


def test_no_report_yet_is_centered():
    g = pad()
    assert g.get_axis(0) == 0.0 and g.get_axis(1) == 0.0
    assert not g.get_button(0)


def test_axis_scaling():
    g = pad(report(x=255, y=0))
    assert g.get_axis(0) == pytest.approx(1.0)
    assert g.get_axis(1) == pytest.approx(-1.0)
    assert abs(g.get_axis(2)) < 0.01  # right stick centered


def test_center_inside_dead_zone():
    g = pad(report())
    assert abs(g.get_axis(0)) < 0.01 and abs(g.get_axis(1)) < 0.01


def test_latest_report_wins():
    g = pad(report(x=0), report(x=200), report(x=255))
    assert g.get_axis(0) == pytest.approx(1.0)


def test_buttons():
    g = pad(report(buttons=(1 << 0) | (1 << 9)))
    assert g.get_button(0) and g.get_button(9)
    assert not g.get_button(1) and not g.get_button(15)
    assert not g.get_button(99)  # unknown index


def test_drives_input_device():
    # stick hard right and slightly down -> rightward, slightly downward action
    env = gym.make('pts/PushT-v0')
    env.reset(seed=0)
    device = InputDevice(joystick=pad(report(x=255, y=160)), gain=0.3)
    assert device.kind == 'gamepad' and device.name == 'fake 2C'
    action = device.read(env)
    assert action[0] > 0.25 and 0 < action[1] < 0.15
    env.close()
