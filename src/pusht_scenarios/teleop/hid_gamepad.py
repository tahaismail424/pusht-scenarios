# Gamepad reader that bypasses SDL, via hidapi.
#
# SDL (pygame's input layer) drops some Bluetooth gamepads on recent macOS even
# though IOKit exposes them as ordinary HID gamepads; seen with an 8BitDo
# Ultimate 2C (2dc8:301b), on both SDL 2.28 and 2.32, with every joystick hint.
# This reads the device directly and decodes its reports from its own HID
# report descriptor, so it works for any standard HID gamepad.
#
# Exposes the get_axis/get_button subset of pygame.joystick.Joystick that
# InputDevice uses. Axis order: X, Y, Z, Rx, Ry, Rz (whichever exist), scaled to
# [-1, 1]; +Y is down, matching pygame and the env's screen coordinates.
# Buttons: HID button usage n -> index n - 1.
from dataclasses import dataclass

GENERIC_DESKTOP, BUTTON_PAGE = 0x01, 0x09

# Button names -> indices, SDL GameController vocabulary. Raw HID has no
# standard layout; this is the common Android/"D-input" HID gamepad layout
# (8BitDo and others): A B C X Y Z L1 R1 L2 R2 Select Start Home L3 R3.
# Unverified for any specific pad; check with `uv run pts-buttons`.
ANDROID_HID_LAYOUT = {
    'a': 0, 'b': 1, 'x': 3, 'y': 4,
    'leftshoulder': 6, 'rightshoulder': 7,
    'back': 10, 'start': 11, 'guide': 12,
    'leftstick': 13, 'rightstick': 14,
}
GAMEPAD_USAGES = (0x04, 0x05)  # joystick, gamepad
AXIS_ORDER = (0x30, 0x31, 0x32, 0x33, 0x34, 0x35)  # X Y Z Rx Ry Rz


@dataclass
class Field:
    report_id: int
    usage_page: int
    usage: int
    bit_offset: int  # within the report payload, after the report id byte
    bit_size: int
    logical_min: int
    logical_max: int


def _signed(value: int, nbytes: int) -> int:
    bits = 8 * nbytes
    return value - (1 << bits) if nbytes and value >= 1 << (bits - 1) else value


def parse_report_descriptor(desc: bytes) -> list[Field]:
    """Decode the Input fields of a HID report descriptor (short items only)."""
    fields: list[Field] = []
    usage_page = report_size = report_count = report_id = 0
    logical_min = logical_max = 0
    usages: list[tuple[int, int]] = []
    usage_min = None
    offsets: dict[int, int] = {}  # bit offset per report id

    i = 0
    while i < len(desc):
        prefix = desc[i]
        nbytes = (0, 1, 2, 4)[prefix & 0x3]
        kind, tag = (prefix >> 2) & 0x3, prefix >> 4
        raw = int.from_bytes(bytes(desc[i + 1 : i + 1 + nbytes]), 'little')
        i += 1 + nbytes

        if kind == 1:  # global
            if tag == 0x0:
                usage_page = raw
            elif tag == 0x1:
                logical_min = _signed(raw, nbytes)
            elif tag == 0x2:
                logical_max = _signed(raw, nbytes)
                if logical_max < logical_min:  # e.g. 0..255 encoded in 1 byte
                    logical_max = raw
            elif tag == 0x7:
                report_size = raw
            elif tag == 0x8:
                report_id = raw
            elif tag == 0x9:
                report_count = raw
        elif kind == 2:  # local; 4-byte usages carry their own page
            page = raw >> 16 if nbytes == 4 else usage_page
            if tag == 0x0:
                usages.append((page, raw & 0xFFFF))
            elif tag == 0x1:
                usage_min = raw & 0xFFFF
            elif tag == 0x2 and usage_min is not None:
                usages.extend((page, u) for u in range(usage_min, (raw & 0xFFFF) + 1))
                usage_min = None
        elif kind == 0:  # main
            if tag == 0x8:  # Input
                offset = offsets.get(report_id, 0)
                constant = raw & 0x1
                if not constant and usages:
                    for n in range(report_count):
                        page, usage = usages[min(n, len(usages) - 1)]
                        fields.append(
                            Field(
                                report_id, page, usage, offset + n * report_size,
                                report_size, logical_min, logical_max,
                            )
                        )
                offsets[report_id] = offset + report_size * report_count
            if tag in (0x8, 0x9, 0xB, 0xA, 0xC):  # input/output/feature/collection
                usages, usage_min = [], None
    return fields


def _read_bits(payload: bytes, offset: int, size: int) -> int:
    value = int.from_bytes(bytes(payload), 'little')
    return (value >> offset) & ((1 << size) - 1)


class HIDGamepad:
    def __init__(
        self, device, descriptor: bytes, name: str = 'HID gamepad', layout=None
    ):
        """device: an opened hid.device (or a stand-in with read(n)).
        layout: button name -> index (default ANDROID_HID_LAYOUT)."""
        self.device = device
        self.name = name
        self.layout = dict(ANDROID_HID_LAYOUT if layout is None else layout)
        fields = parse_report_descriptor(descriptor)
        self.numbered = any(f.report_id for f in fields)
        self.axes = sorted(
            (f for f in fields if f.usage_page == GENERIC_DESKTOP and f.usage in AXIS_ORDER),
            key=lambda f: AXIS_ORDER.index(f.usage),
        )
        self.buttons = {
            f.usage - 1: f for f in fields if f.usage_page == BUTTON_PAGE
        }
        self.reports: dict[int, bytes] = {}  # latest payload per report id

    def poll(self) -> None:
        """Drain pending reports (non-blocking); keep the latest per report id."""
        while True:
            data = self.device.read(64)
            if not data:
                return
            if self.numbered:
                self.reports[data[0]] = bytes(data[1:])
            else:
                self.reports[0] = bytes(data)

    def _value(self, f: Field):
        payload = self.reports.get(f.report_id)
        if payload is None:
            return None
        return _read_bits(payload, f.bit_offset, f.bit_size)

    def get_numaxes(self) -> int:
        return len(self.axes)

    def get_axis(self, i: int) -> float:
        self.poll()
        f = self.axes[i]
        v = self._value(f)
        if v is None:  # no report yet: centered
            return 0.0
        span = f.logical_max - f.logical_min
        return 2 * (v - f.logical_min) / span - 1 if span else 0.0

    def get_button(self, i: int) -> bool:
        self.poll()
        f = self.buttons.get(i)
        return bool(f and self._value(f))

    def get_numbuttons(self) -> int:
        return len(self.buttons)

    def button_names(self) -> dict[str, int]:
        return dict(self.layout)


def open_first_gamepad():
    """Open the first HID joystick/gamepad, or return None (also if hidapi is
    unavailable or the device can't be opened)."""
    try:
        import hid
    except ImportError:
        return None
    for info in hid.enumerate():
        if info['usage_page'] != GENERIC_DESKTOP or info['usage'] not in GAMEPAD_USAGES:
            continue
        try:
            device = hid.device()
            device.open_path(info['path'])
            device.set_nonblocking(True)
            descriptor = bytes(device.get_report_descriptor())
            name = info['product_string'] or 'HID gamepad'
            return HIDGamepad(device, descriptor, name)
        except (OSError, ValueError):
            continue
    return None
