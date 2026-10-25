# reads pygame device - default Gamepad, but mouse when unavailable
#
# PushT (relative=True) takes actions in [-1, 1]^2 and itself sets the agent's
# PD target to agent_pos + action * action_scale (100 px). So read() returns
# that relative action, never a target position. Window pixels == world
# coordinates (512x512, y down), so the stick and mouse need no flipping.
import numpy as np
import pygame
import gymnasium


class InputDevice:
    def __init__(
        self,
        stick_thresh=0.1,
        gain=0.6,
        precision_gain=0.1,
        precision_button=None,
        joystick=None,
        mouse_pos=None,
        use_hid=True,
    ):
        """
        Args:
            stick_thresh: radial dead zone on the stick, in [0, 1).
            gain: action at full stick deflection (1.0 = 100 px target offset).
            precision_gain: gain while precision_button is held.
            precision_button: joystick button index for precision mode, or None.
            joystick: object with get_axis/get_button; defaults to the first
                pygame joystick if one is connected, else (use_hid) the first
                HID gamepad read via hidapi. Injectable for tests.
            mouse_pos: callable returning the mouse (x, y) in window pixels;
                defaults to pygame.mouse.get_pos. Injectable for tests.
            use_hid: fall back to hidapi when SDL sees no joystick (SDL drops
                some Bluetooth gamepads on macOS; see hid_gamepad.py).
        """
        self.stick_thresh = stick_thresh
        self.gain = gain
        self.precision_gain = precision_gain
        self.precision_button = precision_button
        self.mouse_pos = mouse_pos or pygame.mouse.get_pos

        if joystick is None:
            pygame.joystick.init()
            if pygame.joystick.get_count() > 0:
                joystick = pygame.joystick.Joystick(0)
            elif use_hid:
                from pusht_scenarios.teleop.hid_gamepad import open_first_gamepad

                joystick = open_first_gamepad()
        # None -> fall back to mouse
        self.device = joystick

    @property
    def kind(self) -> str:
        return 'mouse' if self.device is None else 'gamepad'

    @property
    def name(self) -> str:
        if self.device is None:
            return 'mouse'
        name = getattr(self.device, 'get_name', None)
        return name() if name else getattr(self.device, 'name', 'gamepad')

    def engaged(self, env: gymnasium.Env, action: np.ndarray) -> bool:
        """Has the player started? Stick: out of the dead zone. Mouse: cursor
        near the agent (Diffusion Policy style), so a resting cursor elsewhere
        doesn't start the round."""
        if self.device is not None:
            return bool(np.any(action != 0))
        return float(np.linalg.norm(action)) * env.unwrapped.action_scale < 30

    # detects input and produces action for gym step
    def read(self, env: gymnasium.Env) -> np.ndarray:
        if self.device is not None:
            return self._read_stick()
        return self._read_mouse(env)

    def _read_stick(self) -> np.ndarray:
        stick = np.array(
            [self.device.get_axis(0), self.device.get_axis(1)], dtype=np.float64
        )

        # radial dead zone: per-axis thresholds would snap motion to the axes.
        # Rescale so output ramps from 0 at the dead zone edge to 1 at full tilt.
        mag = np.linalg.norm(stick)
        if mag <= self.stick_thresh:
            return np.zeros(2, dtype=np.float32)
        # stick corners can exceed magnitude 1
        scaled = min((mag - self.stick_thresh) / (1 - self.stick_thresh), 1.0)
        direction = stick / mag

        precise = (
            self.precision_button is not None
            and self.device.get_button(self.precision_button)
        )
        gain = self.precision_gain if precise else self.gain
        return (direction * scaled * gain).astype(np.float32)

    def _read_mouse(self, env: gymnasium.Env) -> np.ndarray:
        # Diffusion Policy style: the mouse is the agent's target. Express it
        # relative to the agent, in action units, clipped to the action space.
        base = env.unwrapped
        mouse = np.array(self.mouse_pos(), dtype=np.float64)
        agent = np.array(base.agent.position, dtype=np.float64)
        action = (mouse - agent) / base.action_scale
        return np.clip(action, -1, 1).astype(np.float32)
