# Show which gamepad button is which: press buttons, see index + name.
#
#   uv run pts-buttons        # Ctrl+C to stop
#
# Uses the same device lookup as the sandbox (SDL first, then hidapi).
import time

import pygame

from pusht_scenarios.teleop.input import InputDevice


def main():
    pygame.init()
    device = InputDevice(precision_button=None)
    if device.kind != 'gamepad':
        print('no gamepad found')
        return
    pad = device.device
    names = {i: n for n, i in device.button_names().items()}
    print(f'{device.name}: {pad.get_numbuttons()} buttons; press some (Ctrl+C to stop)')
    if not names:
        print('(no name mapping for this pad; indices only)')
    held = set()
    try:
        while True:
            pygame.event.pump()
            now = {i for i in range(pad.get_numbuttons()) if pad.get_button(i)}
            for i in sorted(now - held):
                print(f'button {i}: {names.get(i, "?")}')
            held = now
            time.sleep(0.02)
    except KeyboardInterrupt:
        pass
    finally:
        pygame.quit()


if __name__ == '__main__':
    main()
