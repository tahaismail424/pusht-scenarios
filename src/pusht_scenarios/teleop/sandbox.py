# Playable PushT sandbox: no recording, just rounds under configured variations.
#
#   uv run pts-sandbox                       # default env, 5 rounds
#   uv run pts-sandbox -c configs/sandbox.yaml
#
# Keys: R = retry round, N = skip round, Space = pause, Q / Esc = quit,
#       P = queue/unqueue a pause before the next round (P again to continue).
# A round ends when the block holds on the goal outline, or at max_steps.
import argparse
import random
import time
from dataclasses import dataclass, field

import gymnasium as gym
import numpy as np
import pygame
import yaml

import pusht_scenarios.env  # noqa: F401  (registers pts/PushT-v0)
from pusht_scenarios.env import task, variations
from pusht_scenarios.env.bounds import KeepAgentOnBoard
from pusht_scenarios.teleop.input import InputDevice


@dataclass
class Condition:
    name: str = 'default'
    # variation keys resampled per seed; None = swm's defaults (agent start,
    # block start, block angle)
    variation: list[str] | None = None
    # fixed values, dotted keys, e.g. {'block.shape': 1, 'sim.contact_friction': 0.5}
    variation_values: dict = field(default_factory=dict)
    rounds: int | None = None  # overrides rounds_per_condition
    seed: int | None = None  # overrides the session seed (round i uses seed + i)
    info: dict = field(default_factory=dict)  # stored in each demo (suite env id, split, ...)


@dataclass
class SandboxConfig:
    conditions: list[Condition] = field(default_factory=lambda: [Condition()])
    rounds_per_condition: int = 5
    # round i of every condition uses seed + i, so conditions share start states
    seed: int = 0
    max_steps: int = 300  # 30 s at 10 Hz
    fps: int = 10  # display rate; 0 = unthrottled. Sim is always 10 Hz per step.
    wait_for_input: bool = True  # the round clock starts at the first input
    pos_tol: float = 20.0
    angle_tol_deg: float = 20.0
    hold_steps: int = 5  # success must hold this many consecutive steps
    device: dict = field(default_factory=dict)  # InputDevice kwargs
    # play the condition blocks in random order (seeds within a block unchanged)
    shuffle_conditions: bool = False
    shuffle_seed: int | None = None  # None: a fresh random order every session


@dataclass
class RoundResult:
    condition: str
    seed: int
    outcome: str  # success | timeout | retry | skip | quit
    steps: int
    seconds: float
    pause_after: bool = False  # P was queued: pause before the next round


# Keys that belong to the recorder (teleop/record.py); the sandbox ignores them
# so one YAML can drive both.
RECORDER_KEYS = {'output'}


def config_from_dict(raw: dict) -> SandboxConfig:
    raw = {k: v for k, v in raw.items() if k not in RECORDER_KEYS}
    known = set(SandboxConfig.__dataclass_fields__)
    unknown = set(raw) - known
    if unknown:
        raise ValueError(f'unknown sandbox config keys: {sorted(unknown)}')
    conditions = raw.pop('conditions', None)
    config = SandboxConfig(**raw)
    if conditions is not None:
        if not conditions:
            raise ValueError('conditions must not be empty')
        config.conditions = [Condition(**c) for c in conditions]
        names = [c.name for c in config.conditions]
        if len(set(names)) != len(names):
            raise ValueError(f'duplicate condition names: {names}')
    return config


def read_yaml(path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def load_config(path=None) -> SandboxConfig:
    if path is None:
        return SandboxConfig()
    return config_from_dict(read_yaml(path))


def condition_order(config: SandboxConfig) -> list[Condition]:
    """Conditions in play order: as listed, or shuffled (shuffle_conditions)."""
    order = list(config.conditions)
    if config.shuffle_conditions:
        random.Random(config.shuffle_seed).shuffle(order)
    return order


def reset_options(env, condition: Condition) -> dict:
    """Build reset options, casting YAML lists/numbers to each space's dtype."""
    try:
        values = variations.cast_values(env, condition.variation_values)
    except ValueError as e:
        raise ValueError(f'{condition.name}: {e}') from None
    options = {'variation_values': values}
    if condition.variation is not None:
        options['variation'] = list(condition.variation)
    return options


def draw_hud(env, text):
    window = env.unwrapped.window
    if window is None:
        return
    font = pygame.font.Font(None, 24)
    surface = font.render(text, True, (255, 255, 255), (0, 0, 0))
    window.blit(surface, (8, 8))
    pygame.display.update()


class RoundHooks:
    """Callbacks into the round loop; the sandbox uses these no-ops, the
    recorder (teleop/record.py) overrides them to capture demos."""

    def on_round_start(self, env, condition: Condition, seed: int) -> None:
        """Play has started (first input past the dead zone): env is in its
        start state, no steps taken yet."""

    def on_step(self, env, raw_action, info, on_goal: bool, t_wall: float) -> None:
        """After each env.step. info['applied_action'] is the bounded action applied."""

    def on_round_end(self, env, result: 'RoundResult', started: bool) -> None:
        """Every exit: success, timeout, retry, skip, quit."""

    def on_event(self, env, event) -> None:
        """Key/joystick events the loop itself doesn't handle (e.g. U = undo)."""

    def status(self) -> str:
        """Extra text for the on-screen status line (e.g. 'saved 12')."""
        return ''


def run_round(
    env,
    device,
    condition: Condition,
    seed: int,
    config: SandboxConfig,
    label='',
    get_events=pygame.event.get,
    clock=None,
    hooks: RoundHooks | None = None,
    pause_queued: bool = False,
) -> RoundResult:
    clock = clock or pygame.time.Clock()
    hooks = hooks or RoundHooks()
    env.reset(seed=seed, options=reset_options(env, condition))
    angle_tol = np.deg2rad(config.angle_tol_deg)

    started = not config.wait_for_input
    paused = False
    steps = hold = 0
    t0 = time.monotonic()
    if started:
        hooks.on_round_start(env, condition, seed)

    def result(outcome):
        r = RoundResult(
            condition.name, seed, outcome, steps, time.monotonic() - t0,
            pause_after=pause_queued,
        )
        hooks.on_round_end(env, r, started)
        return r

    while True:
        for event in get_events():
            if event.type == pygame.QUIT:
                return result('quit')
            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                    return result('quit')
                if event.key == pygame.K_r:
                    return result('retry')
                if event.key == pygame.K_n:
                    return result('skip')
                if event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_p:
                    pause_queued = not pause_queued
                else:
                    hooks.on_event(env, event)
            else:
                hooks.on_event(env, event)

        status = f'{label}  step {steps}/{config.max_steps}'
        if pause_queued:
            status += '  PAUSE QUEUED'
        if paused:
            status += '  PAUSED'
        elif not started:
            action = device.read(env)
            if device.engaged(env, action):
                started = True
                t0 = time.monotonic()
                hooks.on_round_start(env, condition, seed)
            else:
                status += '  move to start'

        if started and not paused:
            raw_action = device.read(env)
            *_, info = env.step(raw_action)
            steps += 1
            on_goal = task.block_on_goal(env, config.pos_tol, angle_tol)
            hold = hold + 1 if on_goal else 0
            hooks.on_step(env, raw_action, info, on_goal, time.monotonic() - t0)

        extra = hooks.status()
        env.render()
        draw_hud(env, f'{status}  {extra}' if extra else status)

        if hold >= config.hold_steps:
            return result('success')
        if steps >= config.max_steps:
            return result('timeout')
        clock.tick(config.fps)


def wait_between_rounds(env, next_label, get_events=pygame.event.get, clock=None, fps=10) -> bool:
    """Hold on a pause screen until P (continue: True) or Q/Esc/close (False)."""
    clock = clock or pygame.time.Clock()
    while True:
        for event in get_events():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                    return False
                if event.key == pygame.K_p:
                    return True
        env.render()
        draw_hud(env, f'PAUSED - press P for next: {next_label}')
        clock.tick(fps)


def run_session(
    config, env=None, device=None, get_events=pygame.event.get, hooks=None
):
    """Play every round of every condition. Retry replays the same seed."""
    env = env or KeepAgentOnBoard(gym.make('pts/PushT-v0', render_mode='human'))
    device = device or InputDevice(**config.device)
    results = []
    clock = pygame.time.Clock()

    def rounds(c):
        return config.rounds_per_condition if c.rounds is None else c.rounds

    n = sum(rounds(c) for c in config.conditions)
    plan = []  # (condition, round index) in play order
    for condition in condition_order(config):
        plan.extend((condition, i) for i in range(rounds(condition)))

    k = 0
    while k < len(plan):
        condition, i = plan[k]
        base_seed = config.seed if condition.seed is None else condition.seed
        seed = base_seed + i
        label = f'{condition.name}  {k + 1}/{n}  seed {seed}'
        r = run_round(
            env, device, condition, seed, config, label, get_events, clock, hooks,
        )
        results.append(r)
        if r.outcome == 'quit':
            return results
        if r.outcome != 'retry':
            k += 1
        if r.pause_after and k < len(plan):
            nxt, j = plan[k]
            nseed = (config.seed if nxt.seed is None else nxt.seed) + j
            if not wait_between_rounds(env, f'{nxt.name} seed {nseed}', get_events, clock, config.fps):
                return results
    return results


def summarize(results) -> str:
    lines = [f'{"condition":<20} {"seed":>4}  {"outcome":<8} {"steps":>5} {"secs":>6}']
    for r in results:
        lines.append(
            f'{r.condition:<20} {r.seed:>4}  {r.outcome:<8} {r.steps:>5} {r.seconds:>6.1f}'
        )
    done = [r for r in results if r.outcome in ('success', 'timeout')]
    if done:
        wins = sum(r.outcome == 'success' for r in done)
        lines.append(f'success {wins}/{len(done)} played rounds')
    return '\n'.join(lines)


def run(argv=None):
    parser = argparse.ArgumentParser(description='Play PushT under configured variations.')
    parser.add_argument('-c', '--config', help='sandbox YAML (default: plain PushT, 5 rounds)')
    args = parser.parse_args(argv)

    config = load_config(args.config)
    pygame.init()
    try:
        device = InputDevice(**config.device)
        print(f'input: {device.kind} ({device.name})')
        if device.kind == 'gamepad':
            if device.precision_index is None:
                print(f'precision: off (no button {device.precision_button!r} on this pad; '
                      'see `uv run pts-buttons`)')
            else:
                print(f'precision: hold {device.precision_button!r} (button {device.precision_index})')
        results = run_session(config, device=device)
    finally:
        pygame.quit()
    print(summarize(results))
    return results


def main(argv=None):
    """Console entry point (returns None so the exit status is 0)."""
    run(argv)


if __name__ == '__main__':
    main()
