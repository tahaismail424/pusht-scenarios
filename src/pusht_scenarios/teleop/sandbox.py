# Playable PushT sandbox: no recording, just rounds under configured variations.
#
#   uv run pts-sandbox                       # default env, 5 rounds
#   uv run pts-sandbox -c configs/sandbox.yaml
#
# Keys: R = retry round, N = skip round, Space = pause, Q / Esc = quit.
# A round ends when the block holds on the goal outline, or at max_steps.
import argparse
import time
from dataclasses import dataclass, field

import gymnasium as gym
import numpy as np
import pygame
import yaml

import pusht_scenarios.env  # noqa: F401  (registers pts/PushT-v0)
from pusht_scenarios.env import task, utils
from pusht_scenarios.teleop.input import InputDevice


@dataclass
class Condition:
    name: str = 'default'
    # variation keys resampled per seed; None = swm's defaults (agent start,
    # block start, block angle)
    variation: list[str] | None = None
    # fixed values, dotted keys, e.g. {'block.shape': 1, 'sim.contact_friction': 0.5}
    variation_values: dict = field(default_factory=dict)


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


@dataclass
class RoundResult:
    condition: str
    seed: int
    outcome: str  # success | timeout | retry | skip | quit
    steps: int
    seconds: float


def load_config(path=None) -> SandboxConfig:
    if path is None:
        return SandboxConfig()
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
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
    return config


def reset_options(env, condition: Condition) -> dict:
    """Build reset options, casting YAML lists/numbers to each space's dtype."""
    space = env.unwrapped.variation_space
    values = {}
    for key, value in condition.variation_values.items():
        try:
            sub = utils.get_in(space, key.split('.'))
        except (KeyError, TypeError):
            raise ValueError(f'{condition.name}: unknown variation {key!r}')
        values[key] = np.asarray(value, dtype=sub.dtype).reshape(sub.shape)
        if values[key].shape == ():
            values[key] = values[key][()]
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


def run_round(
    env,
    device,
    condition: Condition,
    seed: int,
    config: SandboxConfig,
    label='',
    get_events=pygame.event.get,
    clock=None,
) -> RoundResult:
    clock = clock or pygame.time.Clock()
    env.reset(seed=seed, options=reset_options(env, condition))
    angle_tol = np.deg2rad(config.angle_tol_deg)

    started = not config.wait_for_input
    paused = False
    steps = hold = 0
    t0 = time.monotonic()

    def result(outcome):
        return RoundResult(
            condition.name, seed, outcome, steps, time.monotonic() - t0
        )

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

        status = f'{label}  step {steps}/{config.max_steps}'
        if paused:
            status += '  PAUSED'
        elif not started:
            action = device.read(env)
            if device.engaged(env, action):
                started = True
                t0 = time.monotonic()
            else:
                status += '  move to start'

        if started and not paused:
            env.step(device.read(env))
            steps += 1
            hold = hold + 1 if task.block_on_goal(
                env, config.pos_tol, angle_tol
            ) else 0

        env.render()
        draw_hud(env, status)

        if hold >= config.hold_steps:
            return result('success')
        if steps >= config.max_steps:
            return result('timeout')
        clock.tick(config.fps)


def run_session(config, env=None, device=None, get_events=pygame.event.get):
    """Play every round of every condition. Retry replays the same seed."""
    env = env or gym.make('pts/PushT-v0', render_mode='human')
    device = device or InputDevice(**config.device)
    results = []
    clock = pygame.time.Clock()
    n = len(config.conditions) * config.rounds_per_condition
    k = 0
    for condition in config.conditions:
        i = 0
        while i < config.rounds_per_condition:
            seed = config.seed + i
            label = f'{condition.name}  {k + 1}/{n}  seed {seed}'
            r = run_round(
                env, device, condition, seed, config, label, get_events, clock
            )
            results.append(r)
            if r.outcome == 'quit':
                return results
            if r.outcome != 'retry':
                i += 1
                k += 1
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


def main(argv=None):
    parser = argparse.ArgumentParser(description='Play PushT under configured variations.')
    parser.add_argument('-c', '--config', help='sandbox YAML (default: plain PushT, 5 rounds)')
    args = parser.parse_args(argv)

    config = load_config(args.config)
    pygame.init()
    try:
        device = InputDevice(**config.device)
        print(f'input: {device.kind} ({device.name})')
        results = run_session(config, device=device)
    finally:
        pygame.quit()
    print(summarize(results))
    return results


if __name__ == '__main__':
    main()
