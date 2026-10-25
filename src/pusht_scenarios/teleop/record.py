# Teleop recorder: the sandbox loop + hooks that capture and save demos.
#
#   uv run pts-record -c configs/demo.yaml
#
# Keys: as in the sandbox (R retry, N skip, Space pause, Q/Esc quit), plus
# U = undo the last saved demo (it stays on disk, marked rejected).
#
# How the pieces connect:
#
#   main()
#     load_record_config(yaml)          -> RecordConfig (sandbox settings + output)
#     SessionWriter(output, snapshot)   -> <root>/<session_id>/ (io/session.py)
#     DemoRecorder(session, device, …)  -> RoundHooks implementation (below)
#     sandbox.run_session(config, hooks=recorder)
#        on_round_start -> StepBuffer.start(env)                 (io/demo.py)
#        on_step        -> StepBuffer.add(env, raw, info, …)
#        on_round_end   -> success, or timeout if save_failures:
#                            session.write_demo(buffer.to_demo(meta))
#                          retry/skip/quit: logged, buffer discarded
#        on_event(U)    -> session.undo_last()
import argparse
import random
import uuid
from dataclasses import asdict, dataclass, field

import pygame

from pusht_scenarios.io.demo import DemoMeta, StepBuffer
from pusht_scenarios.io.session import (
    OutputConfig, SessionWriter, git_commit, now_iso, package_version,
)
from pusht_scenarios.teleop import sandbox
from pusht_scenarios.teleop.input import InputDevice
from pusht_scenarios.teleop.sandbox import RoundHooks, RoundResult, SandboxConfig


@dataclass
class RecordConfig:
    """A sandbox config (top-level keys) plus an `output:` block. The same YAML
    also runs in the sandbox, which ignores `output`."""

    sandbox: SandboxConfig = field(default_factory=SandboxConfig)
    output: OutputConfig = field(default_factory=OutputConfig)


def record_config_from_dict(raw: dict) -> RecordConfig:
    output = raw.get('output') or {}
    unknown = set(output) - set(OutputConfig.__dataclass_fields__)
    if unknown:
        raise ValueError(f'unknown output config keys: {sorted(unknown)}')
    return RecordConfig(sandbox.config_from_dict(raw), OutputConfig(**output))


def load_record_config(path) -> RecordConfig:
    return record_config_from_dict(sandbox.read_yaml(path))


def device_info(device) -> dict:
    return {
        'kind': device.kind,
        'name': device.name,
        'gain': device.gain,
        'stick_thresh': device.stick_thresh,
        'precision_gain': device.precision_gain,
        'precision_button': device.precision_button,
    }


class DemoRecorder(RoundHooks):
    """Captures each round into a StepBuffer and saves finished demos."""

    def __init__(self, session: SessionWriter, device, config: RecordConfig, verbose=True):
        self.session = session
        self.device = device
        self.config = config
        self.verbose = verbose
        self.buffer: StepBuffer | None = None
        self.retries: dict[tuple[str, int], int] = {}  # (condition, seed) -> retries
        self.message = ''
        self.reset_options: dict = {}
        self.condition_info: dict = {}

    def _say(self, text):
        self.message = text
        if self.verbose:
            print(text)

    def on_round_start(self, env, condition, seed) -> None:
        # what run_round passed to reset (sandbox.reset_options), as plain data
        self.reset_options = {
            'variation': None if condition.variation is None else list(condition.variation),
            'variation_values': dict(condition.variation_values),
        }
        self.condition_info = dict(condition.info)
        self.buffer = StepBuffer()
        self.buffer.start(env)
        self.session.log('round_start', condition=condition.name, seed=seed)

    def on_step(self, env, raw_action, info, on_goal, t_wall) -> None:
        if self.buffer is not None:
            self.buffer.add(env, raw_action, info, on_goal, t_wall)

    def on_round_end(self, env, result: RoundResult, started: bool) -> None:
        key = (result.condition, result.seed)
        self.session.log(
            'round_end', condition=result.condition, seed=result.seed,
            outcome=result.outcome, steps=result.steps, started=started,
        )
        keep = result.outcome == 'success' or (
            result.outcome == 'timeout' and self.config.output.save_failures
        )
        if started and keep and self.buffer is not None and len(self.buffer):
            demo = self.buffer.to_demo(self.build_meta(result))
            path = self.session.write_demo(demo)
            self._say(f'saved {len(self.session.saved)}: {path.name} ({result.outcome})')
            self.retries.pop(key, None)
        elif result.outcome == 'retry':
            self.retries[key] = self.retries.get(key, 0) + 1
        self.buffer = None

    def on_event(self, env, event) -> None:
        if event.type == pygame.KEYDOWN and event.key == pygame.K_u:
            undone = self.session.undo_last()
            self._say(f'undid {undone[:8]}' if undone else 'nothing to undo')

    def status(self) -> str:
        return f'saved {len(self.session.saved) - len(self.session.undone)}  {self.message}'

    def build_meta(self, result: RoundResult) -> DemoMeta:
        return DemoMeta(
            demo_id=uuid.uuid4().hex,
            session_id=self.session.session_id,
            condition=result.condition,
            seed=result.seed,
            variation={},  # filled from the buffer by StepBuffer.to_demo
            outcome=result.outcome,
            steps=result.steps,
            seconds=round(result.seconds, 3),
            attempt=1 + self.retries.get((result.condition, result.seed), 0),
            reset_options=self.reset_options,
            condition_info=self.condition_info,
            device=device_info(self.device),
            operator=self.config.output.operator,
            created_at=now_iso(),
            package_version=package_version(),
            git_commit=git_commit(),
        )


def record(config: RecordConfig, env=None, device=None, get_events=pygame.event.get,
           verbose=True):
    """Run one recording session. Returns (session_writer, round results)."""
    play = config.sandbox
    if play.shuffle_conditions and play.shuffle_seed is None:
        # pin the random order so the session snapshot records it
        play.shuffle_seed = random.SystemRandom().randrange(2**31)
    device = device or InputDevice(**play.device)
    snapshot = {
        **asdict(play),
        'output': asdict(config.output),
        'condition_order': [c.name for c in sandbox.condition_order(play)],
    }
    with SessionWriter(config.output, snapshot) as session:
        recorder = DemoRecorder(session, device, config, verbose=verbose)
        results = sandbox.run_session(play, env=env, device=device,
                                      get_events=get_events, hooks=recorder)
    return session, results


def run(argv=None):
    parser = argparse.ArgumentParser(description='Record PushT demos with a gamepad.')
    parser.add_argument('-c', '--config', required=True, help='record YAML (e.g. configs/demo.yaml)')
    args = parser.parse_args(argv)
    config = load_record_config(args.config)

    pygame.init()
    try:
        device = InputDevice(**config.sandbox.device)
        print(f'input: {device.kind} ({device.name})')
        session, results = record(config, device=device)
    finally:
        pygame.quit()
    print(sandbox.summarize(results))
    kept = len(session.saved) - len(session.undone)
    print(f'{kept} demos kept ({len(session.undone)} undone) in {session.path}')
    return session, results


def main(argv=None):
    """Console entry point (returns None so the exit status is 0)."""
    run(argv)


if __name__ == '__main__':
    main()
