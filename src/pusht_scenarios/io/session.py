# One recording session on disk.
#
#   <root>/<session_id>/
#     session.yaml     config snapshot (the exact config used), operator,
#                      start time, package version + git commit
#     events.jsonl     append-only log: session start/end, round start/end,
#                      demo saved, undo; one JSON object per line
#     demos/
#       0001_<condition>_seed<k>_<id8>.npz   one per demo, written as the round ends
#
# session_id: '<YYYYmmdd-HHMMSS>[-<name>]' so sessions sort by time.
# Nothing is ever deleted: "undo last" writes an 'undo' event, and readers
# treat that demo as rejected (rejected_ids).
import datetime
import json
import subprocess
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path

import yaml

from pusht_scenarios.io.demo import Demo, save_demo


@dataclass
class OutputConfig:
    root: str = 'data/raw'  # sessions are created under this directory
    session_name: str = ''  # suffix for the session id (e.g. 'pilot')
    save_failures: bool = True  # also keep timeouts (meta.outcome = 'timeout')
    operator: str = ''
    # skip rounds (condition, seed) that already have a kept demo under root,
    # so a long suite can be recorded over several sessions
    resume: bool = False


def now_iso() -> str:
    return datetime.datetime.now().astimezone().isoformat(timespec='seconds')


def package_version() -> str:
    try:
        return metadata.version('pusht-scenarios')
    except metadata.PackageNotFoundError:
        return ''


def git_commit() -> str:
    """HEAD of the repo this package lives in ('' if unavailable), '+dirty' if
    there are uncommitted changes."""
    here = Path(__file__).resolve().parent
    try:
        sha = subprocess.run(
            ['git', 'rev-parse', 'HEAD'], cwd=here, capture_output=True,
            text=True, timeout=5, check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ['git', 'status', '--porcelain', '--untracked-files=no'], cwd=here,
            capture_output=True, text=True, timeout=5,
        ).stdout.strip()
        return sha + ('+dirty' if dirty else '')
    except (OSError, subprocess.SubprocessError):
        return ''


class SessionWriter:
    """Owns one session directory; use as a context manager."""

    def __init__(self, output: OutputConfig, config_snapshot: dict, session_id=None):
        self.output = output
        self.snapshot = config_snapshot
        stamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
        name = output.session_name
        self.session_id = session_id or (f'{stamp}-{name}' if name else stamp)
        self.path = Path(output.root) / self.session_id
        if session_id is None:  # two sessions in the same second: add a suffix
            base, k = self.session_id, 2
            while self.path.exists():
                self.session_id = f'{base}-{k}'
                self.path = Path(output.root) / self.session_id
                k += 1
        self.saved: list[tuple[str, Path]] = []  # (demo_id, path), in save order
        self.undone: list[str] = []
        self._events = None

    def __enter__(self) -> 'SessionWriter':
        if self.path.exists():
            raise FileExistsError(f'session directory already exists: {self.path}')
        (self.path / 'demos').mkdir(parents=True)
        info = {
            'session_id': self.session_id,
            'operator': self.output.operator,
            'started_at': now_iso(),
            'package_version': package_version(),
            'git_commit': git_commit(),
            'config': self.snapshot,
        }
        with open(self.path / 'session.yaml', 'w') as f:
            yaml.safe_dump(info, f, sort_keys=False)
        self._events = open(self.path / 'events.jsonl', 'a')
        self.log('session_start')
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        try:
            self.log(
                'session_end',
                saved=len(self.saved),
                undone=len(self.undone),
                error=repr(exc) if exc else None,
            )
        finally:
            self._events.close()
            self._events = None

    def log(self, event: str, **fields) -> None:
        if self._events is None:
            raise RuntimeError('session is not open')
        record = {'t': now_iso(), 'event': event, **fields}
        self._events.write(json.dumps(record) + '\n')
        self._events.flush()

    def demo_path(self, demo: Demo) -> Path:
        m = demo.meta
        name = f'{len(self.saved) + 1:04d}_{m.condition}_seed{m.seed}_{m.demo_id[:8]}.npz'
        return self.path / 'demos' / name

    def write_demo(self, demo: Demo) -> Path:
        path = save_demo(demo, self.demo_path(demo))
        self.saved.append((demo.meta.demo_id, path))
        self.log(
            'demo_saved', demo_id=demo.meta.demo_id, file=path.name,
            condition=demo.meta.condition, seed=demo.meta.seed,
            outcome=demo.meta.outcome, steps=demo.meta.steps,
        )
        return path

    def undo_last(self) -> str | None:
        """Reject the most recent saved demo that isn't already undone."""
        for demo_id, path in reversed(self.saved):
            if demo_id not in self.undone:
                self.undone.append(demo_id)
                self.log('undo', demo_id=demo_id, file=path.name)
                return demo_id
        return None


def read_events(session_dir) -> list[dict]:
    path = Path(session_dir) / 'events.jsonl'
    if not path.exists():
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def rejected_ids(session_dir) -> set[str]:
    """Demo ids undone during the session."""
    return {e['demo_id'] for e in read_events(session_dir) if e['event'] == 'undo'}


def session_dir_of(demo_path) -> Path | None:
    """The session directory a demo file belongs to (…/<session>/demos/x.npz)."""
    p = Path(demo_path).resolve()
    if p.parent.name == 'demos' and (p.parent.parent / 'session.yaml').exists():
        return p.parent.parent
    return None
