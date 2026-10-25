# Working with many demos: index, curate, merge, trim, verify.
#
# A *collection* is a directory holding demos/ (a recording session or an
# augmentation run) or any directory of .npz demos. Nothing here edits a demo
# file in place; curation is a selection, decided at export time:
#
#   - reject/unreject:  <collection>/curation.yaml lists rejected demo_ids with
#                       a reason. Undo events from a recording session
#                       (events.jsonl) count as rejects too. A derived demo
#                       (augmented/trimmed) inherits its parent's rejection.
#   - merge:            copy the kept demos of several sources into a new
#                       collection (collection.yaml records where they came
#                       from). Idempotent by demo_id.
#   - trim:             write a NEW demo over steps [0, end) of a parent, with
#                       derived_from provenance. Only end-trims: they replay
#                       exactly from reset. A start-trim would need a replay
#                       from a restored mid-episode state, which isn't exact
#                       (io/demo.restore_sim_state), so it's refused.
#   - verify:           replay demos and report the state error (should be 0).
#
# "Concatenating runs" means merging sets of demos; two demos can't be joined
# end to end (the second wouldn't start where the first ended).
import shutil
import uuid
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from pathlib import Path

import yaml

from pusht_scenarios.io.demo import Demo, DemoMeta, find_demos, load_demo, load_meta, save_demo
from pusht_scenarios.io.session import now_iso, rejected_ids as session_undo_ids

CURATION_FILE = 'curation.yaml'
TRIM_NAMESPACE = uuid.UUID('0b6f0a9c-3f0e-4f7c-8e5e-7d1c2b9a4e11')


def is_recorded(meta: DemoMeta) -> bool:
    """A recording, or an end-trim of one: trims *replace* their source demo, so
    for splitting, augmentation and reject inheritance they act as recordings.
    Augmented copies are not recorded."""
    if not meta.derived_from:
        return True
    return all(op.get('op') == 'trim' for op in meta.derived_from.get('chain', [])) and \
        bool(meta.derived_from.get('chain'))


def collection_dir_of(path) -> Path:
    """<collection>/demos/x.npz -> <collection>; otherwise the file's directory."""
    p = Path(path).resolve()
    return p.parent.parent if p.parent.name == 'demos' else p.parent


def read_curation(collection) -> dict:
    """{demo_id: reason} from <collection>/curation.yaml."""
    path = Path(collection) / CURATION_FILE
    if not path.exists():
        return {}
    data = yaml.safe_load(path.read_text()) or {}
    return {k: (v or {}).get('reason', '') for k, v in (data.get('rejected') or {}).items()}


def _write_curation(collection, rejected: dict) -> None:
    path = Path(collection) / CURATION_FILE
    body = {'rejected': {k: {'reason': v['reason'], 'at': v['at']} for k, v in sorted(rejected.items())}}
    tmp = path.with_suffix('.tmp')
    tmp.write_text(yaml.safe_dump(body, sort_keys=False))
    tmp.replace(path)


def _curation_entries(collection) -> dict:
    path = Path(collection) / CURATION_FILE
    if not path.exists():
        return {}
    data = yaml.safe_load(path.read_text()) or {}
    return {k: dict(v or {}) for k, v in (data.get('rejected') or {}).items()}


def reject_in(collection, demo_ids, reason: str = '') -> None:
    entries = _curation_entries(collection)
    for demo_id in demo_ids:
        entries[demo_id] = {'reason': reason, 'at': now_iso()}
    _write_curation(collection, entries)


def unreject_in(collection, demo_ids) -> list[str]:
    """Remove curation rejects. Returns ids that stay rejected by a session undo."""
    entries = _curation_entries(collection)
    for demo_id in demo_ids:
        entries.pop(demo_id, None)
    _write_curation(collection, entries)
    undone = session_undo_ids(collection)
    return [d for d in demo_ids if d in undone]


def collection_rejects(collection) -> dict:
    """{demo_id: reason}: curation.yaml plus session undo events."""
    out = {d: 'undone while recording' for d in session_undo_ids(collection)}
    out.update(read_curation(collection))
    return out


@dataclass
class DemoRecord:
    """One row of the index: where a demo lives, its metadata, its status."""

    path: Path
    meta: DemoMeta
    collection: Path
    rejected: bool = False
    reject_reason: str = ''

    @property
    def is_derived(self) -> bool:
        """True for augmented copies; recordings and their trims are not derived."""
        return not is_recorded(self.meta)

    @property
    def parent_id(self) -> str:
        return self.meta.derived_from['parent'] if self.is_derived else self.meta.demo_id

    @property
    def base(self) -> str:
        """'original' for recorded demos, else the augmentation base (or op)."""
        if not self.is_derived:
            return 'original'
        df = self.meta.derived_from
        return df.get('base') or (df.get('chain') or [{}])[-1].get('op', 'derived')

    @property
    def noise_std(self) -> float:
        if not self.is_derived:
            return 0.0
        sig = [op.get('sigma', 0.0) for op in self.meta.derived_from.get('chain', []) if op.get('op') == 'noise']
        return float(sig[-1]) if sig else 0.0

    @property
    def is_clean_original(self) -> bool:
        """The recorded demo itself, or a noise-free untransformed copy of it."""
        return self.base == 'original' and self.noise_std == 0.0


class DemoIndex:
    """A filterable table of DemoRecords (reads only metadata)."""

    def __init__(self, records: list[DemoRecord]):
        self.records = records

    @classmethod
    def scan(cls, *roots, inherit_rejects: bool = True) -> 'DemoIndex':
        records, rejects = [], {}
        for path in find_demos(*roots):
            collection = collection_dir_of(path)
            if collection not in rejects:
                rejects[collection] = collection_rejects(collection)
            meta = load_meta(path)
            reason = rejects[collection].get(meta.demo_id)
            records.append(DemoRecord(path, meta, collection, reason is not None, reason or ''))
        index = cls(records)
        if inherit_rejects:
            index._inherit_rejects()
        return index

    def _inherit_rejects(self) -> None:
        rejected_parents = {r.meta.demo_id: r.reject_reason for r in self.records
                            if r.rejected and not r.is_derived}
        for r in self.records:
            if r.is_derived and not r.rejected and r.parent_id in rejected_parents:
                r.rejected = True
                r.reject_reason = f'parent rejected: {rejected_parents[r.parent_id]}'.rstrip(': ')

    def __iter__(self):
        return iter(self.records)

    def __len__(self) -> int:
        return len(self.records)

    def find(self, demo_id_prefix: str) -> DemoRecord:
        hits = [r for r in self.records if r.meta.demo_id.startswith(demo_id_prefix)]
        if len(hits) != 1:
            raise KeyError(f'{demo_id_prefix!r} matches {len(hits)} demos')
        return hits[0]

    def select(self, *, conditions=None, outcomes=None, derived=None,
               include_rejected=False, where=None) -> 'DemoIndex':
        """derived: None (both), False (recorded only), True (augmented/trimmed only)."""
        out = []
        for r in self.records:
            if not include_rejected and r.rejected:
                continue
            if conditions is not None and r.meta.condition not in conditions:
                continue
            if outcomes is not None and r.meta.outcome not in outcomes:
                continue
            if derived is not None and r.is_derived != derived:
                continue
            if where is not None and not where(r):
                continue
            out.append(r)
        return DemoIndex(out)

    def summary(self) -> str:
        rows = defaultdict(Counter)
        for r in self.records:
            c = rows[r.meta.condition]
            c['rejected' if r.rejected else ('derived' if r.is_derived else 'recorded')] += 1
            if not r.rejected:
                c[r.meta.outcome] += 1
                c['steps'] += r.meta.steps
        lines = [f'{"condition":<48} {"recorded":>8} {"derived":>8} {"rejected":>8} '
                 f'{"success":>8} {"fail/to":>8} {"steps":>9}']
        total = Counter()
        for cond in sorted(rows):
            c = rows[cond]
            total.update(c)
            fails = c['failure'] + c['timeout']
            lines.append(f'{cond:<48} {c["recorded"]:>8} {c["derived"]:>8} {c["rejected"]:>8} '
                         f'{c["success"]:>8} {fails:>8} {c["steps"]:>9}')
        fails = total['failure'] + total['timeout']
        lines.append(f'{"total":<48} {total["recorded"]:>8} {total["derived"]:>8} {total["rejected"]:>8} '
                     f'{total["success"]:>8} {fails:>8} {total["steps"]:>9}')
        return '\n'.join(lines)


def reject(index: DemoIndex, demo_ids, reason: str = '') -> list[DemoRecord]:
    """Reject demos (ids or unique id prefixes) in their own collections."""
    hits = [index.find(d) for d in demo_ids]
    by_collection = defaultdict(list)
    for r in hits:
        by_collection[r.collection].append(r.meta.demo_id)
    for collection, ids in by_collection.items():
        reject_in(collection, ids, reason)
    return hits


def unreject(index: DemoIndex, demo_ids) -> list[str]:
    hits = [index.find(d) for d in demo_ids]
    still = []
    by_collection = defaultdict(list)
    for r in hits:
        by_collection[r.collection].append(r.meta.demo_id)
    for collection, ids in by_collection.items():
        still += unreject_in(collection, ids)
    return still


def merge(sources, dest, *, include_rejected=False, select=None) -> Path:
    """Copy kept demos from sources (paths or a DemoIndex) into dest/demos/.

    select: optional DemoRecord -> bool filter. Re-merging into the same dest
    skips demos already there (by demo_id). Writes dest/collection.yaml."""
    index = sources if isinstance(sources, DemoIndex) else DemoIndex.scan(*sources)
    picked = index.select(include_rejected=include_rejected, where=select)
    dest = Path(dest)
    demos_dir = dest / 'demos'
    demos_dir.mkdir(parents=True, exist_ok=True)
    present = {load_meta(p).demo_id for p in demos_dir.glob('*.npz')}
    added = []
    for r in picked:
        if r.meta.demo_id in present:
            continue
        name = r.path.name
        target = demos_dir / name
        if target.exists():  # same file name from another source
            target = demos_dir / f'{r.collection.name}__{name}'
        shutil.copy2(r.path, target)
        present.add(r.meta.demo_id)
        added.append(r)

    manifest_path = dest / 'collection.yaml'
    manifest = yaml.safe_load(manifest_path.read_text()) if manifest_path.exists() else {'merges': []}
    manifest['merges'].append({
        'at': now_iso(),
        'sources': sorted({str(r.collection) for r in picked}),
        'include_rejected': include_rejected,
        'added': len(added),
        'skipped_existing': len(picked) - len(added),
    })
    manifest['total'] = len(present)
    manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False))
    return dest


def trim(demo: Demo, end: int, start: int = 0) -> Demo:
    """A new demo with the first `end` steps of `demo` (end-trims only)."""
    if start != 0:
        raise NotImplementedError(
            'start-trims would need an exact replay from a restored mid-episode '
            'state, which pymunk does not give (io/demo.restore_sim_state)')
    if not 0 < end <= len(demo):
        raise ValueError(f'end must be in 1..{len(demo)}, got {end}')
    m = demo.meta
    parent = m.derived_from['parent'] if m.derived_from else m.demo_id
    chain = [*(m.derived_from or {}).get('chain', []), {'op': 'trim', 'end': int(end)}]
    meta = replace(
        m,
        demo_id=uuid.uuid5(TRIM_NAMESPACE, repr([m.demo_id, end])).hex,
        steps=int(end),
        outcome='success' if demo.on_goal[end - 1] else 'failure',
        seconds=float(demo.t_wall[end - 1]),
        created_at=now_iso(),
        derived_from={**(m.derived_from or {}), 'parent': parent, 'trimmed_from': m.demo_id,
                      'chain': chain, 'base': (m.derived_from or {}).get('base', 'original')},
    )
    out = Demo(
        meta=meta,
        state=demo.state[:end + 1].copy(),
        sim_state=demo.sim_state[:end + 1].copy(),
        action=demo.action[:end].copy(),
        raw_action=demo.raw_action[:end].copy(),
        on_goal=demo.on_goal[:end].copy(),
        t_wall=demo.t_wall[:end].copy(),
    )
    out.validate()
    return out


def trim_file(record: DemoRecord, end: int, keep_parent: bool = False) -> Path:
    """Write the trimmed demo next to its parent; reject the parent unless keep_parent."""
    out = trim(load_demo(record.path), end)
    path = save_demo(out, record.path.with_name(f'{record.path.stem}_trim{end}.npz'))
    if not keep_parent:
        reject_in(record.collection, [record.meta.demo_id], f'trimmed to {end} steps ({out.meta.demo_id[:8]})')
    return path


def verify(*paths) -> list[tuple[Path, float]]:
    """Replay each demo; [(path, max state error)]. 0.0 means exact."""
    from pusht_scenarios.gen import replay

    out, env = [], None
    for path in find_demos(*paths):
        demo = load_demo(path)
        if env is None:
            env = replay.make_env(demo)
        out.append((path, replay.divergence(demo, replay.replay(demo, env=env))))
    if env is not None:
        env.close()
    return out
