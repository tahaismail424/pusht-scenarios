"""Curation (io/collection.py, pts-data): index, reject/inherit, select, merge,
end-trims, verify, and the viewer's reject key."""

import shutil

import numpy as np
import pygame
import pytest
import yaml

import pusht_scenarios.env  # noqa: F401
from pusht_scenarios.gen import replay
from pusht_scenarios.io import cli
from pusht_scenarios.io import collection as C
from pusht_scenarios.io.demo import find_demos, load_demo, load_meta
from pusht_scenarios.teleop import viewer


@pytest.fixture
def data(mini, tmp_path):
    """A private copy of the mini raw + aug collections (tests write curation files)."""
    raw, aug = tmp_path / 'raw', tmp_path / 'aug'
    shutil.copytree(mini['raw'], raw)
    shutil.copytree(mini['aug'], aug)
    return raw, aug


def test_load_meta_matches_full_load(mini):
    for path in find_demos(mini['raw'])[:3]:
        assert load_meta(path) == load_demo(path).meta


def test_index_scan_and_record_properties(data):
    raw, aug = data
    index = C.DemoIndex.scan(raw, aug)
    recorded = index.select(derived=False)
    derived = index.select(derived=True)
    assert len(recorded) == 20
    # 2 bases (original, rot180) x 2 noise levels per parent; a few rot180 may be invalid
    assert 20 * 3 <= len(derived) <= 20 * 4
    ids = {r.meta.demo_id for r in recorded}
    for r in derived:
        assert r.parent_id in ids and r.base in ('original', 'rot180')
        assert r.noise_std in (0.0, 0.05)
        assert r.is_clean_original == (r.base == 'original' and r.noise_std == 0.0)
    assert all(r.is_clean_original and r.parent_id == r.meta.demo_id for r in recorded)


def test_reject_writes_curation_and_is_inherited(data):
    raw, aug = data
    index = C.DemoIndex.scan(raw, aug)
    victim = list(index.select(derived=False))[0]
    C.reject(index, [victim.meta.demo_id[:10]], 'sloppy')
    cur = yaml.safe_load((victim.collection / 'curation.yaml').read_text())
    assert cur['rejected'][victim.meta.demo_id]['reason'] == 'sloppy'

    again = C.DemoIndex.scan(raw, aug)
    assert again.find(victim.meta.demo_id).rejected
    children = [r for r in again if r.is_derived and r.parent_id == victim.meta.demo_id]
    assert children and all(r.rejected and 'parent rejected' in r.reject_reason for r in children)
    assert len(again.select()) == len(again) - 1 - len(children)

    C.unreject(again, [victim.meta.demo_id])
    assert not C.DemoIndex.scan(raw, aug).find(victim.meta.demo_id).rejected


def test_session_undo_counts_as_reject(data):
    raw, _ = data
    index = C.DemoIndex.scan(raw)
    r = list(index)[0]
    with open(r.collection / 'events.jsonl', 'a') as f:
        f.write('{"event": "undo", "demo_id": "%s"}\n' % r.meta.demo_id)
    again = C.DemoIndex.scan(raw)
    assert again.find(r.meta.demo_id).reject_reason == 'undone while recording'
    assert C.unreject(again, [r.meta.demo_id]) == [r.meta.demo_id]  # undo can't be unrejected
    assert C.DemoIndex.scan(raw).find(r.meta.demo_id).rejected


def test_find_requires_unique_prefix(data):
    index = C.DemoIndex.scan(*data)
    with pytest.raises(KeyError):
        index.find('')


def test_select_filters(data):
    index = C.DemoIndex.scan(*data)
    cond = list(index)[0].meta.condition
    assert {r.meta.condition for r in index.select(conditions=[cond])} == {cond}
    assert {r.meta.outcome for r in index.select(outcomes=['success'])} == {'success'}
    assert all(r.base == 'rot180' for r in index.select(where=lambda r: r.base == 'rot180'))


def test_summary_counts(data):
    index = C.DemoIndex.scan(*data)
    text = index.summary()
    total = text.splitlines()[-1].split()
    assert total[0] == 'total' and int(total[1]) == 20 and int(total[2]) == len(index) - 20


def test_merge_copies_kept_and_is_idempotent(data, tmp_path):
    raw, aug = data
    index = C.DemoIndex.scan(raw, aug)
    victim = list(index.select(derived=False))[0]
    C.reject(index, [victim.meta.demo_id])
    dest = tmp_path / 'merged'
    C.merge([raw, aug], dest)
    kept = len(C.DemoIndex.scan(raw, aug).select())
    assert len(find_demos(dest)) == kept
    assert victim.meta.demo_id not in {load_meta(p).demo_id for p in find_demos(dest)}
    C.merge([raw, aug], dest)  # again: nothing added
    assert len(find_demos(dest)) == kept
    manifest = yaml.safe_load((dest / 'collection.yaml').read_text())
    assert [m['added'] for m in manifest['merges']] == [kept, 0] and manifest['total'] == kept


def test_end_trim_replays_exactly(data):
    raw, _ = data
    # a long (train-env, 25-step) recording, whatever the scan order
    rec = next(r for r in C.DemoIndex.scan(raw).select(derived=False) if r.meta.steps > 10)
    demo = load_demo(rec.path)
    cut = C.trim(demo, end=len(demo) - 5)
    assert len(cut) == len(demo) - 5 and cut.meta.steps == len(cut)
    np.testing.assert_array_equal(cut.state, demo.state[:len(cut) + 1])
    assert cut.meta.derived_from['parent'] == demo.meta.demo_id
    assert cut.meta.derived_from['chain'][-1] == {'op': 'trim', 'end': len(demo) - 5}
    assert cut.meta.outcome == ('success' if cut.on_goal[-1] else 'failure')
    assert replay.divergence(cut, replay.replay(cut)) == 0.0
    assert C.trim(demo, end=len(demo) - 5).meta.demo_id == cut.meta.demo_id  # deterministic


def test_trim_refuses_start_and_bad_ends(data):
    demo = load_demo(find_demos(data[0])[0])
    with pytest.raises(NotImplementedError):
        C.trim(demo, end=5, start=2)
    for end in (0, len(demo) + 1):
        with pytest.raises(ValueError):
            C.trim(demo, end=end)


def test_trim_file_rejects_parent(data):
    raw, _ = data
    rec = list(C.DemoIndex.scan(raw).select(derived=False))[0]
    path = C.trim_file(rec, end=4)
    index = C.DemoIndex.scan(raw)
    assert index.find(rec.meta.demo_id).rejected
    child = index.find(load_meta(path).demo_id)
    assert not child.rejected and child.meta.steps == 4  # the trimmed copy is kept
    C.trim_file(list(C.DemoIndex.scan(raw).select(derived=False))[0], end=3, keep_parent=True)


def test_verify(data):
    results = C.verify(data[0])
    assert len(results) == 20 and all(err == 0.0 for _, err in results)


def test_cli_commands(data, tmp_path, capsys):
    raw, aug = data
    index = cli.run(['ls', str(raw), str(aug), '-v'])
    out = capsys.readouterr().out
    assert 'total' in out and len(index) > 20
    first = list(C.DemoIndex.scan(raw))[0].meta.demo_id
    cli.run(['reject', first[:8], '--root', str(raw), '--reason', 'meh'])
    assert C.DemoIndex.scan(raw).find(first).reject_reason == 'meh'
    assert len(cli.run(['ls', str(raw)])) == 19
    assert len(cli.run(['ls', str(raw), '--all'])) == 20
    cli.run(['unreject', first[:8], '--root', str(raw)])
    second = list(C.DemoIndex.scan(raw))[1]
    cli.run(['trim', second.meta.demo_id[:8], '--end', '3', '--root', str(raw)])
    assert C.DemoIndex.scan(raw).find(second.meta.demo_id).rejected
    cli.run(['merge', str(raw), '--out', str(tmp_path / 'm')])
    assert len(find_demos(tmp_path / 'm')) == 20  # 19 kept originals + 1 trimmed copy
    results = cli.run(['verify', str(tmp_path / 'm')])
    assert all(e == 0.0 for _, e in results)
    assert '20/20 demos replay exactly' in capsys.readouterr().out


def test_viewer_x_key_toggles_reject(data):
    raw, _ = data
    pygame.init()
    first = find_demos(raw)[0]
    x = pygame.event.Event(pygame.KEYDOWN, key=pygame.K_x)
    q = pygame.event.Event(pygame.KEYDOWN, key=pygame.K_q)
    frames = [[x], [x], [], [q]]  # reject, unreject (on the replayed demo), then quit
    states = []
    original = C.reject_in

    def spy(*a, **k):
        states.append('reject')
        return original(*a, **k)

    C.reject_in = spy
    try:
        viewer.view([first], speed=0, verbose=False, get_events=lambda: frames.pop(0) if frames else [])
    finally:
        C.reject_in = original
    assert states == ['reject']
    assert not C.DemoIndex.scan(raw).find(load_meta(first).demo_id).rejected
    frames[:] = [[x], [q]]
    viewer.view([first], speed=0, verbose=False, get_events=lambda: frames.pop(0) if frames else [])
    assert C.DemoIndex.scan(raw).find(load_meta(first).demo_id).reject_reason == 'rejected in pts-replay'
    pygame.quit()
