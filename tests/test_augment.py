"""Augmenters (gen/augment.py), the simulate stage (gen/simulate.py) and the
recipe runner (gen/recipe.py), on parents recorded headless with a scripted stick."""

from pathlib import Path

import gymnasium as gym
import numpy as np
import pygame
import pytest
import yaml

import pusht_scenarios.env  # noqa: F401
from pusht_scenarios.env.bounds import KeepAgentOnBoard
from pusht_scenarios.gen import augment as A
from pusht_scenarios.gen import recipe as R
from pusht_scenarios.gen import replay
from pusht_scenarios.gen.simulate import InvalidCandidate, check_candidate, simulate, wall_margin
from pusht_scenarios.io.demo import find_demos, load_demo, save_demo
from pusht_scenarios.teleop import record
from pusht_scenarios.teleop.input import InputDevice
from pusht_scenarios.teleop.viewer import derived_label

CONFIGS = Path(__file__).parents[1] / 'configs' / 'augment'
C = A.BOARD_CENTER


class SteeringStick:
    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)
        self.env = None

    def get_axis(self, i):
        b = self.env.unwrapped
        d = np.array(b.block.position) - np.array(b.agent.position)
        v = d / (np.linalg.norm(d) + 1e-9) + self.rng.normal(0, 0.4, 2)
        return float(np.clip(v[i], -1, 1))

    def get_button(self, i):
        return False


CONDITIONS = [
    {'name': 'default'},
    {'name': 'heavy', 'variation_values': {'sim.agent_max_force': 8000.0, 'sim.block_mass': 3.0}},
    {'name': 'slippery', 'variation_values': {'sim.damping': 1.0, 'sim.floor_friction': 0.4}},
    {'name': 'cog', 'variation_values': {'sim.block_cog_offset': [20.0, 5.0]}},
    {'name': 'plus', 'variation_values': {'block.shape': 7}},
    {'name': 'zed', 'variation_values': {'block.shape': 3}},
]


@pytest.fixture(scope='module')
def parents_dir(tmp_path_factory):
    """A recorded session: 6 conditions x 2 seeds, 40-step contact-rich demos."""
    pygame.init()
    root = tmp_path_factory.mktemp('raw')
    env = KeepAgentOnBoard(gym.make('pts/PushT-v0', render_mode='human'))
    stick = SteeringStick()
    stick.env = env
    config = record.record_config_from_dict({
        'rounds_per_condition': 2, 'max_steps': 40, 'fps': 0, 'wait_for_input': False,
        'output': {'root': str(root), 'session_name': 'parents'}, 'conditions': CONDITIONS,
    })
    record.record(config, env=env, device=InputDevice(joystick=stick),
                  get_events=lambda: [], verbose=False)
    env.close()
    pygame.quit()
    return root


@pytest.fixture(scope='module')
def parents(parents_dir):
    return {(d.meta.condition, d.meta.seed): d for d in map(load_demo, find_demos(parents_dir))}


@pytest.fixture
def env():
    env = gym.make('pts/PushT-v0')
    yield env
    env.close()


def reset(env, c):
    from pusht_scenarios.gen.simulate import reset_candidate
    reset_candidate(env, c)
    return env.unwrapped


def block_world(base):
    return [np.array([base.block.local_to_world(v) for v in s.get_vertices()]) for s in base.block.shapes]


def goal_world(base):
    x, y, a = base.goal_pose
    Rm = A.rotation(a)
    return [np.array([Rm @ np.asarray(v) + (x, y) for v in s.get_vertices()]) for s in base.block.shapes]


def inside(base, points, pose=None):
    """Signed distance of world points to the block (or to the block at pose)."""
    out = []
    for p in points:
        if pose is not None:  # express p in the block frame at pose, then in the live body's frame
            local = A.rotation(-pose[2]) @ (np.asarray(p) - pose[:2])
            p = base.block.local_to_world(tuple(local))
        out.append(min(s.point_query(tuple(p)).distance for s in base.block.shapes))
    return np.array(out)


# --- transform geometry -------------------------------------------------------


@pytest.mark.parametrize('theta, shift', [(np.pi / 2, (0, 0)), (0.4, (30, -20)), (-2.0, (-15, 25))])
def test_se2_moves_block_goal_and_actions_rigidly(env, parents, theta, shift):
    d = parents[('default', 0)]
    root = A.from_demo(d)
    p0 = reset(env, A.noise(root, 0.0, np.random.default_rng(0)))
    before_block, before_goal = block_world(p0), goal_world(p0)
    c = A.se2(root, theta, shift)
    p1 = reset(env, c)
    Rm = A.rotation(theta)
    for b, a in zip(before_block, block_world(p1)):
        np.testing.assert_allclose(a, (Rm @ (b - C).T).T + C + shift, atol=1e-6)
    for b, a in zip(before_goal, goal_world(p1)):
        np.testing.assert_allclose(a, (Rm @ (b - C).T).T + C + shift, atol=1e-6)
    np.testing.assert_allclose(c.actions, (Rm @ d.action.T).T, atol=1e-6)
    assert c.explicit and c.chain == [{'op': 'se2', 'theta': theta, 'shift': list(map(float, shift))}]


@pytest.mark.parametrize('cond, axis', [('default', 90), ('default', 0), ('default', 30), ('plus', 45), ('cog', 90)])
def test_mirror_reflects_block_and_goal_shapes(env, parents, cond, axis):
    d = parents[(cond, 0)]
    root = A.from_demo(d)
    p0 = reset(env, A.noise(root, 0.0, np.random.default_rng(0)))
    local = A.local_mirror_axes(env)[0]
    rng = np.random.default_rng(1)
    pts = rng.uniform(20, 490, size=(400, 2))
    before = inside(p0, pts)
    before_goal = inside(p0, pts, pose=np.array(p0.goal_pose))
    offset = np.array(d.meta.variation['sim.block_cog_offset'])
    c = A.mirror(root, axis, local)
    p1 = reset(env, c)
    M = A.reflection(np.radians(axis))
    mirrored = (M @ (pts - C).T).T + C
    clear = (np.abs(before) > 0.5) & (np.abs(before_goal) > 0.5)
    assert np.all((inside(p1, mirrored)[clear] < 0) == (before[clear] < 0))
    goal_after = inside(p1, mirrored, pose=np.array(p1.goal_pose))
    assert np.all((goal_after[clear] < 0) == (before_goal[clear] < 0))
    np.testing.assert_allclose(c.actions, (M @ d.action.T).T, atol=1e-6)
    S = A.reflection(np.radians(local))
    np.testing.assert_allclose(c.variation['sim.block_cog_offset'], S @ offset)


def test_double_mirror_and_inverse_rotation_are_identity(parents):
    d = parents[('default', 1)]
    root = A.from_demo(d)
    twice = A.mirror(A.mirror(root, 90, 90), 90, 90)
    back = A.se2(A.se2(root, 0.7, (10, -5)), -0.7)
    for c in (twice,):
        np.testing.assert_allclose(c.start_state, root.start_state, atol=1e-9)
        np.testing.assert_allclose(c.actions, root.actions, atol=1e-6)
        np.testing.assert_allclose(c.variation['goal.position'], root.variation['goal.position'], atol=1e-9)
    shift_back = A.rotation(-0.7) @ np.array([10.0, -5.0])
    np.testing.assert_allclose(back.start_state[2:4], root.start_state[2:4] + shift_back, atol=1e-9)


@pytest.mark.parametrize('shape, axes', [(1, []), (2, [90.0]), (3, []), (4, [0.0, 45.0, 90.0, 135.0]),
                                         (5, [0.0, 90.0]), (6, [90.0]), (7, [0.0, 45.0, 90.0, 135.0])])
def test_local_mirror_axes(env, shape, axes):
    env.reset(seed=0, options={'variation_values': {'block.shape': np.int64(shape)}})
    assert A.local_mirror_axes(env) == axes


def test_mirror_axes_ignore_cog_offset(env):
    # the offset is mirrored with the block, so it doesn't break the symmetry
    env.reset(seed=0, options={'variation_values': {'sim.block_cog_offset': np.array([20.0, 5.0])}})
    assert A.local_mirror_axes(env) == [90.0]


# --- dynamics are (nearly) equivariant ---------------------------------------


@pytest.mark.parametrize('cond', ['default', 'heavy'])
@pytest.mark.parametrize('make', ['rot90', 'mirror90'])
def test_transformed_rollout_tracks_transformed_parent_early(env, parents, cond, make):
    # physics has no preferred direction, so early on the transformed rollout
    # follows the transformed parent (contact chaos may separate them later)
    d = parents[(cond, 0)]
    root = A.from_demo(d)
    if make == 'rot90':
        c, L = A.se2(root, np.pi / 2), A.rotation(np.pi / 2)
    else:
        c, L = A.mirror(root, 90, 90), A.reflection(np.pi / 2)
    out = simulate(env, c)
    expected = (L @ (d.state[:8, 2:4] - C).T).T + C
    np.testing.assert_allclose(out.state[:8, 2:4], expected, atol=0.5)


# --- validity ----------------------------------------------------------------


def test_block_or_agent_in_wall_is_invalid(env, parents):
    c = A.se2(A.from_demo(parents[('default', 0)]), 0.0)
    c.start_state[2:4] = [495.0, 256.0]  # block through the right wall
    assert 'block_outside_walls' in check_candidate(env, c)
    with pytest.raises(InvalidCandidate, match='block_outside_walls'):
        simulate(env, c)
    c = A.se2(A.from_demo(parents[('default', 0)]), 0.0)
    c.start_state[0:2] = [3.0, 256.0]  # agent beyond the left wall
    assert 'agent_outside_walls' in check_candidate(env, c)


def test_agent_overlapping_block_is_invalid(env, parents):
    c = A.se2(A.from_demo(parents[('default', 0)]), 0.0)
    c.start_state[0:2] = c.start_state[2:4]  # agent on top of the block
    assert 'agent_overlaps_block' in check_candidate(env, c)


def test_goal_outside_walls_is_invalid(env, parents):
    c = A.se2(A.from_demo(parents[('default', 0)]), 0.0)
    c.variation['goal.position'] = [52.0, 52.0]
    assert 'goal_outside_walls' in check_candidate(env, c)


def test_goal_out_of_variation_range(env, parents):
    c = A.se2(A.from_demo(parents[('default', 0)]), 0.0)
    c.variation['goal.position'] = [20.0, 256.0]  # below the space's 50 px minimum
    assert check_candidate(env, c) == ['variation_out_of_range']


def test_untransformed_copies_are_not_checked(env, parents):
    # they start exactly like their (played) parent, even if it grazes a wall
    c = A.noise(A.from_demo(parents[('default', 0)]), 0.01, np.random.default_rng(0))
    assert check_candidate(env, c) == []


def test_transformed_start_is_inside_walls(env, parents):
    c = A.se2(A.from_demo(parents[('default', 1)]), 0.3, (20, 10))
    assert check_candidate(env, c) == []
    assert wall_margin(reset(env, c).block.shapes) >= 0


# --- noise / recolor ---------------------------------------------------------


@pytest.mark.parametrize('cond', ['default', 'heavy', 'slippery'])
def test_noise_free_copy_is_the_parent(env, parents, cond):
    d = parents[(cond, 0)]
    out = simulate(env, A.noise(A.from_demo(d), 0.0, np.random.default_rng(0)))
    np.testing.assert_array_equal(out.state, d.state)
    np.testing.assert_array_equal(out.sim_state, d.sim_state)
    np.testing.assert_array_equal(out.action, d.action)


def test_noise_is_seeded_and_bounded(env, parents):
    root = A.from_demo(parents[('default', 0)])
    a = A.noise(root, 0.05, np.random.default_rng(3))
    b = A.noise(root, 0.05, np.random.default_rng(3))
    np.testing.assert_array_equal(a.actions, b.actions)
    assert not np.array_equal(a.actions, root.actions)
    big = A.noise(root, 5.0, np.random.default_rng(0))
    assert np.abs(big.actions).max() <= 1.0
    out = simulate(env, big)
    np.testing.assert_array_equal(out.raw_action, big.actions)  # proposed, pre-bounding
    assert np.abs(out.action).max() <= 1.0


def test_more_noise_drifts_further(env, parents):
    drift = {}
    for sigma in (0.003, 0.03, 0.15):
        errs = []
        for cond in ('default', 'heavy'):
            for seed in (0, 1):
                d = parents[(cond, seed)]
                for k in range(3):
                    out = simulate(env, A.noise(A.from_demo(d), sigma, np.random.default_rng(k)))
                    # early steps: later, contact chaos saturates the drift at any sigma
                    errs.append(np.abs(out.state[1:11, 0:4] - d.state[1:11, 0:4]).mean())
        drift[sigma] = np.mean(errs)
    assert drift[0.003] < drift[0.03] < drift[0.15]


def test_recolor_changes_colors_not_physics(env, parents):
    d = parents[('slippery', 0)]
    palette = {'block.color': [10, 200, 10], 'background.color': [0, 0, 0]}
    c = A.recolor(A.from_demo(d), palette, 'green')
    out = simulate(env, c)
    np.testing.assert_array_equal(out.state, d.state)
    assert out.meta.variation['block.color'] == [10, 200, 10]
    assert not c.explicit  # keeps the parent's exact reset call
    plain = env.render()
    reset(env, c)
    assert not np.array_equal(env.render(), plain)


def test_recolor_rejects_unknown_keys(parents):
    with pytest.raises(ValueError, match='block.colour'):
        A.recolor(A.from_demo(parents[('default', 0)]), {'block.colour': [1, 2, 3]})


def test_recolor_after_transform(env, parents):
    c = A.recolor(A.se2(A.from_demo(parents[('default', 0)]), np.pi), {'block.color': [200, 0, 0]})
    out = simulate(env, c)
    assert out.meta.variation['block.color'] == [200, 0, 0]
    assert replay.divergence(out, replay.replay(out)) == 0.0


# --- simulate output ---------------------------------------------------------


def test_outputs_replay_exactly_and_carry_provenance(env, parents, tmp_path):
    d = parents[('cog', 1)]
    root = A.from_demo(d)
    cands = [A.noise(root, 0.02, np.random.default_rng(0)),
             A.noise(A.se2(root, 1.0, (10, 10)), 0.01, np.random.default_rng(1)),
             A.noise(A.mirror(root, 90, 90), 0.0, np.random.default_rng(2)),
             A.recolor(A.se2(root, -0.5), {'agent.color': [0, 0, 0]})]
    for i, c in enumerate(cands):
        out = load_demo(save_demo(simulate(env, c, copy_index=i, recipe='t'), tmp_path / f'{i}.npz'))
        assert replay.divergence(out, replay.replay(out)) == 0.0
        df = out.meta.derived_from
        assert df['parent'] == d.meta.demo_id and df['copy'] == i and df['recipe'] == 't'
        assert df['chain'] == c.chain and set(df['metrics']) >= {'ever_on_goal', 'min_wall_margin'}
        assert out.meta.outcome in ('success', 'failure')
        assert len(out) == len(d)  # DINO-WM: same length, no early stop


def test_outcome_judged_at_final_step(env, parents):
    d = parents[('default', 0)]
    out = simulate(env, A.noise(A.from_demo(d), 0.0, np.random.default_rng(0)))
    assert out.meta.outcome == ('success' if out.on_goal[-1] else 'failure')


def test_demo_ids_are_deterministic(env, parents):
    c = A.noise(A.from_demo(parents[('default', 0)]), 0.01, np.random.default_rng(0))
    assert simulate(env, c, copy_index=3).meta.demo_id == simulate(env, c, copy_index=3).meta.demo_id
    assert simulate(env, c, copy_index=3).meta.demo_id != simulate(env, c, copy_index=4).meta.demo_id


# --- recipes -----------------------------------------------------------------


def small_recipe(src, out, **kw):
    raw = {'name': 'r', 'sources': [str(src)], 'output_root': str(out), 'seed': 0,
           'bases': {'original': True, 'rotations_deg': [180],
                     'se2': {'count': 1, 'shift_px': [-30, 30]}, 'mirror_axes_deg': [90]},
           'palettes': {'palettes': {'red': {'block.color': [200, 30, 30]}}},
           'noise': [[0.0, 1], [0.02, 2]]}
    raw.update(kw)
    return R.recipe_from_dict(raw)


def test_recipe_defaults_to_dino_wm():
    r = R.recipe_from_dict({'name': 'x'})
    assert r.copies_per_base() == 101 and r.noise[0] == [0.0, 11]
    assert R.recipe_from_dict({'name': 'x', 'noise': 'dino_wm'}).noise == r.noise


@pytest.mark.parametrize('raw, match', [
    ({'name': 'x', 'colour': 1}, 'colour'),
    ({'name': 'x', 'bases': {'rotate': [90]}}, 'rotate'),
    ({'name': 'x', 'bases': {'se2': {'n': 1}}}, 'n'),
    ({'name': 'x', 'palettes': {'apply_to': 'some'}}, 'apply_to'),
    ({'name': 'x', 'noise': [[-0.1, 1]]}, 'sigma'),
])
def test_recipe_validation(raw, match):
    with pytest.raises(ValueError, match=match):
        R.recipe_from_dict(raw)


@pytest.mark.parametrize('path', sorted(CONFIGS.glob('*.yaml')), ids=lambda p: p.name)
def test_shipped_recipes_parse(path):
    r = R.load_recipe(path)
    assert r.name and r.copies_per_base() > 0


def test_select_parents_filters(parents_dir):
    assert len(R.select_parents(small_recipe(parents_dir, '.'))) == 12
    assert len(R.select_parents(small_recipe(parents_dir, '.', conditions=['heavy']))) == 2
    assert len(R.select_parents(small_recipe(parents_dir, '.', parents_per_condition=1))) == 6


def test_select_parents_skips_rejected_and_derived(parents_dir, tmp_path):
    import shutil
    src = tmp_path / 'raw'
    shutil.copytree(parents_dir, src)
    session = next(src.iterdir())
    victim = load_demo(next((session / 'demos').glob('*.npz')))
    with open(session / 'events.jsonl', 'a') as f:
        f.write('{"event": "undo", "demo_id": "%s"}\n' % victim.meta.demo_id)
    derived = simulate(gym.make('pts/PushT-v0'), A.noise(A.from_demo(victim), 0.0, np.random.default_rng(0)))
    save_demo(derived, session / 'demos' / 'zz_derived.npz')
    picked = {load_demo(p).meta.demo_id for p in R.select_parents(small_recipe(src, '.'))}
    assert len(picked) == 11 and victim.meta.demo_id not in picked and derived.meta.demo_id not in picked
    assert victim.meta.demo_id in {load_demo(p).meta.demo_id
                                   for p in R.select_parents(small_recipe(src, '.', include_rejected=True))}


@pytest.fixture(scope='module')
def run_once(parents_dir, tmp_path_factory):
    out = tmp_path_factory.mktemp('aug')
    path = R.run_recipe(small_recipe(parents_dir, out), workers=1, verbose=False)
    return path


def test_recipe_counts_and_summary(run_once):
    summary = yaml.safe_load((run_once / 'summary.yaml').read_text())
    files = find_demos(run_once)
    assert summary['total']['copies'] == len(files)
    by = summary['by_condition']
    # bases: original, rot180, se2, mirror90 (+ red on original) = 5; 'zed' has no mirror axis
    for cond in ('default', 'heavy', 'slippery', 'cog', 'plus'):
        assert by[cond]['copies'] <= 2 * 5 * 3
        assert by[cond]['copies'] >= 2 * 3 * 3  # at most a few invalid transforms
    assert by['zed']['skipped'].get('mirror:no_axis') == 2
    assert (run_once / 'recipe.yaml').exists()


def test_recipe_outputs_replay_exactly(run_once):
    files = find_demos(run_once)
    for path in files[::7]:
        d = load_demo(path)
        assert replay.divergence(d, replay.replay(d)) == 0.0
        assert derived_label(d.meta.derived_from).split()[0] == d.meta.derived_from['base']


def test_recipe_is_deterministic_across_workers(parents_dir, run_once, tmp_path):
    again = R.run_recipe(small_recipe(parents_dir, tmp_path), workers=2, verbose=False)
    a, b = find_demos(run_once), find_demos(again)
    assert [p.name for p in a] == [p.name for p in b]
    for pa, pb in zip(a[::5], b[::5]):
        da, db = load_demo(pa), load_demo(pb)
        assert da.meta.demo_id == db.meta.demo_id
        np.testing.assert_array_equal(da.state, db.state)


def test_recipe_refuses_existing_output(parents_dir, run_once):
    with pytest.raises(FileExistsError):
        R.run_recipe(small_recipe(parents_dir, run_once.parent), workers=1, verbose=False)


def test_noise_free_original_copy_equals_parent(run_once, parents):
    for path in find_demos(run_once):
        d = load_demo(path)
        df = d.meta.derived_from
        if df['base'] == 'original' and df['chain'][-1]['sigma'] == 0.0:
            parent = next(p for p in parents.values() if p.meta.demo_id == df['parent'])
            np.testing.assert_array_equal(d.state, parent.state)


def test_suite_splits_filter(mini):
    picked = R.select_parents(R.recipe_from_dict({'name': 'x', 'sources': [str(mini['raw'])],
                                                  'suite_splits': ['train']}))
    assert len(picked) == 12
    assert {load_demo(p).meta.condition_info['split'] for p in picked} == {'train'}


def test_suite_splits_from_a_built_suite(mini, tmp_path):
    # the split comes from the suite file, not the one stored at recording time
    from pusht_scenarios import suite as S
    from conftest import MINI_SUITE

    S.write(S.build(S.apply_overrides(S.spec_from_dict(MINI_SUITE), ['mass=heavy'])), tmp_path)
    picked = R.select_parents(R.recipe_from_dict({'name': 'x', 'sources': [str(mini['raw'])],
                                                  'suite_splits': ['train'],
                                                  'suite': str(tmp_path / 'suite.yaml')}))
    conds = {load_demo(p).meta.condition for p in picked}
    assert conds and all('mass-heavy' not in c for c in conds)


def test_recipe_suite_restricts_to_its_envs(mini, tmp_path):
    import shutil

    src = tmp_path / 'raw'
    shutil.copytree(mini['raw'], src)
    other = load_demo(next(iter(sorted(src.rglob('*.npz')))))
    other.meta.condition, other.meta.demo_id = 'not_in_suite', 'e' * 32
    save_demo(other, src / 'extra' / 'demos' / 'x.npz')
    picked = R.select_parents(R.recipe_from_dict({'name': 'x', 'sources': [str(src)],
                                                  'suite': str(mini['suite_path'])}))
    assert len(picked) == 20
    assert 'not_in_suite' not in {load_demo(p).meta.condition for p in picked}
