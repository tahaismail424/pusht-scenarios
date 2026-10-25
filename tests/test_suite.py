"""Environment suites (suite.py): factorial train envs, single-factor tests,
the holdout knob, generated record configs."""

import copy
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
import yaml

import pusht_scenarios.env  # noqa: F401
from pusht_scenarios import suite as S
from pusht_scenarios.env import variations
from pusht_scenarios.io.demo import find_demos, load_meta
from pusht_scenarios.teleop import record, sandbox

from conftest import MINI_SUITE

V0 = Path(__file__).parents[1] / 'configs' / 'suites' / 'v0.yaml'


def spec(**changes):
    raw = copy.deepcopy(MINI_SUITE)
    raw.update(changes)
    return S.spec_from_dict(raw)


def test_v0_shape():
    suite = S.build(S.load_spec(V0))
    train, test = suite.split('train'), suite.split('test')
    assert len(train) == 16 and len(test) == 4
    assert [e.env_id for e in suite.envs] == list(range(20))
    assert all(e.split == 'train' for e in suite.envs[:16])  # train ids first: embedding index
    assert sorted(e.test_kind for e in test) == ['single:color', 'single:friction',
                                                 'single:mass', 'single:shape']
    assert len({e.name for e in suite.envs}) == 20
    assert len({e.env_hash for e in suite.envs}) == 20


def test_every_env_shares_the_base():
    for e in S.build(S.load_spec(V0)).envs:
        assert e.variation_values['sim.agent_max_force'] == 10000.0


def test_train_envs_never_use_heldout_levels():
    suite = S.build(S.load_spec(V0))
    held = {f.name: f.heldout_level(suite.spec.holdout_policy) for f in suite.spec.factors}
    for e in suite.split('train'):
        assert all(e.levels[f] != held[f] for f in held)
    for e in suite.split('test'):
        (f,) = e.heldout_factors
        assert e.levels[f] == held[f]
        refs = {g.name: g.reference_level(suite.spec.holdout_policy) for g in suite.spec.factors}
        assert all(e.levels[g] == refs[g] for g in refs if g != f)


@pytest.mark.parametrize('policy, level', [('low', 'light'), ('middle', 'normal'), ('high', 'heavy')])
def test_holdout_policies(policy, level):
    built = S.build(spec(holdout_policy=policy))
    mass_test = [e for e in built.split('test') if e.heldout_factors == ['mass']]
    assert [e.levels['mass'] for e in mass_test] == [level]
    assert all(e.levels['mass'] != level for e in built.split('train'))


def test_holdout_override_per_factor_and_suite_wide():
    s = S.apply_overrides(spec(), ['mass=heavy'])
    assert s.factors[0].heldout_level(s.holdout_policy) == 'heavy'
    s = S.apply_overrides(spec(), ['high'])
    assert s.factors[0].heldout_level(s.holdout_policy) == 'heavy'
    assert s.factors[1].heldout_level(s.holdout_policy) == 'night'  # categorical keeps its level
    with pytest.raises(ValueError, match='unknown factor'):
        S.apply_overrides(spec(), ['size=big'])
    with pytest.raises(ValueError, match='policy|one of'):
        S.apply_overrides(spec(), ['sideways'])


def test_categorical_needs_an_explicit_holdout():
    raw = copy.deepcopy(MINI_SUITE)
    del raw['factors']['color']['holdout']
    with pytest.raises(ValueError, match='categorical'):
        S.build(S.spec_from_dict(raw))


def test_test_context_all_and_all_heldout():
    built = S.build(spec(test_context='all', include_all_heldout=True))
    test = built.split('test')
    # mass held out in each of 2 color train levels, color held out in each of 2 mass levels
    assert len([e for e in test if e.test_kind == 'single:mass']) == 2
    assert len([e for e in test if e.test_kind == 'single:color']) == 2
    (allh,) = [e for e in test if e.test_kind == 'all_heldout']
    assert allh.levels == {'mass': 'normal', 'color': 'night'}


def test_reference_must_be_a_train_level():
    raw = copy.deepcopy(MINI_SUITE)
    raw['factors']['mass']['reference'] = 'normal'  # held out
    with pytest.raises(ValueError, match='reference'):
        S.build(S.spec_from_dict(raw))


@pytest.mark.parametrize('mutate, match', [
    (lambda r: r['factors']['color']['levels']['warm'].update({'sim.block_mass': 2.0}), 'both'),
    (lambda r: r['factors']['mass']['levels'].pop('normal') and r['factors']['mass']['levels'].pop('heavy'), 'at least 2'),
    (lambda r: r['factors'].update({'bad name': {'levels': {'a': {}, 'b': {}}}}), 'letters'),
    (lambda r: r.update({'colour': 1}), 'unknown suite keys'),
    (lambda r: r['factors']['mass'].update({'weight': 1}), 'unknown keys'),
    (lambda r: r.update({'test_context': 'some'}), 'test_context'),
])
def test_spec_validation(mutate, match):
    raw = copy.deepcopy(MINI_SUITE)
    mutate(raw)
    with pytest.raises(ValueError, match=match):
        S.build(S.spec_from_dict(raw))


def test_hash_is_content_based():
    a, b = S.build(spec()), S.build(spec())
    assert [e.env_hash for e in a.envs] == [e.env_hash for e in b.envs]
    raw = copy.deepcopy(MINI_SUITE)
    raw['factors']['mass']['levels']['heavy'] = {'sim.block_mass': 4.0}
    c = S.build(S.spec_from_dict(raw))
    changed = {e.name for e, f in zip(a.envs, c.envs) if e.env_hash != f.env_hash}
    assert changed == {e.name for e in a.envs if e.levels['mass'] == 'heavy'}


def test_record_config_roundtrip_and_envs_reset():
    built = S.build(spec())
    raw = S.record_config(built)
    cfg = record.record_config_from_dict(raw)
    assert len(cfg.sandbox.conditions) == len(built.envs)
    env = gym.make('pts/PushT-v0')
    for cond, e in zip(cfg.sandbox.conditions, built.envs):
        assert cond.name == e.name
        assert cond.rounds == (3 if e.split == 'train' else 4)
        assert cond.seed == (0 if e.split == 'train' else 1000)
        assert cond.info['env_id'] == e.env_id and cond.info['split'] == e.split
        env.reset(seed=0, options=sandbox.reset_options(env, cond))
        flat = variations.flatten(env)
        assert flat['sim.block_mass'] == e.variation_values['sim.block_mass']
        assert flat['sim.agent_max_force'] == 10000.0


def test_write_and_load_roundtrip(tmp_path):
    built = S.build(spec())
    S.write(built, tmp_path)
    back = S.load_suite(tmp_path / 'suite.yaml')
    assert [(e.env_id, e.name, e.split, e.env_hash) for e in back.envs] == \
           [(e.env_id, e.name, e.split, e.env_hash) for e in built.envs]
    rec = yaml.safe_load((tmp_path / 'record.yaml').read_text())
    assert rec['output']['session_name'] == 'mini'
    assert sandbox.config_from_dict(rec).conditions[0].info['suite'] == 'mini'


def test_cli(tmp_path, capsys):
    S.run(['build', str(V0), '--out', str(tmp_path), '--holdout', 'mass=high'])
    out = capsys.readouterr().out
    assert '16 train, 4 test' in out and 'held out: heavy' in out
    assert (tmp_path / 'suite.yaml').exists() and (tmp_path / 'record.yaml').exists()
    S.run(['show', str(V0), '--test-context', 'all'])
    assert '16 train, 32 test' in capsys.readouterr().out


def test_record_config_partial_overrides():
    raw = copy.deepcopy(MINI_SUITE)
    raw['record'] = {'rounds': {'train': 7}}
    rec = S.record_config(S.build(S.spec_from_dict(raw)))
    rounds = {c['info']['split']: c['rounds'] for c in rec['conditions']}
    assert rounds == {'train': 7, 'test': 12}


def test_recorded_demos_carry_condition_info(mini):
    metas = [load_meta(p) for p in find_demos(mini['raw'])]
    assert len(metas) == 4 * 3 + 2 * 4
    by_name = mini['suite'].by_name()
    for m in metas:
        info = m.condition_info
        env = by_name[m.condition]
        assert info['env_id'] == env.env_id and info['env_hash'] == env.env_hash
        assert info['suite'] == 'mini' and info['split'] == env.split
        assert m.seed >= (1000 if env.split == 'test' else 0)
    assert np.mean([m.outcome == 'success' for m in metas if m.condition_info['split'] == 'test']) == 1.0
