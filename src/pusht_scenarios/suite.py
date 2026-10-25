# Environment suites: factors x levels -> named environments with stable ids
# and a train/test split, for adaptation experiments (T2-z style: one latent
# per training environment, adapted on held-out ones).
#
#   uv run pts-suite show  configs/suites/v0.yaml
#   uv run pts-suite build configs/suites/v0.yaml --out configs/generated/v0 \
#       [--holdout mass=high --holdout friction=middle] [--test-context all]
#   uv run pts-record -c configs/generated/v0/record.yaml
#
# Design:
#   - every factor has levels; ONE level per factor is held out (the knob:
#     `holdout`, per factor or suite-wide). Ordered factors accept low|middle|
#     high (list order is low -> high); categorical factors need a level name.
#   - train envs: the full factorial over the remaining (train) levels.
#   - test envs: single-factor shifts. One factor at its held-out level, the
#     others at their `reference` train level (test_context: reference), or at
#     every train combination (test_context: all). Optionally one env with every
#     factor held out (include_all_heldout).
#   - holdout_combos: joint holdouts, e.g. [{mass: heavy, friction: stop}]. Test
#     envs have ALL the combo's levels at once (others at reference, or every
#     train combination with test_context: all); train envs are everything
#     else, so each level is seen in training, just never together.
#     CLI: --holdout-combo mass=heavy,friction=stop (repeatable); like --only,
#     it makes every factor hold nothing out on its own.
#   - record_all: record EVERY level combination (the full factorial), so the
#     train/test split can be changed after recording: rebuild the suite with
#     another --holdout and re-export. Combinations that are neither train nor
#     test under the current holdout are split 'unused' (recorded, not exported).
#     Env names are built from level names, so recordings map onto any rebuild.
#   - env_id: train envs 0..N-1 (the latent-embedding index), then test envs,
#     then unused ones. env_hash: a content hash of the resolved variation
#     values; export refuses recordings whose hash no longer matches.
import argparse
import hashlib
import itertools
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml

NAME = re.compile(r'^[A-Za-z0-9]+$')
POLICIES = ('low', 'middle', 'high')


@dataclass
class Factor:
    name: str
    levels: dict  # level name -> variation_values (ordered: low -> high)
    ordered: bool = True
    holdout: str | None = None  # level name, low|middle|high, or 'none'; None = suite default
    reference: str | None = None  # train level used while another factor is tested

    def heldout_level(self, default_policy: str) -> str | None:
        """The held-out level, or None if this factor holds nothing out ('none')."""
        choice = self.holdout or default_policy
        names = list(self.levels)
        if choice in self.levels:
            return choice
        if choice == 'none':
            return None
        if choice not in POLICIES:
            raise ValueError(f'factor {self.name}: unknown holdout {choice!r} '
                             f"(a level name, one of {POLICIES}, or 'none')")
        if not self.ordered:
            raise ValueError(f'factor {self.name} is categorical: holdout must name a level '
                             f'({names}), not {choice!r}')
        return {'low': names[0], 'middle': names[len(names) // 2], 'high': names[-1]}[choice]

    def train_levels(self, default_policy: str) -> list[str]:
        held = self.heldout_level(default_policy)
        return [n for n in self.levels if n != held]

    def reference_level(self, default_policy: str) -> str:
        train = self.train_levels(default_policy)
        if self.reference is None:
            return train[0]
        if self.reference not in train:
            raise ValueError(f'factor {self.name}: reference {self.reference!r} must be a '
                             f'train level ({train})')
        return self.reference


@dataclass
class SuiteSpec:
    name: str
    factors: list[Factor]
    base: dict = field(default_factory=dict)  # variation values shared by every env
    holdout_policy: str = 'middle'
    test_context: str = 'reference'  # reference | all
    include_all_heldout: bool = False
    record_all: bool = False  # record every level combination (splits chosen at export)
    holdout_combos: list = field(default_factory=list)  # [{factor: level, ...}] joint holdouts
    record: dict = field(default_factory=dict)  # session settings for record.yaml


@dataclass
class Env:
    env_id: int
    name: str
    split: str  # train | test | unused (record_all: recorded, not part of this split)
    levels: dict  # factor -> level
    heldout_factors: list
    variation_values: dict
    env_hash: str
    kind: str = ''  # set for joint-holdout tests: 'combo:mass+friction'

    @property
    def test_kind(self) -> str:
        if self.split in ('train', 'unused'):
            return self.split
        if self.kind:
            return self.kind
        return 'single:' + self.heldout_factors[0] if len(self.heldout_factors) == 1 else 'all_heldout'


@dataclass
class Suite:
    spec: SuiteSpec
    envs: list[Env]

    def by_name(self) -> dict[str, Env]:
        return {e.name: e for e in self.envs}

    def split(self, split: str) -> list[Env]:
        return [e for e in self.envs if e.split == split]


# --- parsing -----------------------------------------------------------------


def spec_from_dict(raw: dict) -> SuiteSpec:
    known = set(SuiteSpec.__dataclass_fields__)
    unknown = set(raw) - known
    if unknown:
        raise ValueError(f'unknown suite keys: {sorted(unknown)}')
    raw = dict(raw)
    factors_raw = raw.pop('factors', None) or {}
    if not factors_raw:
        raise ValueError('a suite needs at least one factor')
    factors = []
    for fname, f in factors_raw.items():
        f = dict(f)
        bad = set(f) - set(Factor.__dataclass_fields__) - {'name'}
        if bad:
            raise ValueError(f'factor {fname}: unknown keys {sorted(bad)}')
        levels = {str(k): dict(v or {}) for k, v in (f.pop('levels', None) or {}).items()}
        if len(levels) < 2:
            raise ValueError(f'factor {fname}: needs at least 2 levels')
        for n in (fname, *levels):
            if not NAME.match(n):
                raise ValueError(f'name {n!r}: use letters and digits only')
        factors.append(Factor(name=fname, levels=levels, **f))
    spec = SuiteSpec(factors=factors, **raw)
    spec.holdout_combos = [validate_combo(spec, c) for c in spec.holdout_combos]
    if spec.test_context not in ('reference', 'all'):
        raise ValueError("test_context must be 'reference' or 'all'")
    if not NAME.match(spec.name):
        raise ValueError(f'suite name {spec.name!r}: use letters and digits only')
    return spec


def load_spec(path) -> SuiteSpec:
    with open(path) as f:
        return spec_from_dict(yaml.safe_load(f) or {})


def validate_combo(spec: SuiteSpec, combo: dict) -> dict:
    by_name = {f.name: f for f in spec.factors}
    if not combo:
        raise ValueError('a holdout combo needs at least one factor=level')
    for fname, level in combo.items():
        if fname not in by_name:
            raise ValueError(f'holdout combo: unknown factor {fname!r} (factors: {list(by_name)})')
        if level not in by_name[fname].levels:
            raise ValueError(f'holdout combo: {fname} has no level {level!r} '
                             f'({list(by_name[fname].levels)})')
    return {f.name: combo[f.name] for f in spec.factors if f.name in combo}  # factor order


def parse_combo(text: str) -> dict:
    """'mass=heavy,friction=stop' -> {'mass': 'heavy', 'friction': 'stop'}"""
    combo = {}
    for part in filter(None, (p.strip() for p in text.split(','))):
        if '=' not in part:
            raise ValueError(f'--holdout-combo: use factor=level,factor=level (got {text!r})')
        k, v = part.split('=', 1)
        combo[k.strip()] = v.strip()
    return combo


def apply_overrides(spec: SuiteSpec, holdouts=(), test_context=None, only=None,
                    combos=()) -> SuiteSpec:
    """holdouts: ['mass=high', 'friction=middle', 'color=none', 'middle' (suite default)].
    only: 'mass=heavy' holds out just that level; every other factor holds
    nothing out (all its levels are train).
    combos: ['mass=heavy,friction=stop', ...] joint holdouts; every factor then
    holds nothing out on its own (as with only)."""
    by_name = {f.name: f for f in spec.factors}
    if combos:
        spec.holdout_combos = [validate_combo(spec, parse_combo(c)) for c in combos]
        for f in spec.factors:
            f.holdout = 'none'
    if only:
        fname, choice = only.split('=', 1) if '=' in only else (only, None)
        if fname not in by_name or not choice:
            raise ValueError(f'--only: use factor=level (factors: {list(by_name)})')
        for f in spec.factors:
            f.holdout = choice if f.name == fname else 'none'
    for item in holdouts:
        if '=' in item:
            fname, choice = item.split('=', 1)
            if fname not in by_name:
                raise ValueError(f'--holdout: unknown factor {fname!r}')
            by_name[fname].holdout = choice
        else:  # suite-wide policy: replaces ordered factors' holdouts only
            if item not in POLICIES:
                raise ValueError(f'--holdout {item!r}: use factor=level or one of {POLICIES}')
            spec.holdout_policy = item
            for f in spec.factors:
                if f.ordered:
                    f.holdout = None
    if test_context:
        spec.test_context = test_context
    return spec


# --- building ----------------------------------------------------------------


def env_hash(values: dict) -> str:
    return hashlib.sha1(json.dumps(values, sort_keys=True).encode()).hexdigest()[:10]


def resolve(spec: SuiteSpec, levels: dict) -> dict:
    """Merge base + each factor's level values; two factors may not set the same key."""
    values = dict(spec.base)
    owner = {k: 'base' for k in values}
    for f in spec.factors:
        for key, value in f.levels[levels[f.name]].items():
            if key in owner and owner[key] != 'base':
                raise ValueError(f'{key!r} is set by both {owner[key]} and {f.name}')
            values[key] = value
            owner[key] = f.name
    return values


def env_name(spec: SuiteSpec, levels: dict) -> str:
    return '.'.join(f'{f.name}-{levels[f.name]}' for f in spec.factors)


def build(spec: SuiteSpec) -> Suite:
    p = spec.holdout_policy
    factors = spec.factors
    held = {f.name: f.heldout_level(p) for f in factors}
    ref = {f.name: f.reference_level(p) for f in factors}

    joint = spec.holdout_combos
    if all(h is None for h in held.values()) and not joint:
        raise ValueError('no factor holds out a level (every holdout is none) and no holdout combo')

    def in_combo(levels):
        return any(all(levels[f] == lv for f, lv in c.items()) for c in joint)

    combos = []
    for train in itertools.product(*(f.train_levels(p) for f in factors)):
        levels = dict(zip((f.name for f in factors), train))
        if not in_combo(levels):
            combos.append(('train', levels, []))
    for f in factors:
        if held[f.name] is None:
            continue
        others = [g for g in factors if g is not f]
        contexts = ([{g.name: ref[g.name] for g in others}] if spec.test_context == 'reference'
                    else [dict(zip((g.name for g in others), c))
                          for c in itertools.product(*(g.train_levels(p) for g in others))])
        for ctx in contexts:
            combos.append(('test', {**ctx, f.name: held[f.name]}, [f.name]))
    for c in joint:
        others = [g for g in factors if g.name not in c]
        contexts = ([{g.name: ref[g.name] for g in others}] if spec.test_context == 'reference'
                    else [dict(zip((g.name for g in others), ctx))
                          for ctx in itertools.product(*(g.train_levels(p) for g in others))])
        for ctx in contexts:
            combos.append(('test', {**ctx, **c}, list(c), 'combo:' + '+'.join(c)))
    tested = [f for f in factors if held[f.name] is not None]
    if spec.include_all_heldout and len(tested) > 1:
        combos.append(('test', {f.name: held[f.name] if held[f.name] is not None else ref[f.name]
                                for f in factors}, [f.name for f in tested]))
    if spec.record_all:
        seen = {tuple(c[1][f.name] for f in factors) for c in combos}
        for combo in itertools.product(*(list(f.levels) for f in factors)):
            if combo not in seen:
                levels = dict(zip((f.name for f in factors), combo))
                combos.append(('unused', levels, [f.name for f in factors if levels[f.name] == held[f.name]]))

    envs, seen_names = [], set()
    for split, levels, heldout, *kind in combos:
        levels = {f.name: levels[f.name] for f in factors}  # factor order
        name = env_name(spec, levels)
        if name in seen_names:  # e.g. a combo env that is also a single-factor test
            continue
        seen_names.add(name)
        values = resolve(spec, levels)
        envs.append(Env(len(envs), name, split, levels, heldout, values, env_hash(values),
                        kind[0] if kind else ''))
    names = [e.name for e in envs]
    if len(set(names)) != len(names):
        raise ValueError('duplicate environment names (a factor has repeated levels?)')
    return Suite(spec, envs)


# --- outputs -----------------------------------------------------------------

RECORD_DEFAULTS = {
    'rounds': {'train': 10, 'test': 12, 'all': 6},
    # test envs get their own start layouts; with record_all every env shares
    # seeds (any env may end up train or test)
    'seed': {'train': 0, 'test': 1000, 'all': 0},
}


def layout_seed(env: Env) -> int:
    """A per-env seed base from the env's content hash: stable across holdout
    choices and factor reordering, distinct per env, room for 100 rounds."""
    return (int(env.env_hash, 16) % 100_000) * 100


def record_config(suite: Suite) -> dict:
    """A pts-record/pts-sandbox config with one condition per environment to
    record: train + test envs, or every combination with record_all.

    record.layouts: 'shared' (every env of a split plays the same start
    layouts: seed + i) or 'per_env' (each env gets its own: layout_seed + i)."""
    rec = dict(suite.spec.record)
    rounds = {**RECORD_DEFAULTS['rounds'], **(rec.pop('rounds', None) or {})}
    seeds = {**RECORD_DEFAULTS['seed'], **(rec.pop('seed', None) or {})}
    layouts = rec.pop('layouts', 'shared')
    if layouts not in ('shared', 'per_env'):
        raise ValueError("record.layouts must be 'shared' or 'per_env'")
    conditions = []
    for e in suite.envs:
        if suite.spec.record_all:
            # the split isn't known yet: it's chosen when the suite is (re)built
            # for export, so only record what identifies the env
            n, seed = rounds['all'], seeds['all']
            info = {'suite': suite.spec.name, 'env_name': e.name, 'levels': e.levels,
                    'env_hash': e.env_hash, 'record_all': True}
        elif e.split == 'unused':
            continue
        else:
            n, seed = rounds[e.split], seeds[e.split]
            info = {'suite': suite.spec.name, 'env_id': e.env_id, 'env_name': e.name,
                    'split': e.split, 'test_kind': e.test_kind, 'levels': e.levels,
                    'env_hash': e.env_hash}
        if layouts == 'per_env':
            seed = layout_seed(e) + seed
        conditions.append({'name': e.name, 'variation_values': e.variation_values,
                           'rounds': int(n), 'seed': int(seed), 'info': info})
    rec.setdefault('output', {})
    rec['output'] = {'session_name': suite.spec.name, **rec['output']}
    return {**rec, 'conditions': conditions}


def suite_dict(suite: Suite) -> dict:
    spec = asdict(suite.spec)
    envs = [{**asdict(e), 'test_kind': e.test_kind} for e in suite.envs]
    return {'name': suite.spec.name, 'spec': spec, 'envs': envs}


def load_suite(path) -> Suite:
    """Read a built suite.yaml back."""
    with open(path) as f:
        d = yaml.safe_load(f)
    spec = d['spec']
    factors = [Factor(**f) for f in spec.pop('factors')]
    s = SuiteSpec(factors=factors, **spec)
    envs = [Env(**{k: v for k, v in e.items() if k != 'test_kind'}) for e in d['envs']]
    return Suite(s, envs)


def write(suite: Suite, out_dir) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    header = '# generated by pts-suite; edit the suite spec, not this file\n'
    with open(out / 'suite.yaml', 'w') as f:
        f.write(header)
        yaml.safe_dump(suite_dict(suite), f, sort_keys=False)
    with open(out / 'record.yaml', 'w') as f:
        f.write(header + '#   uv run pts-record -c ' + str(out / 'record.yaml') + '\n')
        yaml.safe_dump(record_config(suite), f, sort_keys=False)
    return out


def table(suite: Suite) -> str:
    unused = suite.split('unused')
    lines = [f'suite {suite.spec.name}: {len(suite.split("train"))} train, '
             f'{len(suite.split("test"))} test environments'
             + (f', {len(unused)} recorded but unused (record_all)' if unused else '')]
    for f in suite.spec.factors:
        p = suite.spec.holdout_policy
        lines.append(f'  {f.name:<10} levels {list(f.levels)}  held out: {f.heldout_level(p) or "-"}'
                     f'  reference: {f.reference_level(p)}')
    for c in suite.spec.holdout_combos:
        lines.append(f'  joint holdout: {", ".join(f"{k}={v}" for k, v in c.items())}')
    for e in suite.envs:
        lines.append(f'  {e.env_id:3d} {e.split:<6} {e.test_kind:<22} {e.name}')
    return '\n'.join(lines)


def run(argv=None):
    parser = argparse.ArgumentParser(prog='pts-suite', description='Build environment suites.')
    sub = parser.add_subparsers(dest='command', required=True)
    for cmd in ('show', 'build'):
        p = sub.add_parser(cmd)
        p.add_argument('spec')
        p.add_argument('--holdout', action='append', default=[],
                       help="factor=level|low|middle|high, or a suite-wide policy; repeatable")
        p.add_argument('--test-context', choices=('reference', 'all'))
        p.add_argument('--only', help='factor=level: hold out just this level; every other '
                                      'factor trains on all its levels')
        p.add_argument('--holdout-combo', action='append', default=[],
                       help='factor=level,factor=level: hold out this joint combination '
                            '(each level alone stays in training); repeatable')
        if cmd == 'build':
            p.add_argument('--out', required=True, help='directory for suite.yaml + record.yaml')
    args = parser.parse_args(argv)
    spec = apply_overrides(load_spec(args.spec), args.holdout, args.test_context, args.only,
                           args.holdout_combo)
    suite = build(spec)
    print(table(suite))
    if args.command == 'build':
        out = write(suite, args.out)
        print(f'wrote {out / "suite.yaml"} and {out / "record.yaml"}')
    return suite


def main(argv=None):
    """Console entry point (returns None so the exit status is 0)."""
    run(argv)


if __name__ == '__main__':
    main()
