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
#   - env_id: train envs 0..N-1 (the latent-embedding index), then test envs.
#     env_hash: a content hash of the resolved variation values.
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
    holdout: str | None = None  # level name or low|middle|high; None = suite default
    reference: str | None = None  # train level used while another factor is tested

    def heldout_level(self, default_policy: str) -> str:
        choice = self.holdout or default_policy
        names = list(self.levels)
        if choice in self.levels:
            return choice
        if choice not in POLICIES:
            raise ValueError(f'factor {self.name}: unknown holdout {choice!r} '
                             f'(a level name or one of {POLICIES})')
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
    record: dict = field(default_factory=dict)  # session settings for record.yaml


@dataclass
class Env:
    env_id: int
    name: str
    split: str  # train | test
    levels: dict  # factor -> level
    heldout_factors: list
    variation_values: dict
    env_hash: str

    @property
    def test_kind(self) -> str:
        if self.split == 'train':
            return 'train'
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
    if spec.test_context not in ('reference', 'all'):
        raise ValueError("test_context must be 'reference' or 'all'")
    if not NAME.match(spec.name):
        raise ValueError(f'suite name {spec.name!r}: use letters and digits only')
    return spec


def load_spec(path) -> SuiteSpec:
    with open(path) as f:
        return spec_from_dict(yaml.safe_load(f) or {})


def apply_overrides(spec: SuiteSpec, holdouts=(), test_context=None) -> SuiteSpec:
    """holdouts: ['mass=high', 'friction=middle', 'middle' (suite default)]."""
    by_name = {f.name: f for f in spec.factors}
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

    combos = []
    for train in itertools.product(*(f.train_levels(p) for f in factors)):
        combos.append(('train', dict(zip((f.name for f in factors), train)), []))
    for f in factors:
        others = [g for g in factors if g is not f]
        contexts = ([{g.name: ref[g.name] for g in others}] if spec.test_context == 'reference'
                    else [dict(zip((g.name for g in others), c))
                          for c in itertools.product(*(g.train_levels(p) for g in others))])
        for ctx in contexts:
            combos.append(('test', {**ctx, f.name: held[f.name]}, [f.name]))
    if spec.include_all_heldout and len(factors) > 1:
        combos.append(('test', dict(held), [f.name for f in factors]))

    envs = []
    for i, (split, levels, heldout) in enumerate(combos):
        levels = {f.name: levels[f.name] for f in factors}  # factor order
        values = resolve(spec, levels)
        envs.append(Env(i, env_name(spec, levels), split, levels, heldout, values, env_hash(values)))
    names = [e.name for e in envs]
    if len(set(names)) != len(names):
        raise ValueError('duplicate environment names (a factor has repeated levels?)')
    return Suite(spec, envs)


# --- outputs -----------------------------------------------------------------

RECORD_DEFAULTS = {
    'rounds': {'train': 10, 'test': 12},
    'seed': {'train': 0, 'test': 1000},  # test envs get their own start layouts
}


def record_config(suite: Suite) -> dict:
    """A pts-record/pts-sandbox config with one condition per environment."""
    rec = dict(suite.spec.record)
    rounds = {**RECORD_DEFAULTS['rounds'], **(rec.pop('rounds', None) or {})}
    seeds = {**RECORD_DEFAULTS['seed'], **(rec.pop('seed', None) or {})}
    conditions = []
    for e in suite.envs:
        conditions.append({
            'name': e.name,
            'variation_values': e.variation_values,
            'rounds': int(rounds[e.split]),
            'seed': int(seeds[e.split]),
            'info': {'suite': suite.spec.name, 'env_id': e.env_id, 'env_name': e.name,
                     'split': e.split, 'test_kind': e.test_kind, 'levels': e.levels,
                     'env_hash': e.env_hash},
        })
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
    lines = [f'suite {suite.spec.name}: {len(suite.split("train"))} train, '
             f'{len(suite.split("test"))} test environments']
    for f in suite.spec.factors:
        p = suite.spec.holdout_policy
        lines.append(f'  {f.name:<10} levels {list(f.levels)}  held out: {f.heldout_level(p)}'
                     f'  reference: {f.reference_level(p)}')
    for e in suite.envs:
        lines.append(f'  {e.env_id:3d} {e.split:<5} {e.test_kind:<16} {e.name}')
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
        if cmd == 'build':
            p.add_argument('--out', required=True, help='directory for suite.yaml + record.yaml')
    args = parser.parse_args(argv)
    spec = apply_overrides(load_spec(args.spec), args.holdout, args.test_context)
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
