# Augmentation recipes: recorded demos -> a directory of derived demos.
#
#   uv run pts-augment -c configs/augment/full.yaml [--workers 8]
#   uv run pts-replay  data/aug/<name>
#
# For every parent demo:
#   bases  = original + fixed rotations + random SE(2) + mirrors   (augment.py)
#            (+ each palette applied to the bases named in palettes.apply_to)
#   copies = every base x every noise level (x copies per level)   (DINO-WM)
# and each copy is simulated from reset (simulate.py) and written as a Demo:
#
#   <output_root>/<name>/
#     recipe.yaml    the exact recipe used
#     summary.yaml   counts per condition: parents, copies, success/failure,
#                    skipped transforms and why
#     demos/<condition>_<parent8>_<copy>.npz
#
# Randomness is seeded per parent from (recipe seed, parent demo_id), so the
# output doesn't depend on worker count or parent order.
import argparse
import os
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path

import gymnasium as gym
import numpy as np
import yaml

import pusht_scenarios.env  # noqa: F401
from pusht_scenarios.gen import augment, replay
from pusht_scenarios.gen.simulate import InvalidCandidate, check_candidate, simulate
from pusht_scenarios.io.collection import is_recorded
from pusht_scenarios.io.demo import find_demos, load_demo, save_demo
from pusht_scenarios.io.session import rejected_ids, session_dir_of


@dataclass
class SE2Config:
    count: int = 0  # random rotation+shift bases per parent
    rotation_deg: tuple = (-180.0, 180.0)
    shift_px: tuple = (-60.0, 60.0)
    max_tries: int = 25  # resample invalid transforms up to this many times


@dataclass
class BasesConfig:
    original: bool = True
    rotations_deg: list = field(default_factory=list)  # exact rotations, e.g. [90, 180, 270]
    se2: SE2Config = field(default_factory=SE2Config)
    mirror_axes_deg: list = field(default_factory=list)  # world mirror lines, e.g. [0, 90]


@dataclass
class PalettesConfig:
    apply_to: str = 'original'  # 'original' (untransformed base only) or 'all' bases
    palettes: dict = field(default_factory=dict)  # name -> {'block.color': [r,g,b], ...}


@dataclass
class Recipe:
    name: str
    sources: list = field(default_factory=lambda: ['data/raw'])
    output_root: str = 'data/aug'
    seed: int = 0
    conditions: list | None = None  # condition names to include (None = all)
    suite_splits: list | None = None  # e.g. [train]: only demos of these suite splits
    # built suite.yaml: only its envs are augmented, and suite_splits follow it
    # (instead of the split recorded in each demo)
    suite: str | None = None
    parents_per_condition: int | None = None  # cap, for quick samples
    include_rejected: bool = False  # parents undone during recording
    include_failures: bool = True  # parents with outcome != success
    bases: BasesConfig = field(default_factory=BasesConfig)
    palettes: PalettesConfig = field(default_factory=PalettesConfig)
    noise: list = field(default_factory=lambda: [list(x) for x in augment.DINO_WM_NOISE_LEVELS])
    pos_tol: float = 20.0
    angle_tol_deg: float = 20.0

    def copies_per_base(self) -> int:
        return sum(int(n) for _, n in self.noise)


def _strict(cls, raw: dict, where: str):
    unknown = set(raw) - set(cls.__dataclass_fields__)
    if unknown:
        raise ValueError(f'unknown {where} keys: {sorted(unknown)}')
    return raw


def recipe_from_dict(raw: dict) -> Recipe:
    raw = dict(_strict(Recipe, raw, 'recipe'))
    bases = dict(_strict(BasesConfig, raw.pop('bases', None) or {}, 'bases'))
    se2 = SE2Config(**_strict(SE2Config, bases.pop('se2', None) or {}, 'bases.se2'))
    palettes = PalettesConfig(**_strict(PalettesConfig, raw.pop('palettes', None) or {}, 'palettes'))
    if palettes.apply_to not in ('original', 'all'):
        raise ValueError("palettes.apply_to must be 'original' or 'all'")
    noise = raw.pop('noise', None)
    if noise == 'dino_wm' or noise is None:
        noise = [list(x) for x in augment.DINO_WM_NOISE_LEVELS]
    noise = [[float(s), int(n)] for s, n in noise]
    if any(s < 0 or n < 0 for s, n in noise):
        raise ValueError('noise levels must be [sigma >= 0, copies >= 0]')
    return Recipe(**raw, bases=BasesConfig(**bases, se2=se2), palettes=palettes, noise=noise)


def load_recipe(path) -> Recipe:
    with open(path) as f:
        return recipe_from_dict(yaml.safe_load(f) or {})


# --- selecting parents -----------------------------------------------------


def select_parents(recipe: Recipe) -> list[Path]:
    suite_split = None
    if recipe.suite:
        from pusht_scenarios.suite import load_suite

        suite_split = {e.name: e.split for e in load_suite(recipe.suite).envs}
    rejected = {}
    by_condition = defaultdict(list)
    for path in find_demos(*recipe.sources):
        demo = load_demo(path)
        m = demo.meta
        if not is_recorded(m):  # never augment augmented demos (trims are fine)
            continue
        if recipe.conditions is not None and m.condition not in recipe.conditions:
            continue
        if suite_split is not None and m.condition not in suite_split:
            continue  # not one of the suite's envs (e.g. another session's conditions)
        if recipe.suite_splits is not None:
            split = suite_split.get(m.condition) if suite_split else m.condition_info.get('split')
            if split not in recipe.suite_splits:
                continue
        if not recipe.include_failures and m.outcome != 'success':
            continue
        session = session_dir_of(path)
        if session not in rejected:
            rejected[session] = rejected_ids(session) if session else set()
        if not recipe.include_rejected and m.demo_id in rejected[session]:
            continue
        by_condition[m.condition].append(path)
    out = []
    for condition in sorted(by_condition):
        paths = by_condition[condition]
        out.extend(paths[:recipe.parents_per_condition] if recipe.parents_per_condition else paths)
    return out


# --- per parent ------------------------------------------------------------


def parent_rng(recipe: Recipe, demo_id: str) -> np.random.Generator:
    return np.random.default_rng([recipe.seed, int(demo_id[:15], 16)])


def build_bases(env, demo, recipe: Recipe, rng, skipped: Counter):
    """(base_name, candidate) list for one parent."""
    root = augment.from_demo(demo)
    bases = [('original', root)] if recipe.bases.original else []

    def valid(c):
        return not check_candidate(env, c)

    for deg in recipe.bases.rotations_deg:
        c = augment.se2(root, np.radians(deg))
        if valid(c):
            bases.append((f'rot{deg:g}', c))
        else:
            skipped[f'rot{deg:g}:invalid'] += 1

    se2 = recipe.bases.se2
    for k in range(se2.count):
        for _ in range(se2.max_tries):
            theta = np.radians(rng.uniform(*se2.rotation_deg))
            shift = rng.uniform(*se2.shift_px, size=2)
            c = augment.se2(root, theta, shift)
            if valid(c):
                bases.append((f'se2_{k}', c))
                break
        else:
            skipped['se2:gave_up'] += 1

    if recipe.bases.mirror_axes_deg:
        replay.reset_to(env, demo)
        local_axes = augment.local_mirror_axes(env)
        for deg in recipe.bases.mirror_axes_deg:
            if not local_axes:
                skipped['mirror:no_axis'] += 1
                continue
            c = augment.mirror(root, deg, local_axes[0])
            if valid(c):
                bases.append((f'mirror{deg:g}', c))
            else:
                skipped[f'mirror{deg:g}:invalid'] += 1

    colored = []
    targets = bases if recipe.palettes.apply_to == 'all' else bases[:1] if recipe.bases.original else []
    for name, palette in recipe.palettes.palettes.items():
        for base_name, c in targets:
            colored.append((f'{base_name}+{name}', augment.recolor(c, palette, name)))
    return bases + colored


_ENV = None


def _env():
    global _ENV
    if _ENV is None:
        _ENV = gym.make('pts/PushT-v0')
    return _ENV


def augment_parent(job) -> dict:
    """Worker: all copies of one parent. Returns counts for the summary."""
    recipe, path, out_dir = job
    env = _env()
    demo = load_demo(path)
    rng = parent_rng(recipe, demo.meta.demo_id)
    skipped = Counter()
    outcomes = Counter()
    copy = 0
    for base_name, base in build_bases(env, demo, recipe, rng, skipped):
        for sigma, n in recipe.noise:
            for _ in range(n):
                cand = augment.noise(base, sigma, rng)
                try:
                    out = simulate(env, cand, copy_index=copy, recipe=recipe.name,
                                   pos_tol=recipe.pos_tol, angle_tol_deg=recipe.angle_tol_deg)
                except InvalidCandidate as e:
                    skipped[f'{base_name}:{e}'] += 1
                    continue
                out.meta.derived_from['base'] = base_name
                name = f'{demo.meta.condition}_{demo.meta.demo_id[:8]}_{copy:04d}.npz'
                save_demo(out, Path(out_dir) / name)
                outcomes[out.meta.outcome] += 1
                copy += 1
    return {'condition': demo.meta.condition, 'outcomes': dict(outcomes),
            'skipped': dict(skipped), 'copies': copy}


def run_recipe(recipe: Recipe, workers: int | None = None, verbose: bool = True) -> Path:
    out = Path(recipe.output_root) / recipe.name
    if out.exists():
        raise FileExistsError(f'output already exists: {out} (pick a new recipe name)')
    parents = select_parents(recipe)
    if not parents:
        raise ValueError(f'no parent demos found in {recipe.sources}')
    demos_dir = out / 'demos'
    demos_dir.mkdir(parents=True)
    with open(out / 'recipe.yaml', 'w') as f:
        yaml.safe_dump(asdict(recipe), f, sort_keys=False)

    jobs = [(recipe, p, demos_dir) for p in parents]
    workers = workers if workers is not None else (os.cpu_count() or 1)
    if verbose:
        print(f'{len(parents)} parents -> {out} ({workers} workers)')
    if workers <= 1:
        results = [augment_parent(j) for j in jobs]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(augment_parent, jobs))

    summary = defaultdict(lambda: {'parents': 0, 'copies': 0, 'outcomes': Counter(), 'skipped': Counter()})
    for r in results:
        s = summary[r['condition']]
        s['parents'] += 1
        s['copies'] += r['copies']
        s['outcomes'].update(r['outcomes'])
        s['skipped'].update(r['skipped'])
    plain = {c: {'parents': s['parents'], 'copies': s['copies'],
                 'outcomes': dict(s['outcomes']), 'skipped': dict(s['skipped'])}
             for c, s in sorted(summary.items())}
    total = {'parents': len(parents), 'copies': sum(s['copies'] for s in plain.values())}
    with open(out / 'summary.yaml', 'w') as f:
        yaml.safe_dump({'total': total, 'by_condition': plain}, f, sort_keys=False)
    if verbose:
        for c, s in plain.items():
            print(f'  {c:<14} parents {s["parents"]:3d}  copies {s["copies"]:6d}  {s["outcomes"]}')
        print(f'{total["copies"]} demos written to {demos_dir}')
    return out


def run(argv=None):
    parser = argparse.ArgumentParser(description='Augment recorded PushT demos.')
    parser.add_argument('-c', '--config', required=True, help='recipe YAML')
    parser.add_argument('--workers', type=int, help='processes (default: all cores)')
    parser.add_argument('--name', help='override the recipe name (output directory)')
    args = parser.parse_args(argv)
    recipe = load_recipe(args.config)
    if args.name:
        recipe.name = args.name
    return run_recipe(recipe, workers=args.workers)


def main(argv=None):
    """Console entry point (returns None so the exit status is 0)."""
    run(argv)


if __name__ == '__main__':
    main()
