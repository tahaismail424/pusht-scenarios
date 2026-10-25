# `pts-data`: inspect and curate demos (see io/collection.py).
#
#   uv run pts-data ls data/raw data/aug/v0            # per-condition summary
#   uv run pts-data ls data/raw -v --condition heavy   # one line per demo
#   uv run pts-data reject  <demo_id or prefix> ... --root data/raw --reason "sloppy"
#   uv run pts-data unreject <demo_id> ... --root data/raw
#   uv run pts-data trim <demo_id> --end 140 --root data/raw   # new demo; parent rejected
#   uv run pts-data merge data/raw/2026* --out data/curated/v0
#   uv run pts-data verify data/raw/<session>          # replay; exact?
import argparse

from pusht_scenarios.io import collection as C


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog='pts-data', description='Inspect and curate PushT demos.')
    sub = parser.add_subparsers(dest='command', required=True)

    ls = sub.add_parser('ls', help='summarize demos')
    ls.add_argument('roots', nargs='+')
    ls.add_argument('--condition', action='append')
    ls.add_argument('--all', action='store_true', help='include rejected')
    ls.add_argument('--recorded', action='store_true', help='recorded demos only (no augmented)')
    ls.add_argument('-v', '--verbose', action='store_true')

    for name in ('reject', 'unreject'):
        p = sub.add_parser(name, help=f'{name} demos (files are kept)')
        p.add_argument('demo_ids', nargs='+', help='demo ids or unique prefixes')
        p.add_argument('--root', action='append', required=True, help='where to look up the ids')
        if name == 'reject':
            p.add_argument('--reason', default='')

    tr = sub.add_parser('trim', help='write a copy of a demo cut at --end steps')
    tr.add_argument('demo_id')
    tr.add_argument('--end', type=int, required=True)
    tr.add_argument('--keep-parent', action='store_true', help="don't reject the original")
    tr.add_argument('--root', action='append', required=True)

    mg = sub.add_parser('merge', help='copy kept demos into a new collection')
    mg.add_argument('sources', nargs='+')
    mg.add_argument('--out', required=True)
    mg.add_argument('--include-rejected', action='store_true')

    vf = sub.add_parser('verify', help='replay demos and report the state error')
    vf.add_argument('roots', nargs='+')
    return parser


def run(argv=None):
    args = build_parser().parse_args(argv)
    if args.command == 'ls':
        index = C.DemoIndex.scan(*args.roots).select(
            conditions=args.condition, include_rejected=args.all,
            derived=False if args.recorded else None)
        print(index.summary())
        if args.verbose:
            for r in index:
                tag = f'  REJECTED ({r.reject_reason})' if r.rejected else ''
                print(f'{r.meta.demo_id[:8]}  {r.meta.condition:<40} seed {r.meta.seed:<5} '
                      f'{r.meta.outcome:<8} {r.meta.steps:4d} steps  {r.base:<16} '
                      f'noise {r.noise_std:g}  {r.path.name}{tag}')
        return index
    if args.command in ('reject', 'unreject'):
        index = C.DemoIndex.scan(*args.root, inherit_rejects=False)
        if args.command == 'reject':
            hits = C.reject(index, args.demo_ids, args.reason)
            print(f'rejected {len(hits)}: ' + ', '.join(r.meta.demo_id[:8] for r in hits))
            return hits
        still = C.unreject(index, args.demo_ids)
        if still:
            print('still rejected by a recording undo: ' + ', '.join(d[:8] for d in still))
        return still
    if args.command == 'trim':
        index = C.DemoIndex.scan(*args.root, inherit_rejects=False)
        path = C.trim_file(index.find(args.demo_id), args.end, keep_parent=args.keep_parent)
        print(f'wrote {path}')
        return path
    if args.command == 'merge':
        dest = C.merge(args.sources, args.out, include_rejected=args.include_rejected)
        print(f'merged into {dest}')
        return dest
    if args.command == 'verify':
        results = C.verify(*args.roots)
        bad = [(p, e) for p, e in results if e != 0.0]
        for p, e in bad:
            print(f'NOT EXACT  {p}  max error {e:.3e}')
        print(f'{len(results) - len(bad)}/{len(results)} demos replay exactly')
        return results
    raise AssertionError(args.command)


def main(argv=None):
    """Console entry point (returns None so the exit status is 0)."""
    run(argv)


if __name__ == '__main__':
    main()
