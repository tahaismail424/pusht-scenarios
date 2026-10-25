# Watch recorded (or augmented) demos by re-simulating them from file.
#
#   uv run pts-replay data/raw                 # every demo under a directory
#   uv run pts-replay data/raw/<session> --speed 2
#   uv run pts-replay path/to/0003_heavy_seed2_ab12cd34.npz
#
# Keys: Space pause, Right/N next demo, Left/P previous, R restart, +/- speed,
# X reject/unreject this demo (writes <collection>/curation.yaml), Q/Esc quit. The status line shows the demo, its outcome, and the max
# difference between the stored and re-simulated states (should be 0).
import argparse
from dataclasses import dataclass

import numpy as np
import pygame

from pusht_scenarios.gen import replay
from pusht_scenarios.io.demo import current_state, find_demos, load_demo
from pusht_scenarios.io import collection
from pusht_scenarios.teleop.sandbox import draw_hud

NEXT, PREV, RESTART, QUIT, REJECT = 'next', 'prev', 'restart', 'quit', 'reject'


@dataclass
class ViewResult:
    path: str
    steps_shown: int
    max_error: float  # over the steps shown
    finished: bool


def derived_label(derived: dict) -> str:
    """e.g. 'se2_0 noise 0.01' for an augmented demo."""
    parts = [derived.get('base', 'derived')]
    for op in derived.get('chain', []):
        if op.get('op') == 'noise' and op.get('sigma'):
            parts.append(f"noise {op['sigma']:g}")
    return ' '.join(parts)


def play_demo(env, demo, label='', fps=10.0, get_events=pygame.event.get, clock=None,
              end_pause_s=0.6):
    """Re-simulate one demo on screen. Returns (command, steps, max_error);
    command is NEXT when it finishes (or is skipped), or PREV/RESTART/QUIT."""
    clock = clock or pygame.time.Clock()
    paused, err, t = False, 0.0, 0
    replay.reset_to(env, demo)
    done = len(demo) == 0
    end_frames = int(end_pause_s * fps) if fps else 0
    while True:
        for event in get_events():
            if event.type == pygame.QUIT:
                return QUIT, t, err
            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                    return QUIT, t, err
                if event.key in (pygame.K_RIGHT, pygame.K_n):
                    return NEXT, t, err
                if event.key in (pygame.K_LEFT, pygame.K_p):
                    return PREV, t, err
                if event.key == pygame.K_r:
                    return RESTART, t, err
                if event.key == pygame.K_x:
                    return REJECT, t, err
                if event.key == pygame.K_SPACE:
                    paused = not paused
                if event.key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
                    fps = min(fps * 2, 160) if fps else fps
                if event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                    fps = max(fps / 2, 1.25) if fps else fps

        if not paused and not done:
            env.step(demo.action[t])
            t += 1
            err = max(err, float(np.max(np.abs(current_state(env) - demo.state[t]))))
            done = t == len(demo)

        env.render()
        state = 'PAUSED' if paused else ('done' if done else f'{fps:g} Hz')
        draw_hud(env, f'{label}  step {t}/{len(demo)}  err {err:.1e}  {state}')
        if done:
            if end_frames <= 0:
                return NEXT, t, err
            end_frames -= 1
        if fps:
            clock.tick(fps)


def view(paths, speed=1.0, get_events=pygame.event.get, env_factory=None, verbose=True):
    """Play demos in order; returns one ViewResult per demo shown."""
    files = find_demos(*paths)
    if not files:
        raise SystemExit(f'no demos found in {paths}')
    results, i, env, env_id = [], 0, None, None
    rejected_cache = {}
    clock = pygame.time.Clock()
    try:
        while 0 <= i < len(files):
            demo = load_demo(files[i])
            if env is None or demo.meta.env_id != env_id:
                if env is not None:
                    env.close()
                env = (env_factory or (lambda d: replay.make_env(d, render_mode='human')))(demo)
                env_id = demo.meta.env_id
            coll = collection.collection_dir_of(files[i])
            if coll not in rejected_cache:
                rejected_cache[coll] = set(collection.collection_rejects(coll))
            m = demo.meta
            is_rejected = m.demo_id in rejected_cache[coll]
            tag = ' REJECTED' if is_rejected else ''
            derived = f' [{derived_label(m.derived_from)}]' if m.derived_from else ''
            label = f'{i + 1}/{len(files)} {m.condition} seed {m.seed} {m.outcome}{tag}{derived}'
            command, shown, err = play_demo(env, demo, label, fps=10.0 * speed,
                                            get_events=get_events, clock=clock)
            results.append(ViewResult(str(files[i]), shown, err, shown == len(demo)))
            if verbose:
                print(f'{files[i].name}: {shown}/{len(demo)} steps, max error {err:.2e}')
            if command == QUIT:
                break
            if command == REJECT:  # toggle, then replay the same demo with the new label
                if is_rejected:
                    collection.unreject_in(coll, [m.demo_id])
                else:
                    collection.reject_in(coll, [m.demo_id], 'rejected in pts-replay')
                rejected_cache[coll] = set(collection.collection_rejects(coll))
                if verbose:
                    print(f'{"unrejected" if is_rejected else "rejected"} {m.demo_id[:8]}')
                continue
            i += {NEXT: 1, PREV: -1, RESTART: 0}[command]
            i = max(i, 0)
    finally:
        if env is not None:
            env.close()
    return results


def run(argv=None):
    parser = argparse.ArgumentParser(description='Replay recorded/augmented PushT demos.')
    parser.add_argument('paths', nargs='+', help='demo .npz files and/or directories')
    parser.add_argument('--speed', type=float, default=1.0, help='playback speed (1 = 10 Hz)')
    args = parser.parse_args(argv)
    pygame.init()
    try:
        return view(args.paths, speed=args.speed)
    finally:
        pygame.quit()


def main(argv=None):
    """Console entry point (returns None so the exit status is 0)."""
    run(argv)


if __name__ == '__main__':
    main()
