# -*- coding: utf-8 -*-
"""
    Narrow-aisle static rendering and animation GIF (paper figures).
"""
import os, json
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.animation as animation

import l2_common as L
from common import bfs_path, closed_loop_assign, simulate_movement_trace, CAR_COLORS, SPOT_EDGE

W, H = L.FAMILY_B['W'], L.FAMILY_B['H']
N_CARS, M = L.FAMILY_B['n_cars'], L.FAMILY_B['M']
P = 0.4
N_TRAIN = 200


def build_demo(seed=20260925):
    rng = np.random.default_rng(seed)
    obstacles, gap_y = L.build_corridor_obstacles(W, H)
    scene = L.make_scene_corridor(W, H, obstacles, N_CARS, M, P, rng)
    train = L.gen_scenes_corridor(N_TRAIN, W, H, obstacles, N_CARS, M, P, rng)
    return scene, obstacles, train


def read_lambda_star():
    path = os.path.join(L.DATA_DIR, 'l2_a1.json')
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            a1 = json.load(f)
        for res in a1['results']:
            if res['family'] == 'narrow_corridor':
                return res['cells']['0.45']['best_lam']
    return 0.6


def draw(ax, scene, obstacles, car_positions, assign, lam, waiting=None, title_extra=''):
    ax.clear()
    ax.set_xlim(-0.5, W - 0.5); ax.set_ylim(H - 0.5, -0.5)
    ax.set_xticks(range(W)); ax.set_yticks(range(H)); ax.grid(alpha=0.2)
    for (x, y) in obstacles:
        ax.add_patch(Rectangle((x-.5, y-.5), 1, 1, color='#333'))
    # gap marker
    gap_y = H // 2
    ax.add_patch(Rectangle((W//2-.5, gap_y-.5), 1, 1, facecolor='none', edgecolor='red', lw=2, ls='--'))
    for j, (x, y) in enumerate(scene['spots']):
        ec = SPOT_EDGE[int(scene['spot_type'][j])]
        ax.add_patch(Rectangle((x-.5, y-.5), 1, 1, facecolor='#f5f5f5', edgecolor=ec, lw=2))
    if assign:
        for i, j in assign.items():
            cx, cy = car_positions.get(i, scene['car_pos'][i])
            sx, sy = scene['spots'][j]
            ax.plot([cx, sx], [cy, sy], ':', color=CAR_COLORS[int(scene['car_type'][i])], lw=1, alpha=0.5)
    for i, pos in car_positions.items():
        x, y = pos
        c = CAR_COLORS[int(scene['car_type'][i])]
        if waiting and i in waiting:
            ax.add_patch(Rectangle((x-.45, y-.45), .9, .9, color=c, alpha=0.5))
            ax.text(x, y, 'W', ha='center', va='center', fontsize=9, fontweight='bold', color='red')
        else:
            ax.add_patch(Rectangle((x-.45, y-.45), .9, .9, color=c))
            ax.text(x, y, str(i), ha='center', va='center', fontsize=7, color='white', fontweight='bold')
    ax.set_title(f'λ={lam:.2f}  {title_extra}', fontsize=11, fontweight='bold')


def render_static(scene, obstacles, pol, lam, out_png, midframe=False):
    cp = torch.tensor(scene['car_pos'], dtype=torch.float32)
    sp = torch.tensor(scene['spots'], dtype=torch.float32)
    ct = torch.tensor(scene['car_type'], dtype=torch.long)
    st = torch.tensor(scene['spot_type'], dtype=torch.long)
    with torch.no_grad():
        scores = pol(cp, sp, ct, st, lam).numpy()
    assign, _ = closed_loop_assign(scores)
    paths = []
    for i in sorted(assign.keys()):
        j = assign[i]
        p = bfs_path(W, H, obstacles, scene['car_pos'][i], scene['spots'][j])
        paths.append((i, p if p else [scene['car_pos'][i]]))
    path_list = [p for (_, p) in sorted(paths, key=lambda x: x[0])]
    trace, wait, nconf = simulate_movement_trace(path_list)
    if midframe and len(trace) > 3:
        t = len(trace) // 2
        pos = {i: trace[t][i] for i in range(len(path_list))}
        wset = set(i for i in range(len(path_list))
                   if t >= 1 and trace[t][i] == trace[t-1][i] and trace[t][i] != path_list[i][-1])
        extra = f'(mid, wait={wait}, conflicts={nconf})'
    else:
        pos = {i: scene['car_pos'][i] for i in range(len(path_list))}
        wset = None
        extra = f'(start, conflicts={nconf})'
    fig, ax = plt.subplots(figsize=(7, 5), dpi=150)
    draw(ax, scene, obstacles, pos, assign, lam, waiting=wset, title_extra=extra)
    fig.tight_layout(); fig.savefig(out_png); plt.close(fig)
    print(f"  static -> {os.path.basename(out_png)} wait={wait} conf={nconf}", flush=True)
    return wait, nconf


def render_gif(scene, obstacles, scores, lam, out_gif):
    assign, _ = closed_loop_assign(scores)
    paths = []
    for i in sorted(assign.keys()):
        j = assign[i]
        p = bfs_path(W, H, obstacles, scene['car_pos'][i], scene['spots'][j])
        paths.append(p if p else [scene['car_pos'][i]])
    trace, wait, nconf = simulate_movement_trace(paths)
    n_lead, slow, n_tail = 5, 3, 5
    frames = []
    for _ in range(n_lead):
        frames.append({i: scene['car_pos'][i] for i in range(len(paths))})
    for t in range(len(trace)):
        for _ in range(slow):
            frames.append({i: trace[t][i] for i in range(len(paths))})
    for _ in range(n_tail):
        frames.append({i: trace[-1][i] for i in range(len(paths))})
    fig, ax = plt.subplots(figsize=(6, 6), dpi=90)

    def render(t):
        pos = frames[t]
        wset = set()
        if t >= 1:
            for i in range(len(paths)):
                if frames[t][i] == frames[t-1][i] and frames[t][i] != paths[i][-1]:
                    wset.add(i)
        draw(ax, scene, obstacles, pos, assign, lam, waiting=wset if wset else None,
             title_extra=f'wait={wait} conf={nconf}')
    ani = animation.FuncAnimation(fig, render, frames=len(frames), interval=120)
    ani.save(out_gif, writer=animation.PillowWriter(fps=8))
    plt.close(fig)
    print(f"  gif -> {os.path.basename(out_gif)} ({len(frames)} frames, wait={wait}, conf={nconf})", flush=True)


def main():
    print("="*70); print("narrow-aisle visualization"); print("="*70, flush=True)
    scene, obstacles, train = build_demo()
    lam_star = read_lambda_star()
    pols = {}
    for lam in [0.0, lam_star]:
        pol = L.ps.ParkingPolicy()
        L.train_base(pol, train, lam, n_epoch=20)
        pols[lam] = pol
    render_static(scene, obstacles, pols[0.0], 0.0, os.path.join(L.FIG_DIR, 'l2_scene_narrow_lam0.png'))
    render_static(scene, obstacles, pols[0.0], 0.0, os.path.join(L.FIG_DIR, 'l2_scene_narrow_lam0_mid.png'), midframe=True)
    render_static(scene, obstacles, pols[lam_star], lam_star, os.path.join(L.FIG_DIR, 'l2_scene_narrow_lamstar.png'))
    render_static(scene, obstacles, pols[lam_star], lam_star, os.path.join(L.FIG_DIR, 'l2_scene_narrow_lamstar_mid.png'), midframe=True)
    # GIF
    for lam, tag in [(0.0, 'lam0'), (lam_star, 'lamstar')]:
        cp = torch.tensor(scene['car_pos'], dtype=torch.float32)
        sp = torch.tensor(scene['spots'], dtype=torch.float32)
        ct = torch.tensor(scene['car_type'], dtype=torch.long)
        st = torch.tensor(scene['spot_type'], dtype=torch.long)
        with torch.no_grad():
            scores = pols[lam](cp, sp, ct, st, lam).numpy()
        render_gif(scene, obstacles, scores, lam,
                   os.path.join(L.FIG_DIR, f'l2_anim_narrow_{tag}.gif'))
    print("[viz done]", flush=True)


if __name__ == '__main__':
    main()
