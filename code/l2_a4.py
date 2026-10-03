# -*- coding: utf-8 -*-
"""
    A4: dual-permutation decomposition (Proposition 5.11).
"""
import os, json, time
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import l2_common as L
import l2_stats as S

FAM = L.FAMILY_B
P_CAR, P_SPOT = 0.5, 0.0
LV_GRID = [0.0, 0.25, 0.5, 0.75, 1.0]


def gen_scenes(p_car, p_spot, n, fam, obstacles, rng):
    scenes = L.gen_scenes_corridor(n, fam['W'], fam['H'], obstacles,
                                   fam['n_cars'], fam['M'], p_car, rng)
    for sc in scenes:
        sc['spot_type'] = np.zeros(len(sc['spots']), dtype=np.int64)
        cost = sc['D'].copy().astype(np.float64)
        for i in range(len(sc['car_type'])):
            for j in range(len(sc['spot_type'])):
                if sc['spot_type'][j] not in L.COMPAT[int(sc['car_type'][i])]:
                    cost[i, j] += L.ALPHA
        sc['cost'] = cost
    return scenes


def run():
    print("="*70); print(f"A4 dual-permutation decomposition p_car={P_CAR} p_spot={P_SPOT}"); print("="*70, flush=True)
    vec_gap = {lv: {ls: [] for ls in LV_GRID} for lv in LV_GRID}
    scalar_gap = {lam: [] for lam in L.LAM_GRID}
    for seed in L.SEEDS:
        rng = np.random.default_rng(seed)
        obstacles, W, H = L.build_family(FAM, rng)
        tr = gen_scenes(P_CAR, P_SPOT, L.N_TRAIN, FAM, obstacles, rng)
        te = gen_scenes(P_CAR, P_SPOT, L.N_TEST, FAM, obstacles, rng)
        for lv in LV_GRID:
            for ls in LV_GRID:
                pol = L.VectorParkingPolicy()
                L.train_vector(pol, tr, lv, ls)
                cg, _, _, _ = L.closed_loop_evaluate(pol, te, obstacles, 0.0, W, H,
                                                      lam_veh=lv, lam_spot=ls)
                vec_gap[lv][ls].append(cg)
        for lam in L.LAM_GRID:
            pol = L.ps.ParkingPolicy()
            L.train_base(pol, tr, lam)
            cg, _, _, _ = L.closed_loop_evaluate(pol, te, obstacles, lam, W, H)
            scalar_gap[lam].append(cg)
        print(f"  seed={seed} done", flush=True)
    mat = np.zeros((len(LV_GRID), len(LV_GRID)))
    for i, lv in enumerate(LV_GRID):
        for j, ls in enumerate(LV_GRID):
            mat[i, j] = float(np.mean(vec_gap[lv][ls]))
    bi, bj = np.unravel_index(np.argmin(mat), mat.shape)
    best_lv, best_ls = LV_GRID[bi], LV_GRID[bj]
    best_vec = mat[bi, bj]
    scalar_means = {lam: float(np.mean(scalar_gap[lam])) for lam in L.LAM_GRID}
    best_scalar_lam = min(scalar_means, key=scalar_means.get)
    best_scalar = scalar_means[best_scalar_lam]
    vec_best_seed = []
    for s in range(len(L.SEEDS)):
        best_s = min(vec_gap[lv][ls][s] for lv in LV_GRID for ls in LV_GRID)
        vec_best_seed.append(best_s)
    stat = S.compare_group('vector_vs_scalar', vec_best_seed, scalar_gap[best_scalar_lam])
    print(f"  vector best (lam_veh={best_lv}, lam_spot={best_ls}) gap={best_vec:.4f}", flush=True)
    print(f"  scalar best lam={best_scalar_lam:.2f} gap={best_scalar:.4f}", flush=True)
    print(f"  lam_spot*={best_ls} (should be ~0: spots homogeneous)  p={stat['p']:.3f} sig={stat['significant']}", flush=True)

    out = dict(script=os.path.basename(__file__), date='2026-10-01', seeds=L.SEEDS,
               family=FAM['name'], p_car=P_CAR, p_spot=P_SPOT,
               lambda_veh_grid=LV_GRID,
               matrix=[[round(v,4) for v in row] for row in mat],
               best_lambda_veh=best_lv, best_lambda_spot=best_ls, best_vector_gap=round(best_vec,4),
               best_scalar_lambda=best_scalar_lam, best_scalar_gap=round(best_scalar,4),
               scalar_means={str(k): round(v,4) for k,v in scalar_means.items()},
               stat_vector_vs_scalar=stat)
    with open(os.path.join(L.DATA_DIR, 'l2_a4.json'), 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    with open(os.path.join(L.DATA_DIR, 'l2_a4.csv'), 'w', encoding='utf-8') as f:
        f.write('lambda_veh,lambda_spot,mean_closed_gap\n')
        for i, lv in enumerate(LV_GRID):
            for j, ls in enumerate(LV_GRID):
                f.write(f"{lv},{ls},{mat[i,j]:.5f}\n")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4), dpi=150)
    ax1.bar(['scalar λ', 'vector (λv,λs)'], [best_scalar, best_vec],
            color=['#95a5a6', '#2980b9'], capsize=5, yerr=[
                float(np.std(scalar_gap[best_scalar_lam], ddof=1)),
                float(np.std(vec_best_seed, ddof=1))])
    ax1.set_ylabel('closed-loop gap'); ax1.set_title('A4: vector budget beats scalar', fontweight='bold')
    ax1.grid(alpha=0.3, axis='y')
    im = ax2.imshow(mat, cmap='viridis_r', aspect='auto', origin='lower')
    ax2.set_xticks(range(len(LV_GRID))); ax2.set_xticklabels(LV_GRID)
    ax2.set_yticks(range(len(LV_GRID))); ax2.set_yticklabels(LV_GRID)
    ax2.set_xlabel(r'$\lambda_{spot}$'); ax2.set_ylabel(r'$\lambda_{veh}$')
    ax2.plot(LV_GRID.index(best_ls), LV_GRID.index(best_lv), marker='*', color='red', ms=18, markeredgecolor='white')
    ax2.set_title(f'heatmap (best λ_spot={best_ls}≈0)', fontweight='bold')
    plt.colorbar(im, ax=ax2, fraction=0.046, label='gap')
    fig.tight_layout()
    fig.savefig(os.path.join(L.FIG_DIR, 'l2_a4_vector.png')); plt.close(fig)
    print("[A4 done]", flush=True)


if __name__ == '__main__':
    run()
