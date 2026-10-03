# -*- coding: utf-8 -*-
"""
    A2: capacity ablation (Theorem 6.2 / diagonal law).
"""
import os, json, time
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import l2_common as L

R_GRID = [0, 1, 2, 4, 8]
P_GRID = [0.0, 0.45]
FAM = L.FAMILY_B


def run():
    print("="*70); print("A2 capacity ablation (family=%s)" % FAM['name']); print("="*70, flush=True)
    result = dict(family=FAM['name'], r_grid=R_GRID, p_grid=P_GRID,
                  lam_grid=L.LAM_GRID, seeds=L.SEEDS, grids={})
    for p in P_GRID:
        # grid_mean[r][lam] = mean closed gap over seeds
        grid_mean = {r: {lam: [] for lam in L.LAM_GRID} for r in R_GRID}
        for seed in L.SEEDS:
            rng = np.random.default_rng(seed)
            obstacles, W, H = L.build_family(FAM, rng)
            tr = L.gen_family(L.N_TRAIN, FAM, obstacles, p, rng)
            te = L.gen_family(L.N_TEST, FAM, obstacles, p, rng)
            for r in R_GRID:
                for lam in L.LAM_GRID:
                    if r == 0 and lam > 0.0:
                        continue
                    pol = L.LowRankParkingPolicy(r=r)
                    L.train_base(pol, tr, lam)
                    cg, wt, cf, dd = L.closed_loop_evaluate(pol, te, obstacles, lam, W, H)
                    grid_mean[r][lam].append(cg)
            print(f"  p={p:.2f} seed={seed} done", flush=True)
        mat = np.zeros((len(R_GRID), len(L.LAM_GRID)))
        for i, r in enumerate(R_GRID):
            for j, lam in enumerate(L.LAM_GRID):
                if r == 0:
                    vals = grid_mean[0][0.0]
                else:
                    vals = grid_mean[r][lam]
                mat[i, j] = float(np.mean(vals))
        ri, li = np.unravel_index(np.argmin(mat), mat.shape)
        r_star, lam_star = R_GRID[ri], L.LAM_GRID[li]
        lam_star_per_r = {}
        for i, r in enumerate(R_GRID):
            best_j = int(np.argmin(mat[i, :]))
            lam_star_per_r[r] = L.LAM_GRID[best_j]
        result['grids'][str(p)] = dict(
            matrix=[[round(v,4) for v in row] for row in mat],
            r_star=r_star, lam_star=lam_star,
            lam_star_per_r={str(k): v for k, v in lam_star_per_r.items()},
        )
        print(f"  >>> p={p:.2f} best (r*={r_star}, lam*={lam_star:.2f}) gap={mat[ri,li]:.4f}", flush=True)
        print(f"      best lam* per r: {lam_star_per_r}", flush=True)
    return result


def plot_heatmap(result):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), dpi=150)
    for ax, p in zip(axes, P_GRID):
        mat = np.array(result['grids'][str(p)]['matrix'])
        im = ax.imshow(mat, cmap='viridis_r', aspect='auto', origin='lower')
        ax.set_xticks(range(len(L.LAM_GRID))); ax.set_xticklabels(L.LAM_GRID)
        ax.set_yticks(range(len(R_GRID))); ax.set_yticklabels(R_GRID)
        ax.set_xlabel(r'$\lambda$'); ax.set_ylabel('residual rank r')
        rs = result['grids'][str(p)]
        ri = R_GRID.index(rs['r_star']); li = L.LAM_GRID.index(rs['lam_star'])
        ax.plot(li, ri, marker='*', color='red', ms=18, markeredgecolor='white')
        ax.set_title(f"p={p:.2f}  (r*={rs['r_star']}, λ*={rs['lam_star']:.2f})", fontweight='bold')
        plt.colorbar(im, ax=ax, fraction=0.046, label='closed-loop gap')
    fig.suptitle('A2: capacity × symmetry heatmap (diagonal law)', fontweight='bold')
    fig.tight_layout()
    out = os.path.join(L.FIG_DIR, 'l2_a2_heatmap.png')
    fig.savefig(out); plt.close(fig)
    return out


def plot_lambda_vs_r(result):
    fig, ax = plt.subplots(figsize=(6.6, 4.2), dpi=150)
    for p in P_GRID:
        lspr = result['grids'][str(p)]['lam_star_per_r']
        rs = sorted(int(k) for k in lspr)
        ys = [lspr[str(r)] for r in rs]
        ax.plot(rs, ys, 'o-', lw=2, ms=7, label=f'p={p:.2f}')
    ax.set_xlabel('residual rank r'); ax.set_ylabel(r'optimal $\lambda^*(r)$')
    ax.set_title('A2: λ* decreases with r (Theorem 6.2)', fontweight='bold')
    ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout()
    out = os.path.join(L.FIG_DIR, 'l2_a2_lambda_vs_r.png')
    fig.savefig(out); plt.close(fig)
    return out


def main():
    t0 = time.time()
    res = run()
    payload = dict(script=os.path.basename(__file__), date='2026-10-01',
                   n_train=L.N_TRAIN, **res)
    with open(os.path.join(L.DATA_DIR, 'l2_a2.json'), 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    with open(os.path.join(L.DATA_DIR, 'l2_a2.csv'), 'w', encoding='utf-8') as f:
        f.write('p,r,lam,mean_closed_gap\n')
        for p in P_GRID:
            mat = res['grids'][str(p)]['matrix']
            for i, r in enumerate(R_GRID):
                for j, lam in enumerate(L.LAM_GRID):
                    f.write(f"{p},{r},{lam},{mat[i][j]}\n")
    f1 = plot_heatmap(res); f2 = plot_lambda_vs_r(res)
    print(f"[A2 done] {time.time()-t0:.0f}s -> {f1}\n           {f2}", flush=True)


if __name__ == '__main__':
    main()
