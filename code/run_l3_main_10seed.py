# -*- coding: utf-8 -*-
"""
    Run the L3 main experiment with 10 seeds.
"""
import os
import json
import numpy as np
import torch
from scipy import stats as st
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import l3_online_dispatcher as lod
import l3_sigma_sensitivity as ss  # reuse run_seed_sigma at sigma2=1.0

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, 'data')
FIG = os.path.join(ROOT, 'figs')
os.makedirs(DATA, exist_ok=True)
os.makedirs(FIG, exist_ok=True)

LAM_GRID = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
SEEDS_10 = [20260925, 7, 42, 123, 2024, 314159, 271828, 161803, 141421, 101010]
SIGMA2 = 1.0


def stats_welch(a, b):
    """Welch t-test, Cohen's d, 95% CI of mean difference."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    t, p = st.ttest_ind(a, b, equal_var=False)
    na, nb = len(a), len(b)
    sp = np.sqrt(((na - 1) * a.std(ddof=1) ** 2 + (nb - 1) * b.std(ddof=1) ** 2) / (na + nb - 2))
    d = (a.mean() - b.mean()) / sp if sp > 0 else float('nan')
    se = np.sqrt(a.var(ddof=1) / na + b.var(ddof=1) / nb)
    df = na + nb - 2
    ci = (a.mean() - b.mean()) + np.array([-1.0, 1.0]) * st.t.ppf(0.975, df) * se
    return {'t': float(t), 'p': float(p), 'd': float(d), 'ci95': [float(ci[0]), float(ci[1])]}


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("device:", device, flush=True)
    results = {}
    for seed in SEEDS_10:
        print(f"--- seed {seed} (sigma2={SIGMA2}) ---", flush=True)
        res, eps2, lam_star = ss.run_seed_sigma(seed, device, SIGMA2)
        results[str(seed)] = res  # full lambda-grid results + _eps2 + _lam_star
        print(f"  eps2={eps2:.4f} lam*={lam_star:.4f} gap*={res['lambda_star']['gap']*100:.2f}% "
              f"gap0={res[0.0]['gap']*100:.2f}% gap1={res[1.0]['gap']*100:.2f}%", flush=True)

    lam_stars = np.array([results[str(s)]['_lam_star'] for s in SEEDS_10])
    eps2s = np.array([results[str(s)]['_eps2'] for s in SEEDS_10])
    gs_star = np.array([results[str(s)]['lambda_star']['gap'] for s in SEEDS_10])

    grid_agg = {}
    for lam in LAM_GRID:
        gs = np.array([results[str(s)][lam]['gap'] for s in SEEDS_10])
        grid_agg[lam] = {'mean': float(gs.mean()), 'std': float(gs.std(ddof=1))}
    g0 = np.array([results[str(s)][0.0]['gap'] for s in SEEDS_10])
    g1 = np.array([results[str(s)][1.0]['gap'] for s in SEEDS_10])

    out = {
        'config': f'L3-Online main, M=16 N=8 frac=0.35, 1-D aisle, sigma2={SIGMA2}, 600/150 eps, 10 seeds, fixed torch seeds',
        'seeds': SEEDS_10,
        'lam_grid': LAM_GRID,
        'grid_gap_mean_std_pct': {str(k): {'mean_pct': v['mean'] * 100, 'std_pct': v['std'] * 100}
                                  for k, v in grid_agg.items()},
        'lam_star_mean': float(lam_stars.mean()),
        'lam_star_std': float(lam_stars.std(ddof=1)),
        'eps2_mean': float(eps2s.mean()),
        'gap_lambda_star_mean_pct': float(gs_star.mean() * 100),
        'gap_lambda_star_std_pct': float(gs_star.std(ddof=1) * 100),
        'welch_vs_lam0': stats_welch(g0, gs_star),
        'welch_vs_lam1': stats_welch(g1, gs_star),
        'per_seed': results,
    }
    with open(os.path.join(DATA, 'l3_main_10seed_result.json'), 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("=== SUMMARY ===", flush=True)
    print(json.dumps({k: v for k, v in out.items() if k not in ('per_seed',)},
                     ensure_ascii=False, indent=2), flush=True)

    # ---- figures ----
    xs = LAM_GRID + [out['lam_star_mean']]
    ys = [grid_agg[l]['mean'] * 100 for l in LAM_GRID] + [gs_star.mean() * 100]
    errs = [grid_agg[l]['std'] * 100 for l in LAM_GRID] + [gs_star.std(ddof=1) * 100]
    plt.figure(figsize=(6.4, 4.0), dpi=120)
    plt.errorbar(xs, ys, yerr=errs, fmt='o-', color='#2c3e50', capsize=4,
                 label='closed-loop gap (mean±std, 10 seeds)')
    plt.axvline(out['lam_star_mean'], color='#c0392b', ls='--', lw=1.3,
                label=f"λ*={out['lam_star_mean']:.2f} (closed form)")
    plt.xlabel('symmetry-injection strength λ')
    plt.ylabel('optimality gap vs oracle (%)')
    plt.title('L3-Online (10 seeds, fixed seeds): λ scan is U-shaped; λ* in the trough',
              fontsize=10, fontweight='bold')
    plt.legend(fontsize=8); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(FIG, 'l3_main_10seed_scan.png')); plt.close()

    names = ['λ=0 (anonymous)', f"λ*={out['lam_star_mean']:.2f} (adaptive)", 'λ=1 (personalized)']
    vals = [g0.mean() * 100, gs_star.mean() * 100, g1.mean() * 100]
    errs2 = [g0.std(ddof=1) * 100, gs_star.std(ddof=1) * 100, g1.std(ddof=1) * 100]
    plt.figure(figsize=(6.4, 4.0), dpi=120)
    bars = plt.bar(names, vals, yerr=errs2, capsize=5, color=['#95a5a6', '#c0392b', '#bdc3c7'])
    for b, v in zip(bars, vals):
        plt.text(b.get_x() + b.get_width() / 2, v + 0.4, f'{v:.1f}%', ha='center', fontsize=9)
    plt.ylabel('optimality gap vs oracle (%)')
    plt.title('L3-Online: adaptive λ* vs extremes (10 seeds, mean±std)', fontsize=10, fontweight='bold')
    plt.grid(alpha=0.3, axis='y'); plt.tight_layout()
    plt.savefig(os.path.join(FIG, 'l3_main_10seed_compare.png')); plt.close()
    print("Saved json + figures.", flush=True)


if __name__ == '__main__':
    main()
