# -*- coding: utf-8 -*-
"""
    Merge partA/partB runs into l3_main_10seed_result.json.
"""
import os, json
import numpy as np
from scipy import stats as st

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(os.path.dirname(HERE), 'data')
LAM_GRID = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

def stats_welch(a, b):
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    t, p = st.ttest_ind(a, b, equal_var=False)
    na, nb = len(a), len(b)
    sp = np.sqrt(((na - 1) * a.std(ddof=1) ** 2 + (nb - 1) * b.std(ddof=1) ** 2) / (na + nb - 2))
    d = (a.mean() - b.mean()) / sp if sp > 0 else float('nan')
    se = np.sqrt(a.var(ddof=1) / na + b.var(ddof=1) / nb)
    df = na + nb - 2
    ci = (a.mean() - b.mean()) + np.array([-1.0, 1.0]) * st.t.ppf(0.975, df) * se
    return {'t': float(t), 'p': float(p), 'd': float(d), 'ci95': [float(ci[0]), float(ci[1])]}

parts = [os.path.join(DATA, 'partA50.json'), os.path.join(DATA, 'partB50.json')]
per_seed, all_seeds = {}, []
for pp in parts:
    d = json.load(open(pp, encoding='utf-8'))
    per_seed.update(d['per_seed'])
    all_seeds.extend(d['seeds'])
seeds = [int(s) for s in all_seeds]
lam_stars = np.array([per_seed[str(s)]['_lam_star'] for s in seeds])
eps2s = np.array([per_seed[str(s)]['_eps2'] for s in seeds])
gs_star = np.array([per_seed[str(s)]['lambda_star']['gap'] for s in seeds])

grid_agg = {}
for lam in LAM_GRID:
    gs = np.array([per_seed[str(s)][str(lam)]['gap'] for s in seeds])
    grid_agg[lam] = {'mean': float(gs.mean()), 'std': float(gs.std(ddof=1))}
g0 = np.array([per_seed[str(s)]['0.0']['gap'] for s in seeds])
g1 = np.array([per_seed[str(s)]['1.0']['gap'] for s in seeds])

out = {
    'config': 'L3-Online main, M=16 N=8 frac=0.35, 1-D aisle, sigma2=1.0, lr=1e-2, 50 epochs, fixed torch seeds',
    'seeds': seeds,
    'lam_grid': LAM_GRID,
    'grid_gap_mean_std_pct': {str(k): {'mean_pct': v['mean'] * 100, 'std_pct': v['std'] * 100} for k, v in grid_agg.items()},
    'lam_star_mean': float(lam_stars.mean()),
    'lam_star_std': float(lam_stars.std(ddof=1)),
    'eps2_mean': float(eps2s.mean()),
    'gap_lambda_star_mean_pct': float(gs_star.mean() * 100),
    'gap_lambda_star_std_pct': float(gs_star.std(ddof=1) * 100),
    'welch_vs_lam0': stats_welch(g0, gs_star),
    'welch_vs_lam1': stats_welch(g1, gs_star),
    'per_seed': per_seed,
}
out_path = os.path.join(DATA, 'l3_main_10seed_result.json')
json.dump(out, open(out_path, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)

print('=== SUMMARY (50ep, sigma2=1.0, 10 seeds) ===')
print(json.dumps({k: v for k, v in out.items() if k not in ('per_seed',)}, ensure_ascii=False, indent=2))
for s in seeds:
    r = per_seed[str(s)]
    print(f"seed {s}: lam*={r['_lam_star']:.4f} gap*={r['lambda_star']['gap']*100:.2f}% "
          f"gap0={r['0.0']['gap']*100:.2f}% gap1={r['1.0']['gap']*100:.2f}%")
