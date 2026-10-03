# -*- coding: utf-8 -*-
"""
    Merge per-seed L3 job results into a single JSON.
"""
import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, 'data')
FIG = os.path.join(ROOT, 'figs')
os.makedirs(FIG, exist_ok=True)

SEEDS = [20260925, 7, 42, 123, 2024]
SIGMA2_GRID = [0.25, 0.5, 1.0, 2.0]
LAM_GRID = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]


def welch(g0, g1):
    from scipy import stats as st
    t, p = st.ttest_ind(np.array(g0), np.array(g1), equal_var=False)
    return {'t': float(t), 'p': float(p)}


def main():
    # ---------- sigma sensitivity ----------
    out = {}
    for s2 in SIGMA2_GRID:
        per_seed = {}
        for s in SEEDS:
            p = os.path.join(DATA, 'job_sigma_%s_%d.json' % (s2, s))
            if not os.path.exists(p):
                raise SystemExit('missing %s' % p)
            per_seed[str(s)] = json.load(open(p, encoding='utf-8'))['res']
        agg = {}
        for lam in LAM_GRID:
            gs = [per_seed[str(s)][str(lam)]['gap'] for s in SEEDS]
            agg[lam] = {'mean': float(np.mean(gs)), 'std': float(np.std(gs))}
        gs_star = [per_seed[str(s)]['lambda_star']['gap'] for s in SEEDS]
        agg['lambda_star'] = {'mean': float(np.mean(gs_star)), 'std': float(np.std(gs_star))}
        agg['lam_star_mean'] = float(np.mean([per_seed[str(s)]['_lam_star'] for s in SEEDS]))
        agg['eps2_mean'] = float(np.mean([per_seed[str(s)]['_eps2'] for s in SEEDS]))
        agg['vs_lam0'] = welch([per_seed[str(s)]['0.0']['gap'] for s in SEEDS], gs_star)
        agg['vs_lam1'] = welch([per_seed[str(s)]['1.0']['gap'] for s in SEEDS], gs_star)
        out[str(s2)] = {'per_seed': per_seed, 'aggregate': agg}
        a = agg
        print('sigma2=%s lam*_mean=%.4f eps2=%.4f gap(lam*)=%.2f%% vs0 p=%.4f vs1 p=%.4f'
              % (s2, a['lam_star_mean'], a['eps2_mean'], a['lambda_star']['mean'] * 100,
                 a['vs_lam0']['p'], a['vs_lam1']['p']))
    with open(os.path.join(DATA, 'l3_sigma_sensitivity_result.json'), 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), dpi=120)
    xs = SIGMA2_GRID
    lam_stars = [out[str(s)]['aggregate']['lam_star_mean'] for s in xs]
    eps2m = float(np.mean([out[str(s)]['aggregate']['eps2_mean'] for s in xs]))
    tt = np.linspace(0.2, 2.2, 100)
    theor = eps2m / (eps2m + tt / 8.0)
    axes[0].plot(tt, theor, '--', color='#7f8c8d', lw=1.5, label='closed form (ε² mean)')
    axes[0].plot(xs, lam_stars, 'o-', color='#c0392b', lw=2, label='measured λ* (5 seeds)')
    axes[0].set_xlabel('σ² (noise level)'); axes[0].set_ylabel('λ*')
    axes[0].set_title('λ* is monotone-decreasing in σ²', fontsize=10, fontweight='bold')
    axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3)
    colors = ['#95a5a6', '#16a085', '#2980b9', '#c0392b']
    for i, s in enumerate(xs):
        agg = out[str(s)]['aggregate']
        xs2 = LAM_GRID + [agg['lam_star_mean']]
        ys2 = [agg[l]['mean'] * 100 for l in LAM_GRID] + [agg['lambda_star']['mean'] * 100]
        errs2 = [agg[l]['std'] * 100 for l in LAM_GRID] + [agg['lambda_star']['std'] * 100]
        axes[1].errorbar(xs2, ys2, yerr=errs2, fmt='o-', color=colors[i], capsize=3, lw=1.3,
                         label='σ²=%s' % s)
        axes[1].scatter([agg['lam_star_mean']], [agg['lambda_star']['mean'] * 100], s=45,
                        facecolor='none', edgecolor=colors[i], zorder=5)
    axes[1].set_xlabel('λ'); axes[1].set_ylabel('gap vs oracle (%)')
    axes[1].set_title('U-shaped troughs: λ* stays interior at every σ²', fontsize=10, fontweight='bold')
    axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, 'l3_sigma_sensitivity.png'))
    plt.close()
    print('sigma sensitivity merged + figure saved')

    # ---------- 2D grid ----------
    per_seed = {}
    eps2s, lamstars = [], []
    for s in SEEDS:
        p = os.path.join(DATA, 'job_2d_%d.json' % s)
        if not os.path.exists(p):
            raise SystemExit('missing %s' % p)
        d = json.load(open(p, encoding='utf-8'))
        per_seed[str(s)] = d['res']
        eps2s.append(d['eps2']); lamstars.append(d['lam_star'])
    agg2 = {}
    for lam in LAM_GRID:
        gs = [per_seed[str(s)][str(lam)]['gap'] for s in SEEDS]
        agg2[lam] = {'mean': float(np.mean(gs)), 'std': float(np.std(gs))}
    gs_star2 = [per_seed[str(s)]['lambda_star']['gap'] for s in SEEDS]
    agg2['lambda_star'] = {'mean': float(np.mean(gs_star2)), 'std': float(np.std(gs_star2))}
    agg2['lam_star_mean'] = float(np.mean(lamstars))
    agg2['eps2_mean'] = float(np.mean(eps2s))
    agg2['vs_lam0'] = welch([per_seed[str(s)]['0.0']['gap'] for s in SEEDS], gs_star2)
    agg2['vs_lam1'] = welch([per_seed[str(s)]['1.0']['gap'] for s in SEEDS], gs_star2)
    print('2D: lam*_mean=%.4f eps2=%.4f gap(lam*)=%.2f%% vs0 p=%.4f vs1 p=%.4f'
          % (agg2['lam_star_mean'], agg2['eps2_mean'], agg2['lambda_star']['mean'] * 100,
             agg2['vs_lam0']['p'], agg2['vs_lam1']['p']))
    out2 = {'env': '2D 4x4 grid lot, M=16 N=8 frac=0.35, Manhattan dist, online arrivals, 5 seeds, sigma2=1.0',
            'lam_grid': LAM_GRID, 'lambda_star_mean': agg2['lam_star_mean'],
            'eps2_mean': agg2['eps2_mean'], 'per_seed': per_seed,
            'aggregate': agg2, 'welch': {'vs_lam0': agg2['vs_lam0'], 'vs_lam1': agg2['vs_lam1']}}
    with open(os.path.join(DATA, 'l3_online_2d_result.json'), 'w', encoding='utf-8') as f:
        json.dump(out2, f, ensure_ascii=False, indent=2)

    xs3 = LAM_GRID + [agg2['lam_star_mean']]
    ys3 = [agg2[l]['mean'] * 100 for l in LAM_GRID] + [agg2['lambda_star']['mean'] * 100]
    errs3 = [agg2[l]['std'] * 100 for l in LAM_GRID] + [agg2['lambda_star']['std'] * 100]
    plt.figure(figsize=(6.2, 3.8), dpi=120)
    plt.errorbar(xs3, ys3, yerr=errs3, fmt='o-', color='#2c3e50', capsize=4, label='online gap (mean±std)')
    plt.axvline(agg2['lam_star_mean'], color='#c0392b', ls='--', lw=1.2,
                label='λ*=%.2f' % agg2['lam_star_mean'])
    plt.xlabel('symmetry-injection strength λ'); plt.ylabel('optimality gap vs oracle (%)')
    plt.title('2D grid lot: soft-equivariant online dispatcher, λ tunes the gap', fontsize=10, fontweight='bold')
    plt.legend(fontsize=8); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(FIG, 'l3_2d_lambda_scan.png')); plt.close()

    names = ['λ=0 (anonymous)', 'λ=1 (personalized)', 'λ* (adaptive)']
    vals = [agg2[0.0]['mean'] * 100, agg2[1.0]['mean'] * 100, agg2['lambda_star']['mean'] * 100]
    errs4 = [agg2[0.0]['std'] * 100, agg2[1.0]['std'] * 100, agg2['lambda_star']['std'] * 100]
    plt.figure(figsize=(6.2, 3.8), dpi=120)
    plt.bar(names, vals, yerr=errs4, capsize=5, color=['#95a5a6', '#bdc3c7', '#c0392b'])
    plt.ylabel('optimality gap vs oracle (%)')
    plt.title('2D grid lot: adaptive λ* vs extremes (5 seeds, mean±std)', fontsize=10, fontweight='bold')
    plt.grid(alpha=0.3, axis='y'); plt.tight_layout()
    plt.savefig(os.path.join(FIG, 'l3_2d_compare.png')); plt.close()
    print('2D merged + figures saved')

    # ---------- main L3-Online experiment (seeded rerun, authoritative) ----------
    per_seed = {}
    eps2s, lamstars = [], []
    for sd in SEEDS:
        pj = os.path.join(DATA, 'job_main_%d.json' % sd)
        if not os.path.exists(pj):
            raise SystemExit('missing %s' % pj)
        dj = json.load(open(pj, encoding='utf-8'))
        per_seed[str(sd)] = dj['res']
        eps2s.append(dj['eps2']); lamstars.append(dj['lam_star'])
    aggm = {}
    for lam in LAM_GRID:
        gs = [per_seed[str(sd)][str(lam)]['gap'] for sd in SEEDS]
        aggm[lam] = {'mean': float(np.mean(gs)), 'std': float(np.std(gs))}
    gs_star_m = [per_seed[str(sd)]['lambda_star']['gap'] for sd in SEEDS]
    aggm['lambda_star'] = {'mean': float(np.mean(gs_star_m)), 'std': float(np.std(gs_star_m))}
    aggm['lam_star_mean'] = float(np.mean(lamstars))
    aggm['eps2_mean'] = float(np.mean(eps2s))
    aggm['vs_lam0'] = welch([per_seed[str(sd)]['0.0']['gap'] for sd in SEEDS], gs_star_m)
    aggm['vs_lam1'] = welch([per_seed[str(sd)]['1.0']['gap'] for sd in SEEDS], gs_star_m)
    print('MAIN: lam*_mean=%.4f eps2=%.4f gap(lam*)=%.2f%% vs0 p=%.4f vs1 p=%.4f'
          % (aggm['lam_star_mean'], aggm['eps2_mean'], aggm['lambda_star']['mean'] * 100,
             aggm['vs_lam0']['p'], aggm['vs_lam1']['p']))
    outm = {'env': 'L3-Online main: 1D aisle M=16 N=8 frac=0.35, 600/150 ep, 5 seeds, sigma2=1.0 (seeded rerun)',
            'lam_grid': LAM_GRID, 'lambda_star_mean': aggm['lam_star_mean'],
            'eps2_mean': aggm['eps2_mean'], 'per_seed': per_seed,
            'aggregate': aggm, 'welch': {'vs_lam0': aggm['vs_lam0'], 'vs_lam1': aggm['vs_lam1']}}
    with open(os.path.join(DATA, 'l3_online_dispatcher_result.json'), 'w', encoding='utf-8') as f:
        json.dump(outm, f, ensure_ascii=False, indent=2)
    print('main experiment merged -> l3_online_dispatcher_result.json')


if __name__ == '__main__':
    main()
