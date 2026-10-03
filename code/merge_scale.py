# -*- coding: utf-8 -*-
"""
    Aggregate l3 scale job results into paper-ready JSONs.
"""
import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats as st

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, 'data')
FIGS = os.path.join(ROOT, 'figs')
os.makedirs(FIGS, exist_ok=True)

SEEDS_SCALE = [1024, 2048, 4096, 8192, 16384, 2026, 2027, 2028, 42, 7]
SIZES = [20, 50, 100]
LAM_GRID = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]


def cohens_d(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    na, nb = len(a), len(b)
    sp = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    return float((a.mean() - b.mean()) / sp) if sp > 0 else 0.0


def main():
    data_dir = sys.argv[1] if len(sys.argv) > 1 else DATA
    summaries = {}
    for N in SIZES:
        per_seed = {}
        lamstars, eps2s = [], []
        for sd in SEEDS_SCALE:
            p = os.path.join(data_dir, 'scale_%d_%d.json' % (N, sd))
            if not os.path.exists(p):
                raise SystemExit('missing %s' % p)
            d = json.load(open(p, encoding='utf-8'))
            per_seed[str(sd)] = d['res']
            lamstars.append(d['lam_star'])
            eps2s.append(d['eps2'])
        agg = {}
        for lam in LAM_GRID:
            gs = [per_seed[str(sd)][str(lam)]['gap'] for sd in SEEDS_SCALE]
            agg[str(lam)] = {'mean': float(np.mean(gs)), 'std': float(np.std(gs))}
        gs_star = [per_seed[str(sd)]['lambda_star']['gap'] for sd in SEEDS_SCALE]
        agg['lambda_star'] = {'mean': float(np.mean(gs_star)), 'std': float(np.std(gs_star))}
        agg['lam_star_mean'] = float(np.mean(lamstars))
        agg['eps2_mean'] = float(np.mean(eps2s))
        for name, other in [('vs_lam0', '0.0'), ('vs_lam1', '1.0')]:
            g0 = np.array([per_seed[str(sd)][other]['gap'] for sd in SEEDS_SCALE])
            t, p = st.ttest_ind(g0, np.array(gs_star), equal_var=False)
            agg[name] = {'t': float(t), 'p': float(p), 'cohens_d': cohens_d(g0, np.array(gs_star))}
        # also report lam0 vs lam1 contrast
        g0 = np.array([per_seed[str(sd)]['0.0']['gap'] for sd in SEEDS_SCALE])
        g1 = np.array([per_seed[str(sd)]['1.0']['gap'] for sd in SEEDS_SCALE])
        t, p = st.ttest_ind(g0, g1, equal_var=False)
        agg['lam0_vs_lam1'] = {'t': float(t), 'p': float(p)}
        out = {'N': N, 'M': 4 * N, 'seeds': len(SEEDS_SCALE), 'aggregate': agg, 'per_seed': per_seed}
        with open(os.path.join(data_dir, 'scale_N%d_result.json' % N), 'w', encoding='utf-8') as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        summaries[N] = agg
        a = agg
        print('N=%d: lam*_mean=%.4f eps2=%.4f gap(lam0)=%.2f%% gap(lam*)=%.2f%% gap(lam1)=%.2f%% '
              'vs0 p=%.4f(d=%.2f) vs1 p=%.4f(d=%.2f)' % (
                  N, a['lam_star_mean'], a['eps2_mean'], a['0.0']['mean'] * 100,
                  a['lambda_star']['mean'] * 100, a['1.0']['mean'] * 100,
                  a['vs_lam0']['p'], a['vs_lam0']['cohens_d'],
                  a['vs_lam1']['p'], a['vs_lam1']['cohens_d']))

    # ---- figures ----
    # Fig A: gap vs lambda (mean over seeds, shaded std) per N
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    colors = {20: '#1f77b4', 50: '#ff7f0e', 100: '#2ca02c'}
    for N in SIZES:
        a = summaries[N]
        xs = LAM_GRID + [a['lam_star_mean']]
        ys = [a[str(lam)]['mean'] * 100 for lam in LAM_GRID] + [a['lambda_star']['mean'] * 100]
        stds = [a[str(lam)]['std'] * 100 for lam in LAM_GRID] + [a['lambda_star']['std'] * 100]
        ax.errorbar(xs, ys, yerr=stds, marker='o', capsize=3, label='N=%d' % N,
                    color=colors[N], linestyle='--', linewidth=1.2)
        ax.scatter([a['lam_star_mean']], [a['lambda_star']['mean'] * 100],
                   marker='*', s=220, color=colors[N], zorder=5, edgecolors='k')
    ax.set_xlabel(r'$\lambda$ (symmetry strength)')
    ax.set_ylabel('gap vs offline oracle (%)')
    ax.set_title('L3-Online on 2D grid: gap vs $\\lambda$ (10 seeds)')
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, 'scale_gap_vs_lambda.png'), dpi=200)
    plt.close(fig)

    # Fig B: gap(lambda*) vs N with lambda=0 / lambda=1 reference lines
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    Ns = SIZES
    for key, mk, lab in [('0.0', 'o', r'$\lambda=0$ (fully equivariant)'),
                         ('lambda_star', '*', r'$\lambda^\*$ (closed-form)'),
                         ('1.0', 's', r'$\lambda=1$ (fully learned)')]:
        ys = [summaries[N][key]['mean'] * 100 for N in Ns]
        errs = [summaries[N][key]['std'] * 100 for N in Ns]
        ax.errorbar(Ns, ys, yerr=errs, marker=mk, capsize=3, label=lab, linewidth=1.5)
    ax.set_xlabel('number of agents N')
    ax.set_ylabel('gap vs offline oracle (%)')
    ax.set_title('L3-Online: scale effect on $\\lambda^\\*$')
    ax.set_xticks(Ns)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, 'scale_gap_vs_N.png'), dpi=200)
    plt.close(fig)

    # Fig C: lambda* vs N (theory check: lambda* increases with N)
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    Ns = SIZES
    lam_means = [summaries[N]['lam_star_mean'] for N in Ns]
    ax.plot(Ns, lam_means, marker='o', linewidth=1.8, color='#8e44ad')
    ax.set_xlabel('number of agents N')
    ax.set_ylabel(r'$\lambda^\*$ (closed-form optimum)')
    ax.set_title(r'Closed-form $\lambda^\*$ vs scale (theory: increasing in N)')
    ax.set_xticks(Ns)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, 'scale_lamstar_vs_N.png'), dpi=200)
    plt.close(fig)

    print('FIGS SAVED: scale_gap_vs_lambda.png / scale_gap_vs_N.png / scale_lamstar_vs_N.png')


if __name__ == '__main__':
    main()
