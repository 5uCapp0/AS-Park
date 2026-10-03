# -*- coding: utf-8 -*-
"""
    L2 standard benchmark extension on offline 2D lots.
"""
import os
import sys
import json
import time
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

import l2_common as L
import l2_stats as S

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = L.DATA_DIR
FIG = L.FIG_DIR

SEEDS10 = [20260925, 7, 42, 123, 2024, 314159, 271828, 161803, 141421, 101010]
P_HETERO = 0.30
N_TRAIN = L.N_TRAIN      # 120
N_TEST = L.N_TEST        # 60
R_DIM = 9
M_IN = L.N_TYPE
N_A = 1

SCALE_CFG = {
    20:  dict(W=12, H=12),
    50:  dict(W=16, H=16),
    100: dict(W=24, H=24),
}

def build_warehouse_obstacles(W, H):
    obs = set()
    for x in range(2, W - 2, 6):
        for y in range(1, H - 1):
            obs.add((x, y))
            obs.add((x + 1, y))
    return obs


def build_random30_obstacles(W, H, rng):
    obs = set()
    interior = [(x, y) for x in range(1, W - 1) for y in range(1, H - 1)]
    n_obs = int(round(0.30 * len(interior)))
    for c in rng.choice(len(interior), size=n_obs, replace=False):
        obs.add(interior[c])
    return obs


def build_family_obstacles(name, W, H, rng):
    if name == 'empty':
        return set()
    if name == 'warehouse':
        return build_warehouse_obstacles(W, H)
    if name == 'random30':
        return build_random30_obstacles(W, H, rng)
    raise ValueError(name)


def gen_open_scenes(n, W, H, obstacles, N, M, p_hetero, rng):
    old = L.ps.rng
    L.ps.rng = rng
    try:
        scenes = [L.ps.make_scene(W, H, obstacles, N, M, p_hetero) for _ in range(n)]
    finally:
        L.ps.rng = old
    return scenes


def defect_energy_penalized(scenes):
    from scipy.optimize import linear_sum_assignment
    out = []
    for sc in scenes:
        _, _, c_full = L.ps.oracle(sc['cost'])
        r_pos, c_pos = linear_sum_assignment(sc['D'])
        c_anon = float(sc['cost'][r_pos, c_pos].sum())
        out.append((c_anon - c_full) / (c_full + 1e-9))
    return out


SIGMA2_FIXED = 1.0

def closed_form_lam(train_scenes, M_eff, n_train=N_TRAIN, sigma2_mode='fixed'):
    e = np.array(defect_energy_penalized(train_scenes))
    eps2 = float(np.mean(e))
    sigma2 = SIGMA2_FIXED if sigma2_mode == 'fixed' else (float(np.var(e, ddof=1)) if len(e) > 1 else 0.0)
    kappa = L.kappa_theorem61(R_DIM, M_IN, N_A, n_train, M_eff)
    lam_hat = L.lambda_star_closed(eps2, sigma2, n_train, kappa)
    lam_hat = float(np.clip(lam_hat, 0.0, 1.0))
    return dict(eps2=eps2, sigma2=sigma2, kappa=kappa, lam_hat=lam_hat)


def cohens_d(a, b):
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return float('nan')
    sp = np.sqrt(((na - 1) * a.std(ddof=1) ** 2 + (nb - 1) * b.std(ddof=1) ** 2) / (na + nb - 2))
    if sp == 0:
        return float('nan')
    return float((a.mean() - b.mean()) / sp)


def holm_correct(pvals, alpha=0.05):
    order = np.argsort(pvals)
    n = len(pvals)
    adj = np.full(n, float('nan'))
    for rank, idx in enumerate(order):
        adj[idx] = min(1.0, pvals[idx] * (n - rank))
    rej = adj <= alpha
    return adj.tolist(), rej.tolist()


def run_cell(fam_name, N, W, H, seeds):
    lam_hats = []
    per_seed = {lam_key: [] for lam_key in ('lam0', 'lamstar', 'lam1')}
    waits = {lam_key: [] for lam_key in ('lam0', 'lamstar', 'lam1')}
    confs = {lam_key: [] for lam_key in ('lam0', 'lamstar', 'lam1')}
    deads = {lam_key: [] for lam_key in ('lam0', 'lamstar', 'lam1')}
    for seed in seeds:
        rng = np.random.default_rng(seed)
        obstacles = build_family_obstacles(fam_name, W, H, rng)
        tr = gen_open_scenes(N_TRAIN, W, H, obstacles, N, N, P_HETERO, rng)
        te = gen_open_scenes(N_TEST, W, H, obstacles, N, N, P_HETERO, rng)
        cf = closed_form_lam(tr, M_eff=N)
        lam_hats.append(cf['lam_hat'])
        for lam_key, lam in (('lam0', 0.0), ('lamstar', cf['lam_hat']), ('lam1', 1.0)):
            pol = L.ps.ParkingPolicy()
            L.train_base(pol, tr, lam)
            cg, wt, cf_, dd = L.closed_loop_evaluate(pol, te, obstacles, lam, W, H)
            per_seed[lam_key].append(cg)
            waits[lam_key].append(wt)
            confs[lam_key].append(cf_)
            deads[lam_key].append(dd)
        print(f"  [{fam_name} N={N}] seed={seed} ε²={cf['eps2']:.4f} σ²={cf['sigma2']:.3f} "
              f"κ={cf['kappa']:.4f} λ̂*={cf['lam_hat']:.3f} done", flush=True)
    return dict(lam_hats=lam_hats, per_seed=per_seed, waits=waits, confs=confs, deads=deads)


def summarize_cell(fam_name, N, cell):
    out = dict(family=fam_name, N=N, P=P_HETERO,
               lam_hat_mean=float(np.mean(cell['lam_hats'])),
               lam_hat_std=float(np.std(cell['lam_hats'], ddof=1)) if len(cell['lam_hats']) > 1 else 0.0,
               arms={})
    keys = ['lam0', 'lamstar', 'lam1']
    pvals = []
    comps = []
    for k in keys:
        g = np.array(cell['per_seed'][k])
        out['arms'][k] = dict(
            gap_mean=float(np.mean(g)), gap_std=float(np.std(g, ddof=1)) if len(g) > 1 else 0.0,
            wait_mean=float(np.mean(cell['waits'][k])),
            conf_mean=float(np.mean(cell['confs'][k])),
            dead_mean=float(np.mean(cell['deads'][k])),
        )
    for name, a, b in (('star_vs_lam0', 'lamstar', 'lam0'),
                       ('star_vs_lam1', 'lamstar', 'lam1'),
                       ('lam0_vs_lam1', 'lam0', 'lam1')):
        ga = np.array(cell['per_seed'][a]); gb = np.array(cell['per_seed'][b])
        wt = S.welch_t(ga, gb)
        pvals.append(wt['p'])
        comps.append((name, a, b, ga, gb, wt))
    adj, rej = holm_correct(np.array(pvals))
    for (name, a, b, ga, gb, wt), p_adj, r in zip(comps, adj, rej):
        out['arms'][name] = dict(
            t=wt['t'], p=wt['p'], p_holm=float(p_adj), significant_holm=bool(r),
            cohens_d=cohens_d(ga, gb),
            mean_diff=float(np.mean(ga) - np.mean(gb)),
            n=len(ga),
        )
    return out


def plot_gap_vs_N(results, out_path):
    fams = list(results['families'].keys())
    n_rows = len(fams)
    fig, axes = plt.subplots(1, n_rows, figsize=(5.6 * n_rows, 4.2), dpi=150)
    if n_rows == 1:
        axes = [axes]
    styles = [('lam0', 's--', '#e67e22', r'$\lambda=0$ (anonymous)'),
              ('lamstar', 'o-', '#2ecc71', r'$\lambda=\hat{\lambda}^*$ (closed-form)'),
              ('lam1', '^--', '#c0392b', r'$\lambda=1$ (full symmetry)')]
    for ax, fam in zip(axes, fams):
        scales = sorted(int(k) for k in results['families'][fam].keys())
        for key, mk, col, lab in styles:
            ys = [results['families'][fam][str(s)]['arms'][key]['gap_mean'] * 100 for s in scales]
            errs = [results['families'][fam][str(s)]['arms'][key]['gap_std'] * 100 for s in scales]
            ax.errorbar(scales, ys, yerr=errs, fmt=mk, color=col, capsize=4, lw=1.6, ms=6, label=lab)
        ax.set_xlabel('number of arriving cars $N$')
        ax.set_ylabel('closed-loop optimality gap vs oracle (%)')
        ax.set_title(fam, fontweight='bold')
        ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.suptitle(r'L2 benchmark: adaptive $\hat{\lambda}^*$ vs $\lambda=0$ (anonymous) vs $\lambda=1$ (full symmetry), '
                 r'across standard maps and scales (10 seeds, mean±std)',
                 fontweight='bold', fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_path); plt.close(fig)
    return out_path


def plot_summary(results, out_path):
    fams = list(results['families'].keys())
    scales = sorted({int(s) for r in results['families'].values() for s in r.keys()})
    fig, axes = plt.subplots(len(scales), len(fams), figsize=(5.2 * len(fams), 4.0 * len(scales)), dpi=140,
                             squeeze=False)
    styles = [('lam0', '#e67e22'), ('lamstar', '#2ecc71'), ('lam1', '#c0392b')]
    for ri, s in enumerate(scales):
        for ci, fam in enumerate(fams):
            ax = axes[ri][ci]
            arm = results['families'][fam].get(str(s))
            if arm is None:
                ax.set_visible(False); continue
            labels = []
            for key, col in styles:
                g = arm['arms'][key]['gap_mean'] * 100
                sd = arm['arms'][key]['gap_std'] * 100
                labels.append((key, g, sd))
            x = np.arange(len(labels))
            ax.bar(x, [v[1] for v in labels], yerr=[v[2] for v in labels],
                   color=[dict(styles)[v[0]] for v in labels], capsize=4, alpha=0.85)
            ax.set_xticks(x)
            ax.set_xticklabels([r'$\lambda=0$', r'$\hat{\lambda}^*$', r'$\lambda=1$'], fontsize=9)
            ax.set_title(f'{fam} · N={s}', fontsize=11, fontweight='bold')
            ax.set_ylabel('gap (%)'); ax.grid(alpha=0.3, axis='y')
            ax.axhline(0, color='k', lw=0.8)
    fig.suptitle('L2 benchmark summary: closed-loop gap vs oracle (10 seeds, mean±std)', fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path); plt.close(fig)
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--families', default='empty,warehouse,random30')
    ap.add_argument('--scales', default='20,50,100')
    ap.add_argument('--seeds', default='10')
    ap.add_argument('--smoke', action='store_true', help='single seed + minimal smoke test')
    args = ap.parse_args()

    fams = [f.strip() for f in args.families.split(',')]
    scales = [int(s) for s in args.scales.split(',')]
    if args.seeds in ('10', 'all'):
        seeds = SEEDS10
    elif args.seeds.isdigit():
        seeds = SEEDS10[:int(args.seeds)]
    else:
        seeds = [int(s) for s in args.seeds.split(',')]
    if args.smoke:
        scales = [min(scales)]
        seeds = seeds[:1]
        print('SMOKE MODE:', fams, scales, seeds)

    t0 = time.time()
    results = dict(script=os.path.basename(__file__), date='2026-10-02',
                   p_hetero=P_HETERO, n_train=N_TRAIN, n_test=N_TEST,
                   seeds=seeds, scale_cfg={str(k): v for k, v in SCALE_CFG.items()},
                   families={})
    for fam in fams:
        results['families'][fam] = {}
        for N in scales:
            W, H = SCALE_CFG[N]['W'], SCALE_CFG[N]['H']
            print(f"\n=== family={fam} N={N} W={W} H={H} seeds={len(seeds)} ===", flush=True)
            cell = run_cell(fam, N, W, H, seeds)
            results['families'][fam][str(N)] = summarize_cell(fam, N, cell)
    if not args.smoke:
        print("\n=== reproduce main exp: open_dense / narrow_corridor ===", flush=True)
        results['families']['open_dense'] = {}
        results['families']['narrow_corridor'] = {}
        for fam, N, M in ((L.FAMILY_A, L.FAMILY_A['n_cars'], L.FAMILY_A['M']),
                          (L.FAMILY_B, L.FAMILY_B['n_cars'], L.FAMILY_B['M'])):
            W, H = fam['W'], fam['H']
            name = fam['name']
            per_seed = {k: [] for k in ('lam0', 'lamstar', 'lam1')}
            waits = {k: [] for k in ('lam0', 'lamstar', 'lam1')}
            confs = {k: [] for k in ('lam0', 'lamstar', 'lam1')}
            deads = {k: [] for k in ('lam0', 'lamstar', 'lam1')}
            lam_hats = []
            for seed in seeds:
                rng = np.random.default_rng(seed)
                obstacles, W_, H_ = L.build_family(fam, rng)
                tr = L.gen_family(N_TRAIN, fam, obstacles, P_HETERO, rng)
                te = L.gen_family(N_TEST, fam, obstacles, P_HETERO, rng)
                cf = closed_form_lam(tr, M_eff=M)
                lam_hats.append(cf['lam_hat'])
                for lam_key, lam in (('lam0', 0.0), ('lamstar', cf['lam_hat']), ('lam1', 1.0)):
                    pol = L.ps.ParkingPolicy()
                    L.train_base(pol, tr, lam)
                    cg, wt, cf_, dd = L.closed_loop_evaluate(pol, te, obstacles, lam, W, H)
                    per_seed[lam_key].append(cg); waits[lam_key].append(wt)
                    confs[lam_key].append(cf_); deads[lam_key].append(dd)
                print(f"  [{name}] seed={seed} λ̂*={cf['lam_hat']:.3f} done", flush=True)
            results['families'][name] = {str(N): summarize_cell(name, N, dict(
                lam_hats=lam_hats, per_seed=per_seed, waits=waits, confs=confs, deads=deads))}
    out_json = os.path.join(DATA, 'l2_benchmark_result.json')
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    f1 = plot_gap_vs_N(results, os.path.join(FIG, 'l2_benchmark_gap_vs_N.png'))
    f2 = plot_summary(results, os.path.join(FIG, 'l2_benchmark_summary.png'))
    print("\n" + "=" * 78)
    for fam in results['families']:
        for N in sorted(results['families'][fam], key=int):
            a = results['families'][fam][N]['arms']
            print(f"{fam:>14} N={N:>3}: "
                  f"λ0 {a['lam0']['gap_mean']*100:6.2f}% | "
                  f"λ* {a['lamstar']['gap_mean']*100:6.2f}% | "
                  f"λ1 {a['lam1']['gap_mean']*100:6.2f}%  "
                  f"[λ*vs0 t={a['star_vs_lam0']['t']:.2f} p_holm={a['star_vs_lam0']['p_holm']:.4f} "
                  f"d={a['star_vs_lam0']['cohens_d']:.2f} | "
                  f"λ*vs1 t={a['star_vs_lam1']['t']:.2f} p_holm={a['star_vs_lam1']['p_holm']:.4f} "
                  f"d={a['star_vs_lam1']['cohens_d']:.2f}]")
    print("=" * 78)
    print(f"[benchmark done] {time.time()-t0:.0f}s -> {out_json}\n{f1}\n{f2}")


if __name__ == '__main__':
    main()
