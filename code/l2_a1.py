# -*- coding: utf-8 -*-
"""
    A1: lambda ablation (paper Sec. 8.6).
"""
import os, json, time
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import l2_common as L
import l2_stats as S

P_GRID = [0.0, 0.15, 0.3, 0.45, 0.6]
FAMS = [L.FAMILY_A, L.FAMILY_B]


def run_family(fam):
    res = dict(family=fam['name'], params={k: fam[k] for k in ('W','H','n_cars','M')},
               p_grid=P_GRID, lam_grid=L.LAM_GRID, seeds=L.SEEDS, cells={})
    for p in P_GRID:
        agg = {lam: dict(closed=[], open_=[], wait=[], conf=[], dead=[]) for lam in L.LAM_GRID}
        for seed in L.SEEDS:
            rng = np.random.default_rng(seed)
            obstacles, W, H = L.build_family(fam, rng)
            tr = L.gen_family(L.N_TRAIN, fam, obstacles, p, rng)
            te = L.gen_family(L.N_TEST, fam, obstacles, p, rng)
            for lam in L.LAM_GRID:
                pol = L.ps.ParkingPolicy()
                L.train_base(pol, tr, lam)
                og = L.open_loop_gap(pol, te, lam)
                cg, wt, cf, dd = L.closed_loop_evaluate(pol, te, obstacles, lam, W, H)
                agg[lam]['closed'].append(cg)
                agg[lam]['open_'].append(og)
                agg[lam]['wait'].append(wt)
                agg[lam]['conf'].append(cf)
                agg[lam]['dead'].append(dd)
            print(f"  [{fam['name']}] p={p:.2f} seed={seed} done", flush=True)
        mean_closed = {lam: float(np.mean(agg[lam]['closed'])) for lam in L.LAM_GRID}
        best_lam = min(mean_closed, key=mean_closed.get)
        best_vals = agg[best_lam]['closed']
        ref0 = agg[0.0]['closed']
        ref1 = agg[1.0]['closed']
        stats_best0 = S.compare_group('best_vs_lam0', best_vals, ref0)
        stats_best1 = S.compare_group('best_vs_lam1', best_vals, ref1)
        res['cells'][str(p)] = dict(
            mean_closed={str(k): round(v,4) for k,v in mean_closed.items()},
            mean_open={str(lam): round(float(np.mean(agg[lam]['open_'])),4) for lam in L.LAM_GRID},
            mean_wait={str(lam): round(float(np.mean(agg[lam]['wait'])),3) for lam in L.LAM_GRID},
            mean_conf={str(lam): round(float(np.mean(agg[lam]['conf'])),3) for lam in L.LAM_GRID},
            mean_dead={str(lam): round(float(np.mean(agg[lam]['dead'])),3) for lam in L.LAM_GRID},
            best_lam=best_lam,
            best_closed=round(float(np.mean(best_vals)),4),
            best_vs_lam0=stats_best0,
            best_vs_lam1=stats_best1,
        )
        print(f"  >>> p={p:.2f} λ*={best_lam:.2f} closed_gap={mean_closed[best_lam]:.4f} "
              f"(λ=0:{mean_closed[0.0]:.4f}, λ=1:{mean_closed[1.0]:.4f})", flush=True)
    return res


def plot_all(results):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), dpi=150)
    cmap = plt.cm.viridis
    for ax, res in zip(axes, results):
        for i, p in enumerate(P_GRID):
            ys = [res['cells'][str(p)]['mean_closed'][str(lam)] for lam in L.LAM_GRID]
            ax.plot(L.LAM_GRID, ys, 'o-', color=cmap(i/ (len(P_GRID)-1)),
                    lw=1.8, ms=5, label=f'p={p:.2f}')
        ax.set_xlabel(r'symmetry injection $\lambda$')
        ax.set_ylabel('closed-loop optimality gap')
        ax.set_title(res['family'], fontweight='bold')
        ax.grid(alpha=0.3); ax.legend(fontsize=7, ncol=2)
    fig.suptitle('A1: interior optimum in closed loop (both families)', fontweight='bold')
    fig.tight_layout()
    out = os.path.join(L.FIG_DIR, 'l2_a1_lambda_scan.png')
    fig.savefig(out); plt.close(fig)
    return out


def main():
    t0 = time.time()
    print("="*70); print("A1 lambda ablation"); print("="*70, flush=True)
    allres = []
    for fam in FAMS:
        print(f"\n--- family {fam['name']} ---", flush=True)
        allres.append(run_family(fam))
    # JSON
    out_json = os.path.join(L.DATA_DIR, 'l2_a1.json')
    payload = dict(script=os.path.basename(__file__), date='2026-10-01',
                   seeds=L.SEEDS, n_train=L.N_TRAIN, n_test=L.N_TEST,
                   n_epoch=L.N_EPOCH, results=allres)
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    # CSV
    out_csv = os.path.join(L.DATA_DIR, 'l2_a1.csv')
    with open(out_csv, 'w', encoding='utf-8') as f:
        f.write('family,p,lam,mean_closed_gap,mean_open_gap,mean_wait,mean_conf,mean_dead\n')
        for res in allres:
            for p in P_GRID:
                c = res['cells'][str(p)]
                for lam in L.LAM_GRID:
                    f.write(f"{res['family']},{p},{lam},"
                            f"{c['mean_closed'][str(lam)]},{c['mean_open'][str(lam)]},"
                            f"{c['mean_wait'][str(lam)]},{c['mean_conf'][str(lam)]},"
                            f"{c['mean_dead'][str(lam)]}\n")
    out_fig = plot_all(allres)
    print(f"\n[A1 done] {time.time()-t0:.0f}s -> {out_json}\n           {out_csv}\n           {out_fig}", flush=True)


if __name__ == '__main__':
    main()
