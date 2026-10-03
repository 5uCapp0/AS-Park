# -*- coding: utf-8 -*-
"""
    A3: estimator ablation (Proposition 5.5 sensitivity shrinkage).
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
P = 0.45
R_DIM, M_IN, N_A = 9, L.N_TYPE, 1
NOISE = 0.20


def nearest_lam(val):
    return min(L.LAM_GRID, key=lambda x: abs(x - val))


def run():
    print("="*70); print(f"A3 estimator ablation family={FAM['name']} p={P}"); print("="*70, flush=True)
    methods = ['closed_form', 'fixed_0.3', 'fixed_0.5', 'fixed_0.7', 'oracle',
               'cf_noise_p20', 'cf_noise_m20']
    acc = {m: [] for m in methods}
    lam_used = {m: [] for m in ['closed_form', 'oracle', 'cf_noise_p20', 'cf_noise_m20']}
    eps2_rec, lamhat_rec = [], []
    for seed in L.SEEDS:
        rng = np.random.default_rng(seed)
        obstacles, W, H = L.build_family(FAM, rng)
        tr = L.gen_family(L.N_TRAIN, FAM, obstacles, P, rng)
        te = L.gen_family(L.N_TEST, FAM, obstacles, P, rng)
        pols, gaps = {}, {}
        for lam in L.LAM_GRID:
            pol = L.ps.ParkingPolicy()
            L.train_base(pol, tr, lam)
            cg, _, _, _ = L.closed_loop_evaluate(pol, te, obstacles, lam, W, H)
            pols[lam] = pol; gaps[lam] = cg
        nA = int(round(len(tr) * 2 / 3))
        e = np.array(L.defect_energy(tr[nA:]))
        eps2 = float(np.mean(e ** 2))
        sigma2 = float(np.var(e, ddof=1)) if len(e) > 1 else 0.0
        kappa = L.kappa_theorem61(R_DIM, M_IN, N_A, L.N_TRAIN, FAM['M'])
        eps2_rec.append(eps2)
        lh = L.lambda_star_closed(eps2, sigma2, L.N_TRAIN, kappa)
        lamhat_rec.append(lh)
        lh_grid = nearest_lam(lh)
        acc['closed_form'].append(gaps[lh_grid]); lam_used['closed_form'].append(lh_grid)
        for m, lam in [('fixed_0.3', 0.3), ('fixed_0.5', 0.5), ('fixed_0.7', 0.7)]:
            acc[m].append(gaps[nearest_lam(lam)])
        best_lam = min(gaps, key=gaps.get)
        acc['oracle'].append(gaps[best_lam]); lam_used['oracle'].append(best_lam)
        for m, eta in [('cf_noise_p20', NOISE), ('cf_noise_m20', -NOISE)]:
            eps2_n = eps2 * (1 + eta)
            lh_n = L.lambda_star_closed(eps2_n, sigma2, L.N_TRAIN, kappa)
            lh_n_grid = nearest_lam(lh_n)
            acc[m].append(gaps[lh_n_grid]); lam_used[m].append(lh_n_grid)
        print(f"  seed={seed} ε̂²={eps2:.4f} λ̂*={lh:.3f}(→{lh_grid}) "
              f"oracleλ={best_lam:.2f}", flush=True)

    summary = {}
    for m in methods:
        v = acc[m]
        summary[m] = dict(mean=float(np.mean(v)), std=float(np.std(v, ddof=1)) if len(v) > 1 else 0.0,
                          ci=S.bootstrap_ci(v))
    stat_cf_oracle = S.compare_group('closed_form_vs_oracle', acc['closed_form'], acc['oracle'])
    stat_cf_fixed = S.compare_group('closed_form_vs_fixed0.5', acc['closed_form'], acc['fixed_0.5'])
    lh_mean = float(np.mean(lamhat_rec))
    theo_dev = (1 - lh_mean) * NOISE
    acc_dev = float(np.mean(np.abs(np.array(lam_used['cf_noise_p20']) - np.array(lam_used['closed_form']))))
    print(f"  ε̂² mean={np.mean(eps2_rec):.4f} λ̂* mean={lh_mean:.3f}", flush=True)
    print(f"  Prop5.5: theoretical lambda bias~=(1-lam_hat*)eta={theo_dev:.3f}  measured grid jump={acc_dev:.3f}", flush=True)

    out = dict(script=os.path.basename(__file__), date='2026-10-01', seeds=L.SEEDS,
               family=FAM['name'], p=P, eps2_mean=float(np.mean(eps2_rec)),
               lambda_hat_mean=lh_mean, theo_lambda_dev_prop55=theo_dev,
               observed_grid_step=acc_dev,
               summary=summary,
               stat_closed_form_vs_oracle=stat_cf_oracle,
               stat_closed_form_vs_fixed05=stat_cf_fixed,
               lam_used={k: [float(x) for x in v] for k, v in lam_used.items()})
    with open(os.path.join(L.DATA_DIR, 'l2_a3.json'), 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    with open(os.path.join(L.DATA_DIR, 'l2_a3.csv'), 'w', encoding='utf-8') as f:
        f.write('method,seed,closed_gap\n')
        for m in methods:
            for s, v in enumerate(acc[m]):
                f.write(f"{m},{L.SEEDS[s]},{v:.5f}\n")

    fig, ax = plt.subplots(figsize=(7.4, 4.4), dpi=150)
    labels = ['closed-form', 'fixed .3', 'fixed .5', 'fixed .7', 'oracle-tuned',
              'cf +20%ε', 'cf -20%ε']
    means = [summary[m]['mean'] for m in methods]
    stds = [summary[m]['std'] for m in methods]
    ax.bar(range(len(labels)), means, yerr=stds, capsize=4,
           color=['#2980b9', '#95a5a6', '#95a5a6', '#95a5a6', '#27ae60', '#e67e22', '#e67e22'])
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=20, fontsize=8)
    ax.set_ylabel('closed-loop gap'); ax.set_title('A3: estimator comparison (lower=better)', fontweight='bold')
    ax.grid(alpha=0.3, axis='y')
    fig.tight_layout()
    fig.savefig(os.path.join(L.FIG_DIR, 'l2_a3_estimator.png')); plt.close(fig)
    print(f"[A3 done]", flush=True)


if __name__ == '__main__':
    run()
