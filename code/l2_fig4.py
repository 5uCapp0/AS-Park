# -*- coding: utf-8 -*-
"""
    Figure-4 alignment experiment (core, paper Sec. 8.6).
"""
import os, json, time
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import l2_common as L

P_GRID = [0.0, 0.15, 0.3, 0.45, 0.6]
FAMS = [L.FAMILY_A, L.FAMILY_B]
R_DIM = 9
M_IN = L.N_TYPE
N_A = 1


def estimate_eps(fam, p, seed):
    rng = np.random.default_rng(seed)
    obstacles, W, H = L.build_family(fam, rng)
    tr = L.gen_family(L.N_TRAIN, fam, obstacles, p, rng)
    nA = int(round(len(tr) * 2 / 3))
    foldB = tr[nA:]
    e = np.array(L.defect_energy(foldB))
    eps2 = float(np.mean(e ** 2))
    sigma2 = float(np.var(e, ddof=1)) if len(e) > 1 else 0.0
    return eps2, sigma2


def measured_lambda_star_from_a1(fam_name, p):
    path = os.path.join(L.DATA_DIR, 'l2_a1.json')
    if not os.path.exists(path):
        return None, None
    with open(path, 'r', encoding='utf-8') as f:
        a1 = json.load(f)
    for res in a1['results']:
        if res['family'] == fam_name:
            c = res['cells'][str(p)]
            return c['best_lam'], c['mean_closed']
    return None, None


def n_scan(fam, p, seed):
    out = {}
    for n in [50, 150]:
        rng = np.random.default_rng(seed)
        obstacles, W, H = L.build_family(fam, rng)
        tr = L.gen_family(n, fam, obstacles, p, rng)
        te = L.gen_family(L.N_TEST, fam, obstacles, p, rng)
        gaps = {}
        for lam in L.LAM_GRID:
            pol = L.ps.ParkingPolicy()
            L.train_base(pol, tr, lam)
            cg, _, _, _ = L.closed_loop_evaluate(pol, te, obstacles, lam, W, H)
            gaps[lam] = cg
        out[n] = min(gaps, key=gaps.get)
        print(f"    n_scan fam={fam['name']} p={p} n={n} λ*={out[n]:.2f}", flush=True)
    return out


def main():
    t0 = time.time()
    print("="*70); print("Fig4 alignment"); print("="*70, flush=True)
    rows = []
    for fam in FAMS:
        M_eff = fam['M']
        kappa = L.kappa_theorem61(R_DIM, M_IN, N_A, L.N_TRAIN, M_eff)
        for p in P_GRID:
            eps_list, sig_list = [], []
            for seed in L.SEEDS:
                e, s = estimate_eps(fam, p, seed)
                eps_list.append(e); sig_list.append(s)
            eps2 = float(np.mean(eps_list))
            sigma2 = float(np.mean(sig_list))
            lam_hat = L.lambda_star_closed(eps2, sigma2, L.N_TRAIN, kappa)
            meas_lam, meas_curve = measured_lambda_star_from_a1(fam['name'], p)
            rows.append(dict(family=fam['name'], p=p, eps2=eps2, sigma2=sigma2,
                             kappa=kappa, lam_hat=lam_hat, lam_meas=meas_lam,
                             meas_curve=meas_curve))
            print(f"  [{fam['name']}] p={p:.2f} ε̂²={eps2:.4f} σ̂²={sigma2:.4f} "
                  f"kappa_hat={kappa:.4f} lam_hat*={lam_hat:.3f} measured lam*={meas_lam}", flush=True)
    nscan = {}
    for fam in FAMS:
        nscan[fam['name']] = {}
        for seed in L.SEEDS:
            r = n_scan(fam, 0.45, seed)
            for n, v in r.items():
                nscan[fam['name']].setdefault(str(n), []).append(v)
        nscan[fam['name']] = {n: float(np.mean(v)) for n, v in nscan[fam['name']].items()}
        print(f"  n_scan [{fam['name']}] p=0.45: {nscan[fam['name']]}", flush=True)

    def corr_for(fam_name):
        rs = [r for r in rows if r['family'] == fam_name]
        x = [r['lam_hat'] for r in rs]
        y = [r['lam_meas'] for r in rs if r['lam_meas'] is not None]
        x2 = [r['lam_hat'] for r in rs if r['lam_meas'] is not None]
        if len(y) >= 2:
            pear = float(np.corrcoef(x2, y)[0, 1])
            mad = float(np.mean(np.abs(np.array(x2) - np.array(y))))
        else:
            pear, mad = float('nan'), float('nan')
        return pear, mad, x2, y

    align = {}
    for fam in FAMS:
        pear, mad, x2, y = corr_for(fam['name'])
        align[fam['name']] = dict(pearson=pear, mad=mad, lam_hat=x2, lam_meas=y)
        print(f"  alignment [{fam['name']}] Pearson={pear:.3f} MAD={mad:.3f}", flush=True)

    out = dict(script=os.path.basename(__file__), date='2026-10-01', seeds=L.SEEDS,
               n_train=L.N_TRAIN, r_dim=R_DIM, m_in=M_IN, n_a=N_A,
               rows=[{k: v for k, v in r.items() if k != 'meas_curve'} for r in rows],
               n_scan=nscan, alignment=align)
    with open(os.path.join(L.DATA_DIR, 'l2_fig4.json'), 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    # CSV
    with open(os.path.join(L.DATA_DIR, 'l2_fig4.csv'), 'w', encoding='utf-8') as f:
        f.write('family,p,eps2,sigma2,kappa,lam_hat,lam_meas\n')
        for r in rows:
            f.write(f"{r['family']},{r['p']},{r['eps2']:.5f},{r['sigma2']:.5f},"
                    f"{r['kappa']:.5f},{r['lam_hat']:.4f},{r['lam_meas']}\n")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), dpi=150)
    for ax, fam in zip(axes, FAMS):
        rs = [r for r in rows if r['family'] == fam['name']]
        ps_ = [r['p'] for r in rs]
        lh = [r['lam_hat'] for r in rs]
        lm = [r['lam_meas'] for r in rs]
        ax.plot(ps_, lh, 'o-', color='#2980b9', lw=2, ms=7, label=r'theory $\hat\lambda^*$')
        ax.plot(ps_, lm, 's--', color='#c0392b', lw=2, ms=7, label=r'measured $\lambda^*$')
        ax.set_xlabel('heterogeneity p'); ax.set_ylabel(r'$\lambda^*$')
        ax.set_title(f"{fam['name']}  (r={align[fam['name']]['pearson']:.2f})", fontweight='bold')
        ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.suptitle('Fig4 alignment: closed-form vs measured optimum', fontweight='bold')
    fig.tight_layout()
    fig.savefig(os.path.join(L.FIG_DIR, 'l2_fig4_alignment.png'))
    plt.close(fig)
    print(f"[fig4 done] {time.time()-t0:.0f}s", flush=True)


if __name__ == '__main__':
    main()
