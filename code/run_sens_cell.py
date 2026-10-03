# -*- coding: utf-8 -*-
"""
    Run a single sigma^2 sensitivity cell.
"""
import os, sys, json, argparse
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import l3_sigma_sensitivity as ss

DATA = os.path.join(os.path.dirname(HERE), 'data')
LAM_GRID = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]


def aggregate(per_seed, seeds):
    lamstars = np.array([per_seed[str(s)]['_lam_star'] for s in seeds])
    eps2s = np.array([per_seed[str(s)]['_eps2'] for s in seeds])
    agg = {}
    for lam in LAM_GRID:
        gs = [per_seed[str(s)][str(lam)]['gap'] for s in seeds]
        agg[lam] = {'mean': float(np.mean(gs)), 'std': float(np.std(gs))}
    gs_star = [per_seed[str(s)]['lambda_star']['gap'] for s in seeds]
    agg['lambda_star'] = {'mean': float(np.mean(gs_star)), 'std': float(np.std(gs_star))}
    agg['lam_star_mean'] = float(np.mean(lamstars))
    agg['eps2_mean'] = float(np.mean(eps2s))
    from scipy import stats as st
    for name, other in [('vs_lam0', '0.0'), ('vs_lam1', '1.0')]:
        g0 = np.array([per_seed[str(s)][other]['gap'] for s in seeds])
        g1 = np.array(gs_star)
        t, p = st.ttest_ind(g0, g1, equal_var=False)
        agg[name] = {'t': float(t), 'p': float(p)}
    return agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sigma2', type=float, required=True)
    ap.add_argument('--out', type=str, required=True)
    a = ap.parse_args()

    if a.sigma2 == 1.0:
        mj_path = os.path.join(DATA, 'l3_main_10seed_result.json')
        mj = json.load(open(mj_path, encoding='utf-8'))
        per_seed = {str(s): mj['per_seed'][str(s)] for s in mj['seeds']}
        seeds = mj['seeds']
        agg = aggregate(per_seed, seeds)
        out = {str(1.0): {'per_seed': per_seed, 'aggregate': agg, 'reused_main10': True}}
        json.dump(out, open(os.path.join(DATA, a.out), 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
        print(f"sigma2=1.0 reused main-10 ({len(seeds)} seeds) -> {a.out}")
        print(f"  lam*_mean={agg['lam_star_mean']:.4f} gap(lam*)={agg['lambda_star']['mean']*100:.2f}% "
              f"vs0 p={agg['vs_lam0']['p']:.4f} vs1 p={agg['vs_lam1']['p']:.4f}")
        return

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("device:", device, flush=True)
    per_seed = {}
    for seed in ss.SEEDS:
        print(f"--- sigma2={a.sigma2} seed {seed} ---", flush=True)
        res, eps2, lam_star = ss.run_seed_sigma(seed, device, a.sigma2)
        per_seed[str(seed)] = res
        print(f"  eps2={eps2:.4f} lam*={lam_star:.4f} gap*={res['lambda_star']['gap']*100:.2f}%", flush=True)
    agg = aggregate(per_seed, ss.SEEDS)
    out = {str(a.sigma2): {'per_seed': per_seed, 'aggregate': agg}}
    json.dump(out, open(os.path.join(DATA, a.out), 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
    print(f"Saved {a.out}", flush=True)
    print(f"  lam*_mean={agg['lam_star_mean']:.4f} eps2_mean={agg['eps2_mean']:.4f} "
          f"gap(lam*)={agg['lambda_star']['mean']*100:.2f}% "
          f"vs0 p={agg['vs_lam0']['p']:.4f} vs1 p={agg['vs_lam1']['p']:.4f}", flush=True)


if __name__ == '__main__':
    main()
