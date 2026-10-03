# -*- coding: utf-8 -*-
"""
    Attribute-aware equivariance auxiliary experiment.
"""
import os, json, sys, time, argparse
import numpy as np
import torch
import l3_sigma_sensitivity as ss

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, 'data')
os.makedirs(DATA, exist_ok=True)

SEEDS_10 = [20260925, 7, 42, 123, 2024, 314159, 271828, 161803, 141421, 101010]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=str, default=None)
    ap.add_argument('--out', type=str, default='l3_typeaware_result.json')
    args = ap.parse_args()
    seeds = [int(s) for s in (args.seeds or ','.join(map(str, SEEDS_10))).split(',') if s.strip()]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("device:", device, flush=True)
    t0 = time.time()

    per_seed = {}
    for seed in seeds:
        torch.manual_seed(seed)
        print(f"--- seed {seed} (typeAware equivariant backbone, sigma2=1.0, 50ep) ---", flush=True)
        res, eps2, lam_star = ss.run_seed_sigma(seed, device, 1.0, use_types=True)
        rec = {
            'lam0_typeAware_gap': res[0.0]['gap'],
            'lam0_typeAware_std': res[0.0]['std'],
            'lam1_gap': res[1.0]['gap'],
            'lam1_std': res[1.0]['std'],
            'lam_star_gap': res['lambda_star']['gap'],
            'lam_star_std': res['lambda_star']['std'],
            'eps2': float(eps2), 'lam_star': float(lam_star),
        }
        per_seed[str(seed)] = rec
        print(f"  lam0(typeAware)={rec['lam0_typeAware_gap']*100:.2f}%  lam1={rec['lam1_gap']*100:.2f}%  "
              f"lam*={rec['lam_star_gap']*100:.2f}%  eps2={eps2:.4f}  lam*={lam_star:.4f}", flush=True)

    g0 = [per_seed[s]['lam0_typeAware_gap'] for s in map(str, seeds)]
    g1 = [per_seed[s]['lam1_gap'] for s in map(str, seeds)]
    gs = [per_seed[s]['lam_star_gap'] for s in map(str, seeds)]
    agg = {
        'lam0_typeAware': {'mean': float(np.mean(g0)), 'std': float(np.std(g0))},
        'lam1': {'mean': float(np.mean(g1)), 'std': float(np.std(g1))},
        'lambda_star': {'mean': float(np.mean(gs)), 'std': float(np.std(gs))},
        'eps2_mean': float(np.mean([per_seed[s]['eps2'] for s in map(str, seeds)])),
        'lam_star_mean': float(np.mean([per_seed[s]['lam_star'] for s in map(str, seeds)])),
    }

    def welch(a, b):
        ma, va, na = np.mean(a), np.var(a, ddof=1), len(a)
        mb, vb, nb = np.mean(b), np.var(b, ddof=1), len(b)
        se = np.sqrt(va / na + vb / nb)
        t = (ma - mb) / (se + 1e-12)
        return float(t)

    # vs main-experiment authoritative arms (10-seed)
    mainp = os.path.join(DATA, 'l3_main_10seed_result.json')
    if os.path.exists(mainp):
        md = json.load(open(mainp, encoding='utf-8'))
        # use per-seed arrays from main json if present
        if 'per_seed' in md:
            m0 = [md['per_seed'][s]['0.0']['gap'] for s in md['per_seed']]
            ms = [md['per_seed'][s]['lambda_star']['gap'] for s in md['per_seed']]
            m1 = [md['per_seed'][s]['1.0']['gap'] for s in md['per_seed']]
            agg['vs_main_lam0_anon'] = {'t': welch(g0, m0), 'main_lam0_mean': float(np.mean(m0)),
                                        'main_lam0_std': float(np.std(m0))}
            agg['typeaware_vs_main_lamstar'] = {'t': welch(gs, ms), 'main_lamstar_mean': float(np.mean(ms)),
                                                'main_lamstar_std': float(np.std(ms))}
            agg['typeaware_vs_main_lam1'] = {'t': welch(g0, m1), 'main_lam1_mean': float(np.mean(m1)),
                                             'main_lam1_std': float(np.std(m1))}

    out = {'exp': 'l3_typeaware_arm', 'env': '1-D aisle M=16 N=8 frac=0.35, sigma2=1.0, 50ep, '
           'attribute-aware equivariant backbone (types in DeepSets, still S_N-equivariant), lambda=0 = pure typeAware equivariant',
           'seeds': seeds, 'runtime_sec': round(time.time() - t0, 1),
           'per_seed': per_seed, 'aggregate': agg}
    jp = os.path.join(DATA, args.out)
    with open(jp, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("Saved:", jp, flush=True)
    print("AGG lam0_typeAware=%.2f%%±%.2f | lam1=%.2f%%±%.2f | lam*=%.2f%%±%.2f" %
          (agg['lam0_typeAware']['mean'] * 100, agg['lam0_typeAware']['std'] * 100,
           agg['lam1']['mean'] * 100, agg['lam1']['std'] * 100,
           agg['lambda_star']['mean'] * 100, agg['lambda_star']['std'] * 100), flush=True)


if __name__ == '__main__':
    main()
