# -*- coding: utf-8 -*-
"""
    A5: dispatcher ablation (four-stage vs end-to-end).
"""
import os, json, time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import l2_common as L
import l2_stats as S

FAM = L.FAMILY_B
P = 0.45
N_EPOCH_FULL = 20


def _prefetch(scenes):
    data = []
    for sc in scenes:
        cp = torch.tensor(sc['car_pos'], dtype=torch.float32)
        sp = torch.tensor(sc['spots'], dtype=torch.float32)
        ct = torch.tensor(sc['car_type'], dtype=torch.long)
        st = torch.tensor(sc['spot_type'], dtype=torch.long)
        r, c, _ = L.ps.oracle(sc['cost'])
        data.append((cp, sp, ct, st, torch.tensor(c, dtype=torch.long)))
    return data


def _epoch(policy, data, lam, params, ce):
    opt = optim.Adam(params, lr=L.LR)
    opt.zero_grad()
    loss = 0.0
    for cp, sp, ct, st, target in data:
        loss = loss + ce(policy(cp, sp, ct, st, lam), target)
    loss.backward(); opt.step()


def trunk_params(pol):
    return list(pol.phi.parameters()) + list(pol.psi.parameters()) + list(pol.score.parameters())


def res_params(pol):
    return list(pol.res.parameters())


def run():
    print("="*70); print(f"A5 dispatcher ablation family={FAM['name']} p={P}"); print("="*70, flush=True)
    e2e_gaps, four_gaps = [], []
    lam_stars = []
    for seed in L.SEEDS:
        rng = np.random.default_rng(seed)
        obstacles, W, H = L.build_family(FAM, rng)
        tr = L.gen_family(L.N_TRAIN, FAM, obstacles, P, rng)
        te = L.gen_family(L.N_TEST, FAM, obstacles, P, rng)
        data = _prefetch(tr)
        ce = nn.CrossEntropyLoss()
        gaps = {}
        for lam in L.LAM_GRID:
            pol = L.ps.ParkingPolicy()
            for _ in range(8):
                _epoch(pol, data, lam, list(pol.parameters()), ce)
            gaps[lam], _, _, _ = L.closed_loop_evaluate(pol, te, obstacles, lam, W, H)
        lam_star = min(gaps, key=gaps.get)
        lam_stars.append(lam_star)
        pol_e2e = L.ps.ParkingPolicy()
        for _ in range(N_EPOCH_FULL):
            _epoch(pol_e2e, data, lam_star, list(pol_e2e.parameters()), ce)
        g_e2e, _, _, _ = L.closed_loop_evaluate(pol_e2e, te, obstacles, lam_star, W, H)
        e2e_gaps.append(g_e2e)
        pol = L.ps.ParkingPolicy()
        for _ in range(8):
            _epoch(pol, data, 0.0, trunk_params(pol), ce)
        for _ in range(6):
            _epoch(pol, data, 0.1 * lam_star, res_params(pol), ce)
        for _ in range(6):
            _epoch(pol, data, lam_star, list(pol.parameters()), ce)
        g4, _, _, _ = L.closed_loop_evaluate(pol, te, obstacles, lam_star, W, H)
        four_gaps.append(g4)
        print(f"  seed={seed} λ*={lam_star:.2f} e2e={g_e2e:.4f} four-stage={g4:.4f}", flush=True)

    stat = S.compare_group('four_stage_vs_e2e', four_gaps, e2e_gaps)
    print(f"  four-stage mean={np.mean(four_gaps):.4f}  end-to-end mean={np.mean(e2e_gaps):.4f} "
          f"p={stat['p']:.3f} sig={stat['significant']}", flush=True)
    out = dict(script=os.path.basename(__file__), date='2026-10-01', seeds=L.SEEDS,
               family=FAM['name'], p=P, n_epoch=N_EPOCH_FULL,
               lambda_star_used=[float(x) for x in lam_stars],
               four_stage_gaps=four_gaps, e2e_gaps=e2e_gaps,
               four_stage_mean=float(np.mean(four_gaps)), e2e_mean=float(np.mean(e2e_gaps)),
               stat=stat)
    with open(os.path.join(L.DATA_DIR, 'l2_a5.json'), 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    with open(os.path.join(L.DATA_DIR, 'l2_a5.csv'), 'w', encoding='utf-8') as f:
        f.write('seed,four_stage,end_to_end\n')
        for s in range(len(L.SEEDS)):
            f.write(f"{L.SEEDS[s]},{four_gaps[s]:.5f},{e2e_gaps[s]:.5f}\n")
    fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=150)
    ax.bar(['four-stage', 'end-to-end'], [np.mean(four_gaps), np.mean(e2e_gaps)],
           yerr=[np.std(four_gaps, ddof=1), np.std(e2e_gaps, ddof=1)],
           capsize=5, color=['#2980b9', '#95a5a6'])
    ax.set_ylabel('closed-loop gap'); ax.set_title(f'A5: scheduler (p={P}, p_t={stat["p"]:.3f})', fontweight='bold')
    ax.grid(alpha=0.3, axis='y')
    fig.tight_layout()
    fig.savefig(os.path.join(L.FIG_DIR, 'l2_a5_scheduler.png')); plt.close(fig)
    print("[A5 done]", flush=True)


if __name__ == '__main__':
    run()
