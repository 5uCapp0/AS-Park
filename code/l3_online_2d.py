# -*- coding: utf-8 -*-
"""
    L3-Online 2D grid extension (defect-dominated regime).
"""
import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import l3_online_dispatcher as lod

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, 'data')
FIG = os.path.join(ROOT, 'figs')
os.makedirs(DATA, exist_ok=True)
os.makedirs(FIG, exist_ok=True)

SEEDS = lod.SEEDS
C_NORMAL, C_EV, C_LARGE = 0, 1, 2


def make_episode_2d(rng, N, M, frac_special=0.35, grid_h=4, grid_w=4):
    """4x4 grid lot: M = grid_h*grid_w stalls at integer coordinates; distance = Manhattan.
    Same type structure as the 1-D aisle make_episode."""
    spot_types = np.zeros(M, dtype=int)
    n_ev = int(round(M * frac_special / 2))
    n_large = int(round(M * frac_special / 2))
    perm = rng.permutation(M)
    spot_types[perm[:n_ev]] = C_EV
    spot_types[perm[n_ev:n_ev + n_large]] = C_LARGE
    car_types = rng.choice([C_NORMAL, C_EV, C_LARGE], size=N, p=[0.5, 0.3, 0.2])
    priority = rng.uniform(0.0, 1.0, size=N)

    rows, cols = np.divmod(np.arange(M), grid_w)          # stall coordinates
    # distance to a reference corner (1,1) -> range 0..(h+w-2); +1 to keep >=1
    dist = (rows + cols).astype(float) + 1.0

    cost = np.zeros((N, M), dtype=float)
    for i in range(N):
        ct = car_types[i]
        for j in range(M):
            st = spot_types[j]
            c = dist[j]
            if ct == C_EV:
                c += {C_EV: 0.0, C_NORMAL: 2.0, C_LARGE: 1.0}[st]
            elif ct == C_LARGE:
                c += {C_LARGE: 0.0, C_NORMAL: 3.0, C_EV: 3.0}[st]
            cost[i, j] = c
    return cost, car_types, spot_types, priority, dist


def run_seed_2d(seed, device, n_ep_train=600, n_epoch=50, n_ep_eval=150):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    N, M, frac = 8, 16, 0.35

    train = []
    for _ in range(n_ep_train):
        cost, ctypes, stypes, prio, dist = make_episode_2d(rng, N, M, frac)
        order = rng.permutation(N)
        labels = lod.online_oracle_labels(cost, N, M, order, rng)
        train.append((cost, ctypes, stypes, prio, labels, order, dist))

    ev = []
    ev_scenes_cost, ev_scenes_D = [], []
    for _ in range(n_ep_eval):
        cost, ctypes, stypes, prio, dist = make_episode_2d(rng, N, M, frac)
        order = rng.permutation(N)
        ev.append((cost, ctypes, stypes, prio, order, dist))
        ev_scenes_cost.append(cost)
        ev_scenes_D.append(np.tile(dist, (N, 1)).astype(float))

    model = lod.SoftEquivariantScorer().to(device)
    opt = optim.Adam(model.parameters(), lr=1e-2)
    ce = nn.CrossEntropyLoss()

    sigma2 = 1.0
    probe_cost = [t[0] for t in train[:50]]
    probe_D = [np.tile(t[6], (N, 1)).astype(float) for t in train[:50]]
    eps2_train = lod.defect_energy_eps2(probe_cost, probe_D)
    lam_star_train = eps2_train / (eps2_train + sigma2 / N + 1e-12)

    for ep in range(n_epoch):
        opt.zero_grad()
        loss = 0.0
        n_steps = 0
        for cost, ctypes, stypes, prio, labels, order, dist in train:
            ct = torch.tensor(ctypes, dtype=torch.long, device=device)
            pr = torch.tensor(prio, dtype=torch.float32, device=device)
            st = torch.tensor(stypes, dtype=torch.long, device=device)
            ds = torch.tensor(dist, dtype=torch.float32, device=device)
            free_mask = np.ones(M, dtype=bool)
            step_loss = 0.0
            for k in range(N):
                i = int(order[k])
                free_idx = np.where(free_mask)[0]
                if len(free_idx) == 0:
                    break
                scores = model(ct[i:i+1], pr[i:i+1], st, ds, lam_star_train)
                logits = scores[0, free_idx]
                tgt = np.where(free_idx == labels[i])[0]
                if tgt.size == 0:
                    continue
                step_loss = step_loss + ce(logits.unsqueeze(0), torch.tensor([int(tgt[0])], dtype=torch.long, device=device))
                n_steps += 1
                free_mask[labels[i]] = False
            loss = loss + step_loss
        loss = loss / max(n_steps, 1) * 4800.0
        loss.backward()
        opt.step()

    eps2 = lod.defect_energy_eps2(ev_scenes_cost, ev_scenes_D)
    lam_star = eps2 / (eps2 + sigma2 / N + 1e-12)

    lam_grid = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    results = {}
    for lam in lam_grid:
        gaps = []
        for cost, ctypes, stypes, prio, order, dist in ev:
            total, misses = lod.online_rollout(model, cost, ctypes, prio, stypes, dist, order, lam, device)
            _, _, oc = lod.hungarian(cost)
            gaps.append((total - oc) / (oc + 1e-9))
        results[lam] = {'gap': float(np.mean(gaps)), 'std': float(np.std(gaps))}
    gaps_star = []
    for cost, ctypes, stypes, prio, order, dist in ev:
        total, misses = lod.online_rollout(model, cost, ctypes, prio, stypes, dist, order, lam_star, device)
        _, _, oc = lod.hungarian(cost)
        gaps_star.append((total - oc) / (oc + 1e-9))
    results['lambda_star'] = {'lam': lam_star, 'gap': float(np.mean(gaps_star)),
                              'std': float(np.std(gaps_star))}
    return results, eps2, lam_star


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("device:", device, flush=True)
    all_results = {}
    eps2s, lamstars = [], []
    for seed in SEEDS:
        print(f"--- seed {seed} ---", flush=True)
        res, eps2, lam_star = run_seed_2d(seed, device)
        all_results[str(seed)] = res
        eps2s.append(eps2); lamstars.append(lam_star)
        print(f"  eps2={eps2:.4f} lam*={lam_star:.4f}", flush=True)

    lam_grid = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    agg = {}
    for lam in lam_grid:
        gs = [all_results[str(s)][lam]['gap'] for s in SEEDS]
        agg[lam] = {'mean': float(np.mean(gs)), 'std': float(np.std(gs))}
    gs_star = [all_results[str(s)]['lambda_star']['gap'] for s in SEEDS]
    agg['lambda_star'] = {'mean': float(np.mean(gs_star)), 'std': float(np.std(gs_star))}
    lam_star_mean = float(np.mean(lamstars))
    agg['lam_star_mean'] = lam_star_mean
    agg['eps2_mean'] = float(np.mean(eps2s))

    from scipy import stats as st
    for name, other in [('vs_lam0', 0.0), ('vs_lam1', 1.0)]:
        g0 = np.array([all_results[str(s)][other]['gap'] for s in SEEDS])
        g1 = np.array(gs_star)
        t, p = st.ttest_ind(g0, g1, equal_var=False)
        agg[name] = {'t': float(t), 'p': float(p)}
        print(f"  λ* vs λ={other}: t={t:.3f} p={p:.4f}", flush=True)

    out = {'env': '2D 4x4 grid lot, M=16 N=8 frac=0.35, Manhattan dist, online arrivals, 5 seeds, sigma2=1.0',
           'lam_grid': lam_grid, 'lambda_star_mean': lam_star_mean,
           'eps2_mean': agg['eps2_mean'], 'per_seed': all_results,
           'aggregate': agg, 'welch': {k: v for k, v in agg.items() if isinstance(k, str) and k.startswith('vs_')}}
    with open(os.path.join(DATA, 'l3_online_2d_result.json'), 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("Saved json:", os.path.join(DATA, 'l3_online_2d_result.json'), flush=True)

    # figures
    xs = lam_grid + [lam_star_mean]
    ys = [agg[l]['mean'] * 100 for l in lam_grid] + [agg['lambda_star']['mean'] * 100]
    errs = [agg[l]['std'] * 100 for l in lam_grid] + [agg['lambda_star']['std'] * 100]
    plt.figure(figsize=(6.2, 3.8), dpi=120)
    plt.errorbar(xs, ys, yerr=errs, fmt='o-', color='#2c3e50', capsize=4, label='online gap (mean±std)')
    plt.axvline(lam_star_mean, color='#c0392b', ls='--', lw=1.2, label=f'λ*={lam_star_mean:.2f}')
    plt.xlabel('symmetry-injection strength λ'); plt.ylabel('optimality gap vs oracle (%)')
    plt.title('2D grid lot: soft-equivariant online dispatcher, λ tunes the gap', fontsize=10, fontweight='bold')
    plt.legend(fontsize=8); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(FIG, 'l3_2d_lambda_scan.png')); plt.close()

    names = ['λ=0 (anonymous)', 'λ=1 (personalized)', f'λ*={lam_star_mean:.2f} (adaptive)']
    vals = [agg[0.0]['mean'] * 100, agg[1.0]['mean'] * 100, agg['lambda_star']['mean'] * 100]
    errs2 = [agg[0.0]['std'] * 100, agg[1.0]['std'] * 100, agg['lambda_star']['std'] * 100]
    plt.figure(figsize=(6.2, 3.8), dpi=120)
    plt.bar(names, vals, yerr=errs2, capsize=5, color=['#95a5a6', '#bdc3c7', '#c0392b'])
    plt.ylabel('optimality gap vs oracle (%)')
    plt.title('2D grid lot: adaptive λ* vs extremes (5 seeds, mean±std)', fontsize=10, fontweight='bold')
    plt.grid(alpha=0.3, axis='y'); plt.tight_layout()
    plt.savefig(os.path.join(FIG, 'l3_2d_compare.png')); plt.close()
    print("Figures saved.", flush=True)


if __name__ == '__main__':
    main()
