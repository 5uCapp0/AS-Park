# -*- coding: utf-8 -*-
"""
    sigma^2 sensitivity scan of the closed form.
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

import l3_online_dispatcher as lod  # reuse env / model / oracle / defect estimator

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, 'data')
FIG = os.path.join(ROOT, 'figs')
os.makedirs(DATA, exist_ok=True)
os.makedirs(FIG, exist_ok=True)

SEEDS = lod.SEEDS
SIGMA2_GRID = [0.25, 0.5, 1.0, 2.0]
MAIN10_JSON = os.path.join(DATA, 'l3_main_10seed_result.json')


def run_seed_sigma(seed, device, sigma2, n_ep_train=600, n_epoch=50, n_ep_eval=150, use_types=False):
    """Same pipeline as l3_online_dispatcher.run_seed but with sigma2 as a parameter."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    N, M, frac = 8, 16, 0.35
    dist = np.arange(1, M + 1, dtype=float)

    train = []
    for _ in range(n_ep_train):
        cost, ctypes, stypes, prio = lod.make_episode(rng, N, M, frac)
        order = rng.permutation(N)
        labels = lod.online_oracle_labels(cost, N, M, order, rng)
        train.append((cost, ctypes, stypes, prio, labels, order))

    ev = []
    ev_scenes_cost, ev_scenes_D = [], []
    for _ in range(n_ep_eval):
        cost, ctypes, stypes, prio = lod.make_episode(rng, N, M, frac)
        order = rng.permutation(N)
        ev.append((cost, ctypes, stypes, prio, order))
        ev_scenes_cost.append(cost)
        ev_scenes_D.append(np.tile(dist, (N, 1)).astype(float))

    model = lod.SoftEquivariantScorer(use_types=use_types).to(device)
    opt = optim.Adam(model.parameters(), lr=1e-2)
    ce = nn.CrossEntropyLoss()

    # train at lambda*_train estimated from a 50-scene probe (same as main script)
    probe_cost = [t[0] for t in train[:50]]
    probe_D = [np.tile(dist, (N, 1)).astype(float) for _ in range(50)]
    eps2_train = lod.defect_energy_eps2(probe_cost, probe_D)
    lam_star_train = eps2_train / (eps2_train + sigma2 / N + 1e-12)

    for ep in range(n_epoch):
        opt.zero_grad()
        loss = 0.0
        n_steps = 0
        for cost, ctypes, stypes, prio, labels, order in train:
            lam_train = lam_star_train
            ct = torch.tensor(ctypes, dtype=torch.long, device=device)
            pr = torch.tensor(prio, dtype=torch.float32, device=device)
            st = torch.tensor(stypes, dtype=torch.long, device=device)
            ds = torch.tensor(dist, dtype=torch.float32, device=device)
            assigned = set(); free_mask = np.ones(M, dtype=bool)
            step_loss = 0.0
            for k in range(N):
                i = int(order[k])
                free_idx = np.where(free_mask)[0]
                if len(free_idx) == 0:
                    break
                scores = model(ct[i:i+1], pr[i:i+1], st, ds, lam_train)
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

    # defect estimator on eval
    eps2 = lod.defect_energy_eps2(ev_scenes_cost, ev_scenes_D)
    lam_star = eps2 / (eps2 + sigma2 / N + 1e-12)

    lam_grid = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    results = {}
    for lam in lam_grid:
        gaps = []
        for cost, ctypes, stypes, prio, order in ev:
            total, misses = lod.online_rollout(model, cost, ctypes, prio, stypes, dist, order, lam, device)
            _, _, oc = lod.hungarian(cost)
            gaps.append((total - oc) / (oc + 1e-9))
        results[lam] = {'gap': float(np.mean(gaps)), 'std': float(np.std(gaps))}
    gaps_star = []
    for cost, ctypes, stypes, prio, order in ev:
        total, misses = lod.online_rollout(model, cost, ctypes, prio, stypes, dist, order, lam_star, device)
        _, _, oc = lod.hungarian(cost)
        gaps_star.append((total - oc) / (oc + 1e-9))
    results['lambda_star'] = {'lam': lam_star, 'gap': float(np.mean(gaps_star)),
                              'std': float(np.std(gaps_star))}
    results['_eps2'] = float(eps2)
    results['_lam_star'] = float(lam_star)
    return results, eps2, lam_star


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("device:", device, flush=True)
    json_path = os.path.join(DATA, 'l3_sigma_sensitivity_result.json')
    out = {}
    if os.path.exists(json_path):
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                out = json.load(f)
            print("resuming; found entries:", list(out.keys()), flush=True)
        except Exception as e:
            print("resume failed, fresh start:", e, flush=True)
            out = {}
    for sigma2 in SIGMA2_GRID:
        print(f"===== sigma2 = {sigma2} =====", flush=True)
        per_seed = out.get(str(sigma2), {}).get('per_seed', {})
        seeds_used = SEEDS
        if sigma2 == 1.0 and os.path.exists(MAIN10_JSON):
            with open(MAIN10_JSON, 'r', encoding='utf-8') as f:
                mj = json.load(f)
            if mj.get('per_seed'):
                per_seed = {str(s): mj['per_seed'][str(s)] for s in mj['seeds']}
                seeds_used = mj['seeds']
                n = len(seeds_used)
                print(f"  sigma2=1.0: reusing main-experiment 10-seed results ({n} seeds)", flush=True)
        lamstars, eps2s = [], []
        for seed in seeds_used:
            if str(seed) in per_seed:
                print(f"  seed {seed} already done, skipping", flush=True)
                res = per_seed[str(seed)]
            else:
                print(f"  seed {seed} ...", flush=True)
                res, eps2, lam_star = run_seed_sigma(seed, device, sigma2)
                per_seed[str(seed)] = res
                out.setdefault(str(sigma2), {})['per_seed'] = per_seed
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(out, f, ensure_ascii=False, indent=2)
                print(f"    eps2={eps2:.4f} lam*={lam_star:.4f} gap(lam*)={res['lambda_star']['gap']*100:.2f}%", flush=True)
            lamstars.append(per_seed[str(seed)]['_lam_star'])
            eps2s.append(per_seed[str(seed)]['_eps2'])

        lam_grid = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        agg = {}
        for lam in lam_grid:
            gs = [per_seed[str(s)][lam]['gap'] for s in seeds_used]
            agg[lam] = {'mean': float(np.mean(gs)), 'std': float(np.std(gs))}
        gs_star = [per_seed[str(s)]['lambda_star']['gap'] for s in seeds_used]
        agg['lambda_star'] = {'mean': float(np.mean(gs_star)), 'std': float(np.std(gs_star))}
        lam_star_mean = float(np.mean(lamstars))
        agg['lam_star_mean'] = lam_star_mean
        agg['eps2_mean'] = float(np.mean(eps2s))

        from scipy import stats as st
        for name, other in [('vs_lam0', 0.0), ('vs_lam1', 1.0)]:
            g0 = np.array([per_seed[str(s)][other]['gap'] for s in seeds_used])
            g1 = np.array(gs_star)
            t, p = st.ttest_ind(g0, g1, equal_var=False)
            agg[name] = {'t': float(t), 'p': float(p)}

        out[str(sigma2)] = {'per_seed': per_seed, 'aggregate': agg}
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print(f"  => lam*_mean={lam_star_mean:.4f} eps2_mean={float(np.mean(eps2s)):.4f} "
              f"gap(lam*)={agg['lambda_star']['mean']*100:.2f}% "
              f"vs0 p={agg['vs_lam0']['p']:.4f} vs1 p={agg['vs_lam1']['p']:.4f}", flush=True)

    print("All sigma2 cells done; json already persisted incrementally.", flush=True)

    # ---- figure: two panels ----
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), dpi=120)
    xs = SIGMA2_GRID
    lam_stars = [out[str(s)]['aggregate']['lam_star_mean'] for s in xs]
    eps2m = float(np.mean([out[str(s)]['aggregate']['eps2_mean'] for s in xs]))
    # theoretical curve lambda* = eps2/(eps2 + sigma2/N), N=8
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
        lam_grid = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        xs2 = lam_grid + [agg['lam_star_mean']]
        ys2 = [agg[l]['mean'] * 100 for l in lam_grid] + [agg['lambda_star']['mean'] * 100]
        errs2 = [agg[l]['std'] * 100 for l in lam_grid] + [agg['lambda_star']['std'] * 100]
        axes[1].errorbar(xs2, ys2, yerr=errs2, fmt='o-', color=colors[i], capsize=3, lw=1.3,
                         label=f'σ²={s}')
        axes[1].scatter([agg['lam_star_mean']], [agg['lambda_star']['mean'] * 100], s=45,
                        facecolor='none', edgecolor=colors[i], zorder=5)
    axes[1].set_xlabel('λ'); axes[1].set_ylabel('gap vs oracle (%)')
    axes[1].set_title('U-shaped troughs: λ* stays interior at every σ²', fontsize=10, fontweight='bold')
    axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, 'l3_sigma_sensitivity.png'))
    print("Saved figure.", flush=True)


if __name__ == '__main__':
    main()
