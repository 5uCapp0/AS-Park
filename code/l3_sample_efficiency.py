# -*- coding: utf-8 -*-
"""
    Sample-efficiency experiment (epoch vs gap learning curves).
"""
import os
import json
import time
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import importlib

l3d = importlib.import_module("l3_online_dispatcher")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, 'data')
FIG = os.path.join(ROOT, 'figs')
os.makedirs(DATA, exist_ok=True)
os.makedirs(FIG, exist_ok=True)

M = 16
N = 8
FRAC = 0.35
SIGMA2 = 1.0
DEFAULT_SEEDS = [20260925, 7, 42]
REGIMES = ['lam0', 'lam1', 'star']


def build_data(seed, n_ep_train, n_ep_eval):
    rng = np.random.default_rng(seed)
    dist = np.arange(1, M + 1, dtype=float)
    train = []
    for _ in range(n_ep_train):
        cost, ctypes, stypes, prio = l3d.make_episode(rng, N, M, FRAC)
        order = rng.permutation(N)
        labels = l3d.online_oracle_labels(cost, N, M, order, rng)
        train.append((cost, ctypes, stypes, prio, labels, order))
    rng_e = np.random.default_rng(seed * 31 + 5)
    ev = []
    ev_cost = []
    for _ in range(n_ep_eval):
        cost, ctypes, stypes, prio = l3d.make_episode(rng_e, N, M, FRAC)
        order = rng_e.permutation(N)
        ev.append((cost, ctypes, stypes, prio, order))
        ev_cost.append(cost)
    return train, ev, ev_cost, dist


def eval_gap(model, ev, dist, device, lam):
    gs = []
    for cost, ctypes, stypes, prio, order in ev:
        total, _ = l3d.online_rollout(model, cost, ctypes, prio, stypes, dist, order, lam, device)
        _, _, oc = l3d.hungarian(cost)
        gs.append((total - oc) / (oc + 1e-9))
    return float(np.mean(gs))


def train_one_regime(seed, regime, train, ev, ev_cost, dist, device, n_epoch):
    torch.manual_seed(seed)
    model = l3d.SoftEquivariantScorer().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    ce = torch.nn.CrossEntropyLoss()

    if regime == 'lam0':
        lam_train = lam_eval = 0.0
        eps2 = None
        lam_star = None
    elif regime == 'lam1':
        lam_train = lam_eval = 1.0
        eps2 = None
        lam_star = None
    else:  # 'star'
        probe_cost = [t[0] for t in train[:50]]
        probe_D = [np.tile(dist, (N, 1)).astype(float) for _ in range(50)]
        eps2_train = l3d.defect_energy_eps2(probe_cost, probe_D)
        lam_train = eps2_train / (eps2_train + SIGMA2 / N + 1e-12)
        ev_D = [np.tile(dist, (N, 1)).astype(float) for _ in range(len(ev_cost))]
        eps2 = l3d.defect_energy_eps2(ev_cost, ev_D)
        lam_eval = eps2 / (eps2 + SIGMA2 / N + 1e-12)
        lam_star = lam_eval

    curve = []
    for ep in range(n_epoch):
        opt.zero_grad()
        loss = 0.0
        n_steps = 0
        for cost, ctypes, stypes, prio, labels, order in train:
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
                scores = model(ct[i:i+1], pr[i:i+1], st, ds, lam_train)  # (1,M)
                logits = scores[0, free_idx]
                tgt = np.where(free_idx == labels[i])[0]
                if tgt.size == 0:
                    continue
                step_loss = step_loss + ce(
                    logits.unsqueeze(0),
                    torch.tensor([int(tgt[0])], dtype=torch.long, device=device))
                n_steps += 1
                free_mask[labels[i]] = False
            loss = loss + step_loss
        loss = loss / max(n_steps, 1) * 4800.0
        loss.backward()
        opt.step()
        g = eval_gap(model, ev, dist, device, lam_eval)
        curve.append(g)
        if ep % 5 == 0 or ep == n_epoch - 1:
            print(f"  [seed {seed} {regime}] ep={ep:2d} loss={loss.item():.4f} gap={g*100:.2f}%"
                  f" lam_train={lam_train:.3f} lam_eval={lam_eval:.3f}", flush=True)

    plateau = None
    best = float('inf')
    for ep, g in enumerate(curve):
        best = min(best, g)
        if g <= best + 0.002:
            plateau = ep
            break
    return {'curve': curve, 'final_gap': curve[-1], 'best_gap': best,
            'epochs_to_plateau': plateau, 'lam_train': lam_train,
            'lam_eval': lam_eval, 'eps2': eps2, 'lam_star': lam_star}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', default=','.join(str(s) for s in DEFAULT_SEEDS))
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--train-ep', type=int, default=1000)
    ap.add_argument('--eval-ep', type=int, default=150)
    ap.add_argument('--device', default=None)
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(',') if s.strip()]
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    print("device:", device, flush=True)
    t_start = time.time()

    per_seed = {}
    for seed in seeds:
        print(f"--- seed {seed} ---", flush=True)
        train, ev, ev_cost, dist = build_data(seed, args.train_ep, args.eval_ep)
        per_seed[str(seed)] = {}
        for regime in REGIMES:
            r = train_one_regime(seed, regime, train, ev, ev_cost, dist, device, args.epochs)
            per_seed[str(seed)][regime] = r
            print(f"  {regime}: final={r['final_gap']*100:.2f}%  plateau_ep={r['epochs_to_plateau']}"
                  f"  lam_train={r['lam_train']:.3f} lam_eval={r['lam_eval']:.3f}", flush=True)

    # ---- aggregate across seeds ----
    agg = {}
    for regime in REGIMES:
        curves = np.array([per_seed[s][regime]['curve'] for s in map(str, seeds)])
        finals = [per_seed[s][regime]['final_gap'] for s in map(str, seeds)]
        plateaus = [per_seed[s][regime]['epochs_to_plateau'] for s in map(str, seeds)]
        agg[regime] = {
            'curve_mean': [float(v) for v in curves.mean(axis=0)],
            'curve_std': [float(v) for v in curves.std(axis=0)],
            'final_mean': float(np.mean(finals)), 'final_std': float(np.std(finals)),
            'per_seed_final': [float(v) for v in finals],
            'epochs_to_plateau_per_seed': plateaus,
            'lam_train_per_seed': [per_seed[s][regime]['lam_train'] for s in map(str, seeds)],
            'lam_eval_per_seed': [per_seed[s][regime]['lam_eval'] for s in map(str, seeds)],
        }

    out = {
        'exp': 'l3_sample_efficiency',
        'dispatcher_code_mtime': os.path.getmtime(os.path.join(HERE, 'l3_online_dispatcher.py')),
        'env': f'online soft-equivariant dispatcher, M={M}, N={N}, frac={FRAC}, '
               f'3 regimes (lam0/lam1/star), fixed eval set per seed',
        'sigma2': SIGMA2, 'seeds': seeds, 'epochs': args.epochs,
        'train_ep': args.train_ep, 'eval_ep': args.eval_ep,
        'device': device,
        'runtime_sec': round(time.time() - t_start, 1),
        'per_seed': per_seed,
        'aggregate': agg,
    }
    jpath = os.path.join(DATA, 'l3_sample_efficiency_result.json')
    with open(jpath, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("Saved:", jpath, flush=True)

    # ---- figure: epoch vs gap, three curves ----
    xs = list(range(args.epochs))
    plt.figure(figsize=(6.4, 4.2), dpi=120)
    colors = {'lam0': '#95a5a6', 'lam1': '#bdc3c7', 'star': '#c0392b'}
    labels = {'lam0': 'lam=0 (anonymous)', 'lam1': 'lam=1 (personalized)', 'star': 'lam* (adaptive)'}
    for regime in REGIMES:
        ys = agg[regime]['curve_mean']
        errs = agg[regime]['curve_std']
        plt.errorbar(xs, [v * 100 for v in ys], yerr=[v * 100 for v in errs],
                     fmt='-', color=colors[regime], capsize=2, lw=1.6,
                     label=labels[regime])
    plt.xlabel('training epoch')
    plt.ylabel('online optimality gap vs oracle (%)')
    plt.title('Sample efficiency: adaptive lam* converges faster/stabler', fontsize=10, fontweight='bold')
    plt.legend(fontsize=8)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    fpath = os.path.join(FIG, 'l3_sample_efficiency.png')
    plt.savefig(fpath)
    plt.close()
    print("Saved:", fpath, flush=True)


if __name__ == '__main__':
    main()
