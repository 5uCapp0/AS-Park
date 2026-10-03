# -*- coding: utf-8 -*-
"""
    Fleet-size (N) generalization experiment.
"""
import os
import json
import time
import argparse
from math import erf
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
FRAC = 0.35
SIGMA2 = 1.0
N_TRAIN = 8
DEFAULT_SEEDS = [20260925, 7, 42]


def train_scorer_n8(seed, device, n_ep_train, n_epoch):
    rng = np.random.default_rng(seed)
    dist = np.arange(1, M + 1, dtype=float)
    train = []
    for _ in range(n_ep_train):
        cost, ctypes, stypes, prio = l3d.make_episode(rng, N_TRAIN, M, FRAC)
        order = rng.permutation(N_TRAIN)
        labels = l3d.online_oracle_labels(cost, N_TRAIN, M, order, rng)
        train.append((cost, ctypes, stypes, prio, labels, order))

    model = l3d.SoftEquivariantScorer().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    ce = torch.nn.CrossEntropyLoss()

    probe_cost = [t[0] for t in train[:50]]
    probe_D = [np.tile(dist, (N_TRAIN, 1)).astype(float) for _ in range(50)]
    eps2_train = l3d.defect_energy_eps2(probe_cost, probe_D)
    lam_train = eps2_train / (eps2_train + SIGMA2 / N_TRAIN + 1e-12)

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
            for k in range(N_TRAIN):
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
        if ep % 5 == 0 or ep == n_epoch - 1:
            print(f"  ep={ep:2d} loss={loss.item():.4f} n_steps={n_steps} lam_train={lam_train:.4f}", flush=True)
    return model


def eval_at_N(model, device, seed, N, n_ep_eval):
    rng = np.random.default_rng(seed * 1000 + N * 10 + 7)
    dist = np.arange(1, M + 1, dtype=float)
    ev = []
    scenes_cost = []
    for _ in range(n_ep_eval):
        cost, ctypes, stypes, prio = l3d.make_episode(rng, N, M, FRAC)
        order = rng.permutation(N)
        ev.append((cost, ctypes, stypes, prio, order))
        scenes_cost.append(cost)
    scenes_D = [np.tile(dist, (N, 1)).astype(float) for _ in range(n_ep_eval)]
    eps2_N = l3d.defect_energy_eps2(scenes_cost, scenes_D)
    lam_star_N = eps2_N / (eps2_N + SIGMA2 / N + 1e-12)

    def rollout_gaps(lam):
        gs, ms = [], []
        for cost, ctypes, stypes, prio, order in ev:
            total, misses = l3d.online_rollout(model, cost, ctypes, prio, stypes, dist, order, lam, device)
            _, _, oc = l3d.hungarian(cost)
            gs.append((total - oc) / (oc + 1e-9))
            ms.append(misses)
        return np.array(gs), np.mean(ms)

    g0, m0 = rollout_gaps(0.0)
    g1, m1 = rollout_gaps(1.0)
    gs, ms = rollout_gaps(lam_star_N)

    def paired_t(a, b):
        d = a - b
        n = d.size
        sd = d.std(ddof=1)
        t = d.mean() / (sd / np.sqrt(n) + 1e-12)
        p = 2.0 * (1.0 - 0.5 * (1.0 + erf(np.abs(t) / np.sqrt(2.0))))
        return float(t), float(p)

    t0, p0 = paired_t(gs, g0)
    t1, p1 = paired_t(gs, g1)
    return {
        'eps2': float(eps2_N), 'lam_star': float(lam_star_N),
        'gap_lam0': float(g0.mean()), 'std_lam0': float(g0.std(ddof=1)),
        'gap_lam1': float(g1.mean()), 'std_lam1': float(g1.std(ddof=1)),
        'gap_star': float(gs.mean()), 'std_star': float(gs.std(ddof=1)),
        'misses_lam0': float(m0), 'misses_lam1': float(m1), 'misses_star': float(ms),
        'paired_t_star_vs_lam0': t0, 'paired_p_star_vs_lam0': p0,
        'paired_t_star_vs_lam1': t1, 'paired_p_star_vs_lam1': p1,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', default=','.join(str(s) for s in DEFAULT_SEEDS))
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--train-ep', type=int, default=1000)
    ap.add_argument('--eval-ep', type=int, default=300)
    ap.add_argument('--device', default=None)
    ap.add_argument('--n-list', default='8,12,16')
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(',') if s.strip()]
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    print("device:", device, flush=True)
    t_start = time.time()

    Ns = [int(x) for x in args.n_list.split(',') if x.strip()]
    per_seed = {}
    for seed in seeds:
        print(f"--- seed {seed}: train N={N_TRAIN} ---", flush=True)
        model = train_scorer_n8(seed, device, args.train_ep, args.epochs)
        per_seed[str(seed)] = {}
        for N in Ns:
            r = eval_at_N(model, device, seed, N, args.eval_ep)
            per_seed[str(seed)][str(N)] = r
            print(f"  N={N:2d}: lam0={r['gap_lam0']*100:6.2f}%+-{r['std_lam0']*100:5.2f}"
                  f"  lam1={r['gap_lam1']*100:6.2f}%+-{r['std_lam1']*100:5.2f}"
                  f"  lam*={r['gap_star']*100:6.2f}%+-{r['std_star']*100:5.2f}"
                  f"  (eps2={r['eps2']:.4f}, lam*={r['lam_star']:.4f})", flush=True)

    # ---- aggregate across seeds ----
    agg = {}
    for N in Ns:
        key = str(N)
        d = {}
        for k, tag in [('lam0', 'gap_lam0'), ('lam1', 'gap_lam1'), ('star', 'gap_star')]:
            vals = [per_seed[s][key][tag] for s in map(str, seeds)]
            d[k] = {'mean': float(np.mean(vals)), 'std': float(np.std(vals)),
                    'per_seed': [float(v) for v in vals]}
        d['eps2_mean'] = float(np.mean([per_seed[s][key]['eps2'] for s in map(str, seeds)]))
        d['lam_star_mean'] = float(np.mean([per_seed[s][key]['lam_star'] for s in map(str, seeds)]))
        d['n_seeds_p_star_vs_lam0_lt_0.05'] = sum(
            1 for s in map(str, seeds) if per_seed[s][key]['paired_p_star_vs_lam0'] < 0.05)
        d['n_seeds_p_star_vs_lam1_lt_0.05'] = sum(
            1 for s in map(str, seeds) if per_seed[s][key]['paired_p_star_vs_lam1'] < 0.05)
        agg[key] = d

    out = {
        'exp': 'l3_generalization',
        'dispatcher_code_mtime': os.path.getmtime(os.path.join(HERE, 'l3_online_dispatcher.py')),
        'env': f'online soft-equivariant dispatcher, M={M}, frac={FRAC}, train N={N_TRAIN}, eval N={Ns}',
        'sigma2': SIGMA2, 'seeds': seeds, 'ns': Ns,
        'device': device,
        'runtime_sec': round(time.time() - t_start, 1),
        'per_seed': per_seed,
        'aggregate': agg,
    }
    jpath = os.path.join(DATA, 'l3_generalization_n' + '_'.join(str(x) for x in Ns) + '.json')
    with open(jpath, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("Saved:", jpath, flush=True)

    # ---- figure: N vs gap, three lines ----
    xs = Ns
    plt.figure(figsize=(6.4, 4.2), dpi=120)
    for k, tag, color, mk in [('lam0', 'lam=0 (anonymous)', '#95a5a6', 'o'),
                              ('lam1', 'lam=1 (personalized)', '#bdc3c7', 's'),
                              ('star', 'lam* (adaptive)', '#c0392b', '^')]:
        ys = [agg[str(n)][k]['mean'] * 100 for n in Ns]
        errs = [agg[str(n)][k]['std'] * 100 for n in Ns]
        plt.errorbar(xs, ys, yerr=errs, fmt=mk + '-', color=color, capsize=4,
                     label=tag, markersize=5)
    plt.xlabel('number of cars N (trained at N=8)')
    plt.ylabel('online optimality gap vs oracle (%)')
    plt.title('N generalization: adaptive lam* transfers to unseen scales', fontsize=10, fontweight='bold')
    plt.xticks(Ns, [str(n) for n in Ns])
    plt.legend(fontsize=8)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    fpath = os.path.join(FIG, 'l3_generalization.png')
    plt.savefig(fpath)
    plt.close()
    print("Saved:", fpath, flush=True)


if __name__ == '__main__':
    main()
