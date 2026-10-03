# -*- coding: utf-8 -*-
"""
    Multi-initialization stability check for the L3 dispatcher.
"""
import os
import sys
import json
import numpy as np
import torch

import l3_online_dispatcher as lod

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, 'data')

SEEDS = [20260925, 7, 42, 123, 2024]
N_INITS = 3
LAM_GRID = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]


def run_seed_init(seed, init_idx, device):
    """Run the main pipeline but re-seed torch with seed*1000+init_idx so that
    numpy (data) is fixed while the model initialization varies."""
    torch.manual_seed(seed * 1000 + init_idx)
    rng = np.random.default_rng(seed)
    N, M, frac = 8, 16, 0.35
    dist = np.arange(1, M + 1, dtype=float)
    sigma2 = 1.0

    train = []
    for _ in range(600):
        cost, ctypes, stypes, prio = lod.make_episode(rng, N, M, frac)
        order = rng.permutation(N)
        labels = lod.online_oracle_labels(cost, N, M, order, rng)
        train.append((cost, ctypes, stypes, prio, labels, order))

    ev = []
    ev_scenes_cost, ev_scenes_D = [], []
    for _ in range(150):
        cost, ctypes, stypes, prio = lod.make_episode(rng, N, M, frac)
        order = rng.permutation(N)
        ev.append((cost, ctypes, stypes, prio, order))
        ev_scenes_cost.append(cost)
        ev_scenes_D.append(np.tile(dist, (N, 1)).astype(float))

    import torch.nn as nn
    import torch.optim as optim
    model = lod.SoftEquivariantScorer().to(device)
    opt = optim.Adam(model.parameters(), lr=1e-2)
    ce = nn.CrossEntropyLoss()

    probe_cost = [t[0] for t in train[:50]]
    probe_D = [np.tile(dist, (N, 1)).astype(float) for _ in range(50)]
    eps2_train = lod.defect_energy_eps2(probe_cost, probe_D)
    lam_star_train = eps2_train / (eps2_train + sigma2 / N + 1e-12)

    for ep in range(15):
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
                scores = model(ct[i:i + 1], pr[i:i + 1], st, ds, lam_star_train)
                logits = scores[0, free_idx]
                tgt = np.where(free_idx == labels[i])[0]
                if tgt.size == 0:
                    continue
                step_loss = step_loss + ce(logits.unsqueeze(0),
                                           torch.tensor([int(tgt[0])], dtype=torch.long, device=device))
                n_steps += 1
                free_mask[labels[i]] = False
            loss = loss + step_loss
        loss = loss / max(n_steps, 1) * 4800.0
        loss.backward()
        opt.step()

    eps2 = lod.defect_energy_eps2(ev_scenes_cost, ev_scenes_D)
    lam_star = eps2 / (eps2 + sigma2 / N + 1e-12)

    results = {}
    for lam in LAM_GRID:
        gaps = []
        for cost, ctypes, stypes, prio, order in ev:
            total, misses = lod.online_rollout(model, cost, ctypes, prio, stypes, dist, order, lam, device)
            _, _, oc = lod.hungarian(cost)
            gaps.append((total - oc) / (oc + 1e-9))
        results[str(lam)] = float(np.mean(gaps))
    gaps_star = []
    for cost, ctypes, stypes, prio, order in ev:
        total, misses = lod.online_rollout(model, cost, ctypes, prio, stypes, dist, order, lam_star, device)
        _, _, oc = lod.hungarian(cost)
        gaps_star.append((total - oc) / (oc + 1e-9))
    results['lambda_star'] = float(np.mean(gaps_star))
    return results, eps2, lam_star


def main():
    mode = sys.argv[1]
    if mode == 'job':
        seed = int(sys.argv[2])
        init_idx = int(sys.argv[3])
        out_json = sys.argv[4]
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        res, eps2, lam = run_seed_init(seed, init_idx, device)
        with open(out_json, 'w', encoding='utf-8') as f:
            json.dump({'seed': seed, 'init': init_idx, 'res': res, 'eps2': eps2, 'lam_star': lam},
                      f, ensure_ascii=False)
        print('saved', out_json, flush=True)
    elif mode == 'merge':
        root = sys.argv[2]
        out_json = sys.argv[3]
        from scipy import stats as st
        per_seed = {}
        for sd in SEEDS:
            gaps = {str(lam): [] for lam in LAM_GRID}
            gaps['lambda_star'] = []
            lamstars, eps2s = [], []
            for init in range(N_INITS):
                p = os.path.join(root, 'multiinit_%d_%d.json' % (sd, init))
                if not os.path.exists(p):
                    raise SystemExit('missing %s' % p)
                d = json.load(open(p, encoding='utf-8'))
                for lam in LAM_GRID:
                    gaps[str(lam)].append(d['res'][str(lam)])
                gaps['lambda_star'].append(d['res']['lambda_star'])
                lamstars.append(d['lam_star']); eps2s.append(d['eps2'])
            per_seed[str(sd)] = {lam: float(np.mean(v)) for lam, v in gaps.items()}
            per_seed[str(sd)]['lambda_star_lam'] = float(np.mean(lamstars))
        agg = {}
        for lam in LAM_GRID + ['lambda_star']:
            gs = [per_seed[str(sd)][lam] for sd in SEEDS]
            agg[lam] = {'mean': float(np.mean(gs)), 'std': float(np.std(gs))}
        agg['lam_star_mean'] = float(np.mean([per_seed[str(sd)]['lambda_star_lam'] for sd in SEEDS]))
        for name, other in [('vs_lam0', '0.0'), ('vs_lam1', '1.0')]:
            g0 = np.array([per_seed[str(sd)][other] for sd in SEEDS])
            g1 = np.array([per_seed[str(sd)]['lambda_star'] for sd in SEEDS])
            t, p = st.ttest_ind(g0, g1, equal_var=False)
            agg[name] = {'t': float(t), 'p': float(p)}
        with open(out_json, 'w', encoding='utf-8') as f:
            json.dump({'n_inits': N_INITS, 'aggregate': agg, 'per_seed': per_seed}, f, ensure_ascii=False, indent=2)
        a = agg
        print('multiinit merged: gap(lam0)=%.2f%% gap(lam*)=%.2f%% gap(lam1)=%.2f%% vs0 p=%.4f vs1 p=%.4f' % (
            a['0.0']['mean'] * 100, a['lambda_star']['mean'] * 100, a['1.0']['mean'] * 100,
            a['vs_lam0']['p'], a['vs_lam1']['p']))
    else:
        raise SystemExit('bad mode')


if __name__ == '__main__':
    main()
