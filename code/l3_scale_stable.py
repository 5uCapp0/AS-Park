# -*- coding: utf-8 -*-
"""
    Stabilized large-fleet experiments (30 epochs + LR decay).
"""
import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import l3_online_dispatcher as lod
import l3_scale

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, 'data')

SEEDS_SCALE = [1024, 2048, 4096, 8192, 16384, 2026, 2027, 2028, 42, 7]
SIZES = [20, 50, 100]
N_EPOCH = 30
LR0 = 1e-2
STEP, GAMMA = 10, 0.5


def run_seed_scale_stable(seed, device, N, M, n_ep_train=600, n_epoch=N_EPOCH, n_ep_eval=150):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    frac = 0.35
    sigma2 = 1.0

    train = []
    for _ in range(n_ep_train):
        cost, ctypes, stypes, prio, dist = l3_scale.make_episode_grid(rng, N, M, frac)
        order = rng.permutation(N)
        labels = lod.online_oracle_labels(cost, N, M, order, rng)
        train.append((cost, ctypes, stypes, prio, labels, order, dist))

    ev = []
    ev_scenes_cost, ev_scenes_D = [], []
    for _ in range(n_ep_eval):
        cost, ctypes, stypes, prio, dist = l3_scale.make_episode_grid(rng, N, M, frac)
        order = rng.permutation(N)
        ev.append((cost, ctypes, stypes, prio, order, dist))
        ev_scenes_cost.append(cost)
        ev_scenes_D.append(np.tile(dist, (N, 1)).astype(float))

    model = lod.SoftEquivariantScorer().to(device)
    opt = optim.Adam(model.parameters(), lr=LR0)
    sched = optim.lr_scheduler.StepLR(opt, step_size=STEP, gamma=GAMMA)
    ce = nn.CrossEntropyLoss()

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
        sched.step()

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
        results[str(lam)] = {'gap': float(np.mean(gaps)), 'std': float(np.std(gaps))}
    gaps_star = []
    for cost, ctypes, stypes, prio, order, dist in ev:
        total, misses = lod.online_rollout(model, cost, ctypes, prio, stypes, dist, order, lam_star, device)
        _, _, oc = lod.hungarian(cost)
        gaps_star.append((total - oc) / (oc + 1e-9))
    results['lambda_star'] = {'lam': lam_star, 'gap': float(np.mean(gaps_star)),
                              'std': float(np.std(gaps_star))}
    results['_eps2'] = float(eps2)
    results['_lam_star'] = float(lam_star)
    return results, eps2, lam_star


def main():
    mode = sys.argv[1]
    if mode == 'job':
        N = int(sys.argv[2])
        seed = int(sys.argv[3])
        out_json = sys.argv[4]
        M = 4 * N
        device = ('cpu' if os.environ.get('L3_FORCE_CPU') == '1'
                  else ('cuda' if torch.cuda.is_available() else 'cpu'))
        res, eps2, lam = run_seed_scale_stable(seed, device, N, M)
        with open(out_json, 'w', encoding='utf-8') as f:
            json.dump({'N': N, 'seed': seed, 'res': res, 'eps2': eps2, 'lam_star': lam,
                       'stable': True, 'n_epoch': N_EPOCH, 'lr': LR0, 'step': STEP, 'gamma': GAMMA},
                      f, ensure_ascii=False)
        print('saved', out_json, 'N=%d seed=%d gap(lam*)=%.2f%%' % (N, seed, res['lambda_star']['gap'] * 100), flush=True)
    elif mode == 'merge':
        root = sys.argv[2]
        from scipy import stats as st
        for N in SIZES:
            per_seed = {}
            lamstars = []
            for sd in SEEDS_SCALE:
                p = os.path.join(root, 'scale_stable_%d_%d.json' % (N, sd))
                if not os.path.exists(p):
                    raise SystemExit('missing %s' % p)
                d = json.load(open(p, encoding='utf-8'))
                per_seed[str(sd)] = d['res']
                lamstars.append(d['lam_star'])
            lam_grid = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
            agg = {}
            for lam in lam_grid:
                gs = [per_seed[str(sd)][str(lam)]['gap'] for sd in SEEDS_SCALE]
                agg[str(lam)] = {'mean': float(np.mean(gs)), 'std': float(np.std(gs))}
            gs_star = [per_seed[str(sd)]['lambda_star']['gap'] for sd in SEEDS_SCALE]
            agg['lambda_star'] = {'mean': float(np.mean(gs_star)), 'std': float(np.std(gs_star))}
            agg['lam_star_mean'] = float(np.mean(lamstars))
            for name, other in [('vs_lam0', '0.0'), ('vs_lam1', '1.0')]:
                g0 = np.array([per_seed[str(sd)][other]['gap'] for sd in SEEDS_SCALE])
                t, p = st.ttest_ind(g0, np.array(gs_star), equal_var=False)
                agg[name] = {'t': float(t), 'p': float(p)}
            out = {'N': N, 'M': 4 * N, 'stable': True, 'seeds': len(SEEDS_SCALE),
                   'aggregate': agg, 'per_seed': per_seed}
            with open(os.path.join(root, 'scale_stable_N%d_result.json' % N), 'w', encoding='utf-8') as f:
                json.dump(out, f, ensure_ascii=False, indent=2)
            a = agg
            print('N=%d: gap(lam*)=%.2f%%±%.2f vs0 p=%.4f vs1 p=%.4f' % (
                N, a['lambda_star']['mean'] * 100, a['lambda_star']['std'] * 100,
                a['vs_lam0']['p'], a['vs_lam1']['p']))
        print('STABLE MERGE DONE')
    else:
        raise SystemExit('bad mode')


if __name__ == '__main__':
    main()
