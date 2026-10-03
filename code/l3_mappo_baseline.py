# -*- coding: utf-8 -*-
"""
    Parameter-shared MAPPO baseline (RL).
"""
import os
import json
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
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

M, N, FRAC = 16, 8, 0.35
DEFAULT_SEEDS = [20260925, 7, 42]
HID = 128
GAMMA, GAE_LAM, CLIP, LR, PPO_EPOCHS, ENT_COEF = 1.0, 0.95, 0.2, 3e-4, 4, 0.01
MB_SIZE = 256


# ---------------------------------------------------------------- obs helpers
def make_state_actor(car_type, spot_types, dist, free_mask, step):
    per = np.stack([dist, free_mask.astype(float)], axis=1)          # (M, 2)
    glo = np.array([step / N, free_mask.sum() / M], dtype=float)
    return np.concatenate([per.reshape(-1), glo])


def make_state_critic(car_type, priority, spot_types, dist, free_mask, step):
    car = np.zeros(l3d.N_TYPE + 1, dtype=float)
    car[car_type] = 1.0
    car[l3d.N_TYPE] = priority
    st = np.zeros((M, l3d.N_TYPE))
    st[np.arange(M), spot_types] = 1.0
    per = np.concatenate([dist[:, None], st, free_mask[:, None].astype(float)], axis=1)  # (M, 1+3+1)
    glo = np.array([step / N, free_mask.sum() / M], dtype=float)
    return np.concatenate([car, per.reshape(-1), glo])


# ---------------------------------------------------------------- networks
class Actor(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, HID), nn.Tanh(),
                                 nn.Linear(HID, HID), nn.Tanh(),
                                 nn.Linear(HID, M))
    def forward(self, x, free_mask):
        logits = self.net(x)
        logits = logits.masked_fill(~free_mask, -1e9)
        return logits


class Critic(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, HID), nn.Tanh(),
                                 nn.Linear(HID, HID), nn.Tanh(),
                                 nn.Linear(HID, 1))
    def forward(self, x):
        return self.net(x).squeeze(-1)


# ---------------------------------------------------------------- rollout
def collect_episode(rng, actor, critic, device):
    cost, ctypes, stypes, prio = l3d.make_episode(rng, N, M, FRAC)
    order = rng.permutation(N)
    dist = np.arange(1, M + 1, dtype=float)
    free = np.ones(M, dtype=bool)
    traj = []
    ret = 0.0
    for k in range(N):
        i = int(order[k])
        ct_i = int(ctypes[i])
        s = make_state_actor(ct_i, stypes, dist, free, k)
        sc = make_state_critic(ct_i, prio[i], stypes, dist, free, k)
        x = torch.tensor(s, dtype=torch.float32, device=device)
        xc = torch.tensor(sc, dtype=torch.float32, device=device)
        fm = torch.tensor(free, dtype=torch.bool, device=device)
        with torch.no_grad():
            logits = actor(x.unsqueeze(0), fm.unsqueeze(0))[0]
            d = torch.distributions.Categorical(logits=logits)
            a = d.sample().item()
            logp = d.log_prob(torch.tensor(a, device=device)).item()
            v = critic(xc.unsqueeze(0)).item()
        free[a] = False
        r = -float(cost[i, a])
        ret += r
        traj.append((s, sc, a, logp, r, v))
    return traj, ret


def greedy_eval(actor, device, ev):
    dist = np.arange(1, M + 1, dtype=float)
    gs = []
    with torch.no_grad():
        for cost, ctypes, stypes, prio, order in ev:
            free = np.ones(M, dtype=bool)
            total = 0.0
            for k in range(N):
                i = int(order[k])
                ct_i = int(ctypes[i])
                s = make_state_actor(ct_i, stypes, dist, free, k)
                x = torch.tensor(s, dtype=torch.float32, device=device)
                fm = torch.tensor(free, dtype=torch.bool, device=device)
                logits = actor(x.unsqueeze(0), fm.unsqueeze(0))[0].cpu().numpy()
                a = int(np.argmax(logits))
                free[a] = False
                total += float(cost[i, a])
            _, _, oc = l3d.hungarian(cost)
            gs.append((total - oc) / (oc + 1e-9))
    return float(np.mean(gs)), float(np.std(gs))


# ---------------------------------------------------------------- training
def train_seed(seed, device, n_updates, n_episodes, eval_ep, eval_every, max_stale):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    actor = Actor(in_dim=2 * M + 2).to(device)
    critic = Critic(in_dim=(l3d.N_TYPE + 1) + M * (1 + l3d.N_TYPE + 1) + 2).to(device)
    opt = optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=LR)

    rng_e = np.random.default_rng(seed * 777 + 13)
    ev = []
    for _ in range(eval_ep):
        cost, ctypes, stypes, prio = l3d.make_episode(rng_e, N, M, FRAC)
        order = rng_e.permutation(N)
        ev.append((cost, ctypes, stypes, prio, order))

    curve = []            # (update, gap)
    best_gap = float('inf')
    stale = 0
    done_updates = 0
    for u in range(n_updates):
        buf = []
        for _ in range(n_episodes):
            traj, ret = collect_episode(rng, actor, critic, device)
            n = len(traj)
            advs = np.zeros(n)
            rets = np.zeros(n)
            last_v = 0.0
            g = 0.0
            adv = 0.0
            for t in reversed(range(n)):
                _, _, _, _, r, v = traj[t]
                g = r + g                       # gamma=1 return-to-go
                delta = r + last_v - v
                adv = delta + GAE_LAM * adv
                advs[t] = adv
                rets[t] = g
                last_v = v
            for t in range(n):
                s, sc, a, logp, r, v = traj[t]
                buf.append((s, sc, a, logp, advs[t], rets[t]))
        S = np.array([b[0] for b in buf])
        SC = np.array([b[1] for b in buf])
        A_idx = np.array([b[2] for b in buf])
        OLP = np.array([b[3] for b in buf])
        ADV = np.array([b[4] for b in buf])
        RET = np.array([b[5] for b in buf])
        ADV = (ADV - ADV.mean()) / (ADV.std() + 1e-8)
        n = len(buf)
        idx = np.arange(n)
        for _ in range(PPO_EPOCHS):
            np.random.shuffle(idx)
            for start in range(0, n, MB_SIZE):
                mb = idx[start:start + MB_SIZE]
                xs = torch.tensor(S[mb], dtype=torch.float32, device=device)
                xsc = torch.tensor(SC[mb], dtype=torch.float32, device=device)
                ai = torch.tensor(A_idx[mb], dtype=torch.long, device=device)
                olp = torch.tensor(OLP[mb], dtype=torch.float32, device=device)
                adv = torch.tensor(ADV[mb], dtype=torch.float32, device=device)
                ret = torch.tensor(RET[mb], dtype=torch.float32, device=device)
                fm = torch.tensor(free_mask_of(buf, mb, M), dtype=torch.bool, device=device)
                logits = actor(xs, fm)
                d = torch.distributions.Categorical(logits=logits)
                logp = d.log_prob(ai)
                ratio = (logp - olp).exp()
                surr1 = ratio * adv
                surr2 = torch.clamp(ratio, 1.0 - CLIP, 1.0 + CLIP) * adv
                p_loss = -torch.min(surr1, surr2).mean()
                e_loss = -d.entropy().mean()
                v_loss = ((critic(xsc) - ret) ** 2).mean()
                loss = p_loss + 0.5 * v_loss + ENT_COEF * e_loss
                opt.zero_grad()
                loss.backward()
                opt.step()
        done_updates = u + 1

        if (u + 1) % eval_every == 0:
            gap, gap_std = greedy_eval(actor, device, ev)
            curve.append((u + 1, gap))
            if gap < best_gap - 1e-4:
                best_gap = gap
                stale = 0
            else:
                stale += eval_every
            print(f"  u={u+1:5d} eval_gap={gap*100:6.2f}% (+-{gap_std*100:5.2f}) best={best_gap*100:6.2f}%", flush=True)
            if stale >= max_stale:
                print(f"  early stop at u={u+1} (no improvement for {stale} updates)", flush=True)
                break

    gap, gap_std = greedy_eval(actor, device, ev)
    return {
        'gap': gap, 'gap_std': gap_std, 'best_gap': best_gap,
        'updates_done': done_updates, 'eval_curve': curve,
        'hyperparams': {'gamma': GAMMA, 'gae_lambda': GAE_LAM, 'clip': CLIP,
                        'lr': LR, 'ppo_epochs': PPO_EPOCHS, 'entropy_coef': ENT_COEF,
                        'n_episodes_per_update': n_episodes, 'hidden': HID},
    }


def free_mask_of(buf, mb, M):
    out = np.zeros((len(mb), M), dtype=bool)
    for r, j in enumerate(mb):
        s = buf[j][0]
        per = s[:M * 2].reshape(M, 2)
        out[r] = per[:, 1] > 0.5
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', default=','.join(str(s) for s in DEFAULT_SEEDS))
    ap.add_argument('--updates', type=int, default=2000)
    ap.add_argument('--episodes', type=int, default=64)
    ap.add_argument('--eval-ep', type=int, default=300)
    ap.add_argument('--eval-every', type=int, default=50)
    ap.add_argument('--max-stale', type=int, default=600)
    ap.add_argument('--out', default='l3_mappo_result.json')
    ap.add_argument('--device', default=None)
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(',') if s.strip()]
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    print("device:", device, flush=True)
    t_start = time.time()

    per_seed = {}
    for seed in seeds:
        print(f"--- seed {seed} ---", flush=True)
        r = train_seed(seed, device, args.updates, args.episodes,
                       args.eval_ep, args.eval_every, args.max_stale)
        per_seed[str(seed)] = r
        print(f"  final gap={r['gap']*100:.2f}% +- {r['gap_std']*100:.2f}", flush=True)

    gaps = [per_seed[s]['gap'] for s in map(str, seeds)]
    agg = {'gap_mean': float(np.mean(gaps)), 'gap_std': float(np.std(gaps)),
           'per_seed_gap': [float(v) for v in gaps]}
    dispatcher_path = os.path.join(DATA, 'l3_online_dispatcher_result.json')
    compare_deferred = not os.path.exists(dispatcher_path)

    out = {
        'exp': 'l3_mappo_baseline',
        'dispatcher_code_mtime': os.path.getmtime(os.path.join(HERE, 'l3_online_dispatcher.py')),
        'env': f'parameter-sharing MAPPO, M={M}, N={N}, frac={FRAC}, '
               f'shared anonymous actor (lambda=0 representative) + centralized critic, PPO clip',
        'seeds': seeds, 'device': device,
        'runtime_sec': round(time.time() - t_start, 1),
        'compare_deferred': compare_deferred,
        'per_seed': per_seed,
        'aggregate': agg,
    }
    jpath = os.path.join(DATA, args.out)
    with open(jpath, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("Saved:", jpath, flush=True)

    # ---- figure 1: MAPPO own summary ----
    plt.figure(figsize=(5.6, 3.8), dpi=120)
    plt.bar(['MAPPO\n(shared actor)'], [agg['gap_mean'] * 100],
            yerr=[agg['gap_std'] * 100], capsize=6, color='#2c3e50')
    plt.ylabel('online optimality gap vs oracle (%)')
    plt.title(f'MAPPO baseline ({len(seeds)} seeds, mean+std)', fontsize=10, fontweight='bold')
    plt.grid(alpha=0.3, axis='y')
    plt.tight_layout()
    f1 = os.path.join(FIG, 'l3_mappo_gap.png')
    plt.savefig(f1)
    plt.close()
    print("Saved:", f1, flush=True)

    # ---- figure 2: compare with dispatcher (if result available) ----
    if not compare_deferred:
        with open(dispatcher_path, encoding='utf-8') as f:
            disp = json.load(f)
        a = disp['aggregate']
        names = ['lam=0\n(anonymous)', 'lam=1\n(personalized)', f"lam*={disp['lambda_star_mean']:.2f}\n(adaptive)", 'MAPPO\n(shared actor)']
        vals = [a['0.0']['mean'] * 100, a['1.0']['mean'] * 100, a['lambda_star']['mean'] * 100, agg['gap_mean'] * 100]
        errs = [a['0.0']['std'] * 100, a['1.0']['std'] * 100, a['lambda_star']['std'] * 100, agg['gap_std'] * 100]
        plt.figure(figsize=(6.6, 4.0), dpi=120)
        plt.bar(names, vals, yerr=errs, capsize=5,
                color=['#95a5a6', '#bdc3c7', '#c0392b', '#2c3e50'])
        plt.ylabel('online optimality gap vs oracle (%)')
        plt.title('Dispatcher (imitation) vs MAPPO (RL) baselines, 3 seeds mean+std', fontsize=10, fontweight='bold')
        plt.grid(alpha=0.3, axis='y')
        plt.tight_layout()
        f2 = os.path.join(FIG, 'l3_mappo_compare.png')
        plt.savefig(f2)
        plt.close()
        print("Saved:", f2, flush=True)
    else:
        print("dispatcher result not available yet; l3_mappo_compare.png deferred to integration", flush=True)


if __name__ == '__main__':
    main()
