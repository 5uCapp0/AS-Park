# -*- coding: utf-8 -*-
"""
    MAPPO baseline at larger fleet sizes.
"""
import os
import sys
import json
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import l3_online_dispatcher as lod
import l3_scale

HERE = os.path.dirname(os.path.abspath(__file__))

HID = 128
GAMMA, GAE_LAM, CLIP, LR, PPO_EPOCHS, ENT_COEF = 1.0, 0.95, 0.2, 3e-4, 4, 0.01
MB_SIZE = 256


class Actor(nn.Module):
    def __init__(self, in_dim, M):
        super().__init__()
        self.M = M
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


def make_state_actor(spot_types, dist, free_mask, step, N, M):
    per = np.stack([dist, free_mask.astype(float)], axis=1)
    glo = np.array([step / N, free_mask.sum() / M], dtype=float)
    return np.concatenate([per.reshape(-1), glo])


def make_state_critic(car_type, priority, spot_types, dist, free_mask, step, N, M):
    car = np.zeros(lod.N_TYPE + 1, dtype=float)
    car[car_type] = 1.0
    car[lod.N_TYPE] = priority
    st = np.zeros((M, lod.N_TYPE))
    st[np.arange(M), spot_types] = 1.0
    per = np.concatenate([dist[:, None], st, free_mask[:, None].astype(float)], axis=1)
    glo = np.array([step / N, free_mask.sum() / M], dtype=float)
    return np.concatenate([car, per.reshape(-1), glo])


def collect_episode(rng, actor, critic, device, N, M, dist_template):
    cost, ctypes, stypes, prio, dist = l3_scale.make_episode_grid(rng, N, M, 0.35)
    order = rng.permutation(N)
    free = np.ones(M, dtype=bool)
    traj = []
    ret = 0.0
    for k in range(N):
        i = int(order[k])
        ct_i = int(ctypes[i])
        s = make_state_actor(stypes, dist, free, k, N, M)
        sc = make_state_critic(ct_i, prio[i], stypes, dist, free, k, N, M)
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


def greedy_eval(actor, device, ev, N, M):
    gs = []
    with torch.no_grad():
        for cost, ctypes, stypes, prio, order, dist in ev:
            free = np.ones(M, dtype=bool)
            total = 0.0
            for k in range(N):
                i = int(order[k])
                ct_i = int(ctypes[i])
                s = make_state_actor(stypes, dist, free, k, N, M)
                x = torch.tensor(s, dtype=torch.float32, device=device)
                fm = torch.tensor(free, dtype=torch.bool, device=device)
                logits = actor(x.unsqueeze(0), fm.unsqueeze(0))[0].cpu().numpy()
                a = int(np.argmax(logits))
                free[a] = False
                total += float(cost[i, a])
            _, _, oc = lod.hungarian(cost)
            gs.append((total - oc) / (oc + 1e-9))
    return float(np.mean(gs)), float(np.std(gs))


def train_seed(seed, device, N, M, n_updates, n_episodes, eval_ep, eval_every, max_stale):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    actor = Actor(in_dim=2 * M + 2, M=M).to(device)
    critic = Critic(in_dim=(lod.N_TYPE + 1) + M * (1 + lod.N_TYPE + 1) + 2).to(device)
    opt = optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=LR)

    rng_e = np.random.default_rng(seed * 777 + 13)
    ev = []
    for _ in range(eval_ep):
        cost, ctypes, stypes, prio, dist = l3_scale.make_episode_grid(rng_e, N, M, 0.35)
        order = rng_e.permutation(N)
        ev.append((cost, ctypes, stypes, prio, order, dist))

    best_gap = float('inf')
    stale = 0
    done_updates = 0
    for u in range(n_updates):
        buf = []
        for _ in range(n_episodes):
            traj, ret = collect_episode(rng, actor, critic, device, N, M, None)
            n = len(traj)
            advs = np.zeros(n)
            rets = np.zeros(n)
            last_v = 0.0
            g = 0.0
            adv = 0.0
            for t in reversed(range(n)):
                _, _, _, _, r, v = traj[t]
                g = r + g
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
            gap, gap_std = greedy_eval(actor, device, ev, N, M)
            if gap < best_gap - 1e-4:
                best_gap = gap
                stale = 0
            else:
                stale += eval_every
            print("  u=%5d eval_gap=%.2f%% (+-%.2f) best=%.2f%%" % (u + 1, gap * 100, gap_std * 100, best_gap * 100), flush=True)
            if stale >= max_stale:
                print("  early stop at u=%d" % (u + 1), flush=True)
                break

    gap, gap_std = greedy_eval(actor, device, ev, N, M)
    return {'gap': gap, 'gap_std': gap_std, 'best_gap': best_gap,
            'updates_done': done_updates,
            'hyperparams': {'gamma': GAMMA, 'gae_lambda': GAE_LAM, 'clip': CLIP,
                            'lr': LR, 'ppo_epochs': PPO_EPOCHS, 'entropy_coef': ENT_COEF,
                            'n_episodes_per_update': n_episodes, 'hidden': HID}}


def free_mask_of(buf, mb, M):
    out = np.zeros((len(mb), M), dtype=bool)
    for r, j in enumerate(mb):
        s = buf[j][0]
        per = s[:M * 2].reshape(M, 2)
        out[r] = per[:, 1] > 0.5
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--N', type=int, default=20)
    ap.add_argument('--seeds', default='1024,2048,4096')
    ap.add_argument('--updates', type=int, default=1000)
    ap.add_argument('--episodes', type=int, default=32)
    ap.add_argument('--eval-ep', type=int, default=150)
    ap.add_argument('--eval-every', type=int, default=100)
    ap.add_argument('--max-stale', type=int, default=400)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()
    N, M = args.N, 4 * args.N
    seeds = [int(s) for s in args.seeds.split(',') if s.strip()]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("N=%d M=%d device=%s seeds=%s" % (N, M, device, seeds), flush=True)
    t_start = time.time()
    per_seed = {}
    for seed in seeds:
        print("--- seed %d ---" % seed, flush=True)
        r = train_seed(seed, device, N, M, args.updates, args.episodes,
                       args.eval_ep, args.eval_every, args.max_stale)
        per_seed[str(seed)] = r
        print("  final gap=%.2f%% +- %.2f" % (r['gap'] * 100, r['gap_std'] * 100), flush=True)
    gaps = [per_seed[s]['gap'] for s in map(str, seeds)]
    out = {
        'exp': 'l3_mappo_scale', 'N': N, 'M': M, 'seeds': seeds,
        'gap_mean': float(np.mean(gaps)), 'gap_std': float(np.std(gaps)),
        'per_seed_gap': [float(v) for v in gaps],
        'device': device, 'runtime_sec': round(time.time() - t_start, 1),
        'per_seed': per_seed,
    }
    out_path = args.out or os.path.join(HERE, 'data', 'mappo_scale_%d.json' % N)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("Saved:", out_path, flush=True)


if __name__ == '__main__':
    main()
