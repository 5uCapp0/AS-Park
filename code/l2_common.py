# -*- coding: utf-8 -*-
"""
    Shared protocol utilities for the L2 offline experiments (paper Sec. 8.6).
"""
import os
import sys
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.optimize import linear_sum_assignment

import common
from common import ps, bfs_path, closed_loop_assign, simulate_movement, simulate_movement_trace

# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(THIS_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
FIG_DIR = os.path.join(PROJECT_ROOT, 'figs')
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

SEEDS = [20260925, 7, 42, 123, 2024]
LAM_GRID = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
N_TRAIN = 120
N_TEST = 60
N_EPOCH = 12
LR = 3e-3

N_TYPE = ps.N_TYPE       # 3: 0 normal / 1 EV / 2 large
ALPHA = ps.ALPHA
COMPAT = ps.COMPAT

# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
FAMILY_A = dict(name='open_dense', W=10, H=10, n_obs=6, n_cars=14, M=14, kind='open')
FAMILY_B = dict(name='narrow_corridor', W=12, H=8, n_cars=8, M=10, kind='corridor')


def build_family(fam, rng):
    if fam['kind'] == 'open':
        W, H = fam['W'], fam['H']
        return build_open_obstacles(W, H, fam['n_obs'], rng), W, H
    else:
        W, H = fam['W'], fam['H']
        obs, gap_y = build_corridor_obstacles(W, H)
        return obs, W, H


def gen_family(n, fam, obstacles, p_hetero, rng):
    if fam['kind'] == 'open':
        return gen_scenes_open(n, fam['W'], fam['H'], obstacles,
                               fam['n_cars'], fam['M'], p_hetero, rng)
    else:
        return gen_scenes_corridor(n, fam['W'], fam['H'], obstacles,
                                   fam['n_cars'], fam['M'], p_hetero, rng)


def reset_seed(seed):
    common.reset_seed(seed)


# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
def build_open_obstacles(W, H, n_obs, rng):
    obs = set()
    guard = 0
    while len(obs) < n_obs and guard < 10000:
        c = (int(rng.integers(1, W - 1)), int(rng.integers(1, H - 1)))
        obs.add(c)
        guard += 1
    return obs


def gen_scenes_open(n, W, H, obstacles, n_cars, M, p_hetero, rng):
    old = ps.rng
    ps.rng = rng
    try:
        scenes = [ps.make_scene(W, H, obstacles, n_cars, M, p_hetero) for _ in range(n)]
    finally:
        ps.rng = old
    return scenes


# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
def build_corridor_obstacles(W, H):
    obstacles = set()
    gap_y = H // 2
    for y in range(H):
        if y != gap_y:
            obstacles.add((W // 2, y))
    return obstacles, gap_y


def make_scene_corridor(W, H, obstacles, n_cars, M, p_hetero, rng):
    mid = W // 2
    left_cells = [(x, y) for x in range(0, mid) for y in range(H)
                  if (x, y) not in obstacles]
    right_cells = [(x, y) for x in range(mid + 1, W) for y in range(H)
                   if (x, y) not in obstacles]
    spot_sel = rng.choice(len(right_cells), size=min(M, len(right_cells)), replace=False)
    spots = [right_cells[i] for i in spot_sel]
    M = len(spots)
    spot_type = np.zeros(M, dtype=np.int64)
    n_sp = int(round(M * p_hetero))
    if n_sp > 0:
        sp = rng.choice(M, size=n_sp, replace=False)
        spot_type[sp] = rng.choice([1, 2], size=n_sp)
    car_sel = rng.choice(len(left_cells), size=min(n_cars, len(left_cells)), replace=False)
    car_pos = [left_cells[i] for i in car_sel]
    N = len(car_pos)
    car_type = np.zeros(N, dtype=np.int64)
    n_cp = int(round(N * p_hetero))
    if n_cp > 0:
        cp = rng.choice(N, size=n_cp, replace=False)
        car_type[cp] = rng.choice([1, 2], size=n_cp)
    dists = ps.bfs_dists(W, H, obstacles, spots)          # (M,H,W)
    D = np.stack([dists[:, y, x] for x, y in car_pos], axis=1).T   # (N,M)
    cost = D.copy().astype(np.float64)
    for i in range(N):
        for j in range(M):
            if spot_type[j] not in COMPAT[int(car_type[i])]:
                cost[i, j] += ALPHA
    return dict(car_pos=car_pos, car_type=car_type, spots=spots, spot_type=spot_type,
                D=D, cost=cost)


def gen_scenes_corridor(n, W, H, obstacles, n_cars, M, p_hetero, rng):
    return [make_scene_corridor(W, H, obstacles, n_cars, M, p_hetero, rng)
            for _ in range(n)]


# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
def closed_loop_evaluate(policy, scenes, obstacles, lam, W, H, car_order=None,
                         return_per_scene=False, lam_veh=None, lam_spot=None):
    gaps, waits, confs, dead = [], [], [], []
    for sc in scenes:
        cp = torch.tensor(sc['car_pos'], dtype=torch.float32)
        sp = torch.tensor(sc['spots'], dtype=torch.float32)
        ct = torch.tensor(sc['car_type'], dtype=torch.long)
        st = torch.tensor(sc['spot_type'], dtype=torch.long)
        with torch.no_grad():
            if lam_veh is not None:
                scores = policy(cp, sp, ct, st, lam_veh, lam_spot).numpy()
            else:
                scores = policy(cp, sp, ct, st, lam).numpy()
        assign, _ = closed_loop_assign(scores, car_order)
        paths, total = [], 0.0
        stranded = 0
        N = len(sc['car_pos'])
        for i in range(N):
            if i not in assign:
                stranded += 1
                continue
            j = assign[i]
            path = bfs_path(W, H, obstacles, sc['car_pos'][i], sc['spots'][j])
            if path is None:
                stranded += 1
                continue
            paths.append((i, path))
        path_list = [p for (_, p) in sorted(paths, key=lambda x: x[0])]
        for (i, path) in sorted(paths, key=lambda x: x[0]):
            j = assign[i]
            d = len(path) - 1
            penalty = 0.0 if sc['spot_type'][j] in COMPAT[int(sc['car_type'][i])] else ALPHA
            total += d + penalty
        _, wait, nconf = simulate_movement(path_list)
        total += wait
        _, _, oc = ps.oracle(sc['cost'])
        gaps.append((total - oc) / (oc + 1e-9))
        waits.append(wait)
        confs.append(nconf)
        dead.append(1.0 if stranded > 0 else 0.0)
    if return_per_scene:
        return gaps, waits, confs, dead
    return (float(np.mean(gaps)), float(np.mean(waits)),
            float(np.mean(confs)), float(np.mean(dead)))


def open_loop_gap(policy, scenes, lam, lam_veh=None, lam_spot=None):
    gaps = []
    for sc in scenes:
        cp = torch.tensor(sc['car_pos'], dtype=torch.float32)
        sp = torch.tensor(sc['spots'], dtype=torch.float32)
        ct = torch.tensor(sc['car_type'], dtype=torch.long)
        st = torch.tensor(sc['spot_type'], dtype=torch.long)
        with torch.no_grad():
            if lam_veh is not None:
                scores = policy(cp, sp, ct, st, lam_veh, lam_spot).numpy()
            else:
                scores = policy(cp, sp, ct, st, lam).numpy()
        r, c = linear_sum_assignment(-scores)
        _, _, oc = ps.oracle(sc['cost'])
        cost_hat = sc['cost'][r, c].sum()
        gaps.append((cost_hat - oc) / (oc + 1e-9))
    return float(np.mean(gaps))


# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
def permutation_defect(policy, scenes, lam, rng, n_perm=2):
    defects = []
    for sc in scenes:
        cp = torch.tensor(sc['car_pos'], dtype=torch.float32)
        sp = torch.tensor(sc['spots'], dtype=torch.float32)
        ct = torch.tensor(sc['car_type'], dtype=torch.long)
        st = torch.tensor(sc['spot_type'], dtype=torch.long)
        with torch.no_grad():
            S = policy(cp, sp, ct, st, lam).numpy()            # (N,M)
        N = S.shape[0]
        perms = []
        for _ in range(n_perm):
            sigma = rng.permutation(N)
            cp_p = cp[sigma]
            ct_p = ct[sigma]
            with torch.no_grad():
                Sp = policy(cp_p, sp, ct_p, st, lam).numpy()   # (N,M)
            ref = S[sigma, :]
            perms.append(float(np.mean((Sp - ref) ** 2)))
        defects.append(float(np.mean(perms)))
    return defects


# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
def defect_energy(scenes):
    from scipy.optimize import linear_sum_assignment
    out = []
    for sc in scenes:
        _, _, c_full = ps.oracle(sc['cost'])
        r_pos, c_pos = linear_sum_assignment(sc['D'])
        c_pos = float(sc['D'][r_pos, c_pos].sum())
        eps_s = (c_pos - c_full) / (c_full + 1e-9)
        out.append(eps_s)
    return out


# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
class LowRankParkingPolicy(nn.Module):
    def __init__(self, r=4, hid=32):
        super().__init__()
        self.r = r
        self.phi = nn.Sequential(nn.Linear(2, hid), nn.ReLU(), nn.Linear(hid, hid))
        self.psi = nn.Sequential(nn.Linear(2, hid), nn.ReLU(), nn.Linear(hid, hid))
        self.score = nn.Sequential(nn.Linear(hid * 3, hid), nn.ReLU(), nn.Linear(hid, 1))
        if r > 0:
            self.u = nn.Embedding(N_TYPE, r)
            self.v = nn.Embedding(N_TYPE, r)
            nn.init.normal_(self.u.weight, std=1e-3)
            nn.init.normal_(self.v.weight, std=1e-3)
        else:
            self.u = self.v = None

    def forward(self, car_pos, spot_pos, car_type, spot_type, lam):
        N = car_pos.shape[0]; M = spot_pos.shape[0]
        hc = self.phi(car_pos.float())
        hs = self.psi(spot_pos.float())
        g = (hc.mean(0) + hs.mean(0)).unsqueeze(0).unsqueeze(0).expand(N, M, -1)
        hc_e = hc.unsqueeze(1).expand(N, M, -1)
        hs_e = hs.unsqueeze(0).expand(N, M, -1)
        eq = self.score(torch.cat([hc_e, hs_e, g], -1)).squeeze(-1)
        if self.r == 0 or lam == 0.0:
            return eq
        ui = self.u(car_type)                 # (N,r)
        vj = self.v(spot_type)                # (M,r)
        res = ui @ vj.t()                     # (N,M)
        return eq + lam * res


# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
class VectorParkingPolicy(nn.Module):
    def __init__(self, hid=32):
        super().__init__()
        self.phi = nn.Sequential(nn.Linear(2, hid), nn.ReLU(), nn.Linear(hid, hid))
        self.psi = nn.Sequential(nn.Linear(2, hid), nn.ReLU(), nn.Linear(hid, hid))
        self.score = nn.Sequential(nn.Linear(hid * 3, hid), nn.ReLU(), nn.Linear(hid, 1))
        self.r_veh = nn.Embedding(N_TYPE, 1)
        self.r_spot = nn.Embedding(N_TYPE, 1)
        nn.init.zeros_(self.r_veh.weight)
        nn.init.zeros_(self.r_spot.weight)

    def forward(self, car_pos, spot_pos, car_type, spot_type, lam_veh, lam_spot):
        N = car_pos.shape[0]; M = spot_pos.shape[0]
        hc = self.phi(car_pos.float())
        hs = self.psi(spot_pos.float())
        g = (hc.mean(0) + hs.mean(0)).unsqueeze(0).unsqueeze(0).expand(N, M, -1)
        hc_e = hc.unsqueeze(1).expand(N, M, -1)
        hs_e = hs.unsqueeze(0).expand(N, M, -1)
        eq = self.score(torch.cat([hc_e, hs_e, g], -1)).squeeze(-1)
        rv = self.r_veh(car_type).squeeze(-1)          # (N,)
        rs = self.r_spot(spot_type).squeeze(-1)       # (M,)
        return eq + lam_veh * rv.unsqueeze(1).expand(N, M) \
                   + lam_spot * rs.unsqueeze(0).expand(N, M)


# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
def _prefetch(scenes):
    data = []
    for sc in scenes:
        cp = torch.tensor(sc['car_pos'], dtype=torch.float32)
        sp = torch.tensor(sc['spots'], dtype=torch.float32)
        ct = torch.tensor(sc['car_type'], dtype=torch.long)
        st = torch.tensor(sc['spot_type'], dtype=torch.long)
        r, c, _ = ps.oracle(sc['cost'])
        target = torch.tensor(c, dtype=torch.long)
        data.append((cp, sp, ct, st, target))
    return data


def train_base(policy, scenes, lam, n_epoch=N_EPOCH, lr=LR):
    opt = optim.Adam(policy.parameters(), lr=lr)
    data = _prefetch(scenes)
    ce = nn.CrossEntropyLoss()
    for ep in range(n_epoch):
        opt.zero_grad()
        loss = 0.0
        for cp, sp, ct, st, target in data:
            scores = policy(cp, sp, ct, st, lam)
            loss = loss + ce(scores, target)
        loss.backward()
        opt.step()
    return policy


def train_vector(policy, scenes, lam_veh, lam_spot, n_epoch=N_EPOCH, lr=LR):
    opt = optim.Adam(policy.parameters(), lr=lr)
    data = _prefetch(scenes)
    ce = nn.CrossEntropyLoss()
    for ep in range(n_epoch):
        opt.zero_grad()
        loss = 0.0
        for cp, sp, ct, st, target in data:
            scores = policy(cp, sp, ct, st, lam_veh, lam_spot)
            loss = loss + ce(scores, target)
        loss.backward()
        opt.step()
    return policy


# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
def kappa_theorem61(r_dim, m_in, n_a, n_train, M_eff):
    return r_dim * (m_in + n_a) / (2.0 * n_train * M_eff)


def lambda_star_closed(eps2, sigma2, n_train, kappa):
    return eps2 / (eps2 + sigma2 / n_train + kappa + 1e-12)
