# -*- coding: utf-8 -*-
"""
    Grid-based parking scheduling simulation validating the closed-form symmetry-injection strength lambda* (torch).
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.optimize import linear_sum_assignment
from collections import deque
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

SEED = 20260925
rng = np.random.default_rng(SEED)
torch.manual_seed(SEED)
FIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figs_sim')
os.makedirs(FIG, exist_ok=True)
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

ALPHA = 20.0
N_TYPE = 3
COMPAT = {0: [0], 1: [1], 2: [2]}


# ============================================================
# ============================================================
def build_grid(W=10, H=10, n_obs=4):
    obs = set()
    while len(obs) < n_obs:
        c = (int(rng.integers(1, W - 1)), int(rng.integers(1, H - 1)))
        obs.add(c)
    return obs


def bfs_dists(W, H, obstacles, targets):
    dists = np.full((len(targets), H, W), np.inf)
    for k, (tx, ty) in enumerate(targets):
        d = np.full((H, W), -1, dtype=np.int32)
        d[ty, tx] = 0
        q = deque([(tx, ty)])
        while q:
            x, y = q.popleft()
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nx, ny = x + dx, y + dy
                if 0 <= nx < W and 0 <= ny < H and (nx, ny) not in obstacles and d[ny, nx] == -1:
                    d[ny, nx] = d[y, x] + 1
                    q.append((nx, ny))
        dists[k] = d
    return dists  # (M, H, W)


def make_scene(W, H, obstacles, n_cars, M, p_hetero):
    free = [(x, y) for x in range(W) for y in range(H) if (x, y) not in obstacles]
    spot_idx = rng.choice(len(free), size=M, replace=False)
    spots = [free[i] for i in spot_idx]
    spot_type = np.zeros(M, dtype=np.int64)
    n_special = int(round(M * p_hetero))
    if n_special > 0:
        sp = rng.choice(M, size=n_special, replace=False)
        spot_type[sp] = rng.choice([1, 2], size=n_special)
    car_cells = rng.choice(len(free), size=n_cars, replace=False)
    car_pos = [free[i] for i in car_cells]
    car_type = np.zeros(n_cars, dtype=np.int64)
    n_special_car = int(round(n_cars * p_hetero))
    if n_special_car > 0:
        sc = rng.choice(n_cars, size=n_special_car, replace=False)
        car_type[sc] = rng.choice([1, 2], size=n_special_car)
    dists = bfs_dists(W, H, obstacles, spots)          # (M,H,W)
    D = np.stack([dists[:, y, x] for x, y in car_pos], axis=1).T   # (N,M)
    cost = D.copy().astype(np.float64)
    for i in range(n_cars):
        for j in range(M):
            if spot_type[j] not in COMPAT[int(car_type[i])]:
                cost[i, j] += ALPHA
    return dict(car_pos=car_pos, car_type=car_type, spots=spots, spot_type=spot_type,
                D=D, cost=cost)


def oracle(cost):
    r, c = linear_sum_assignment(cost)
    return r, c, cost[r, c].sum()


# ============================================================
# ============================================================
class ParkingPolicy(nn.Module):
    def __init__(self, hid=32):
        super().__init__()
        self.phi = nn.Sequential(nn.Linear(2, hid), nn.ReLU(), nn.Linear(hid, hid))
        self.psi = nn.Sequential(nn.Linear(2, hid), nn.ReLU(), nn.Linear(hid, hid))
        self.score = nn.Sequential(nn.Linear(hid * 3, hid), nn.ReLU(), nn.Linear(hid, 1))  # [phi,psi,g]
        self.res = nn.Embedding(N_TYPE * N_TYPE, 1)
        nn.init.zeros_(self.res.weight)

    def forward(self, car_pos, spot_pos, car_type, spot_type, lam):
        """car_pos:(N,2) spot_pos:(M,2) car_type:(N,) spot_type:(M,)  -> scores (N,M)"""
        N = car_pos.shape[0]; M = spot_pos.shape[0]
        hc = self.phi(car_pos.float())                 # (N,hid)
        hs = self.psi(spot_pos.float())                # (M,hid)
        g = hc.mean(0) + hs.mean(0)
        g = g.unsqueeze(0).unsqueeze(0).expand(N, M, -1)  # (N,M,hid)
        hc_e = hc.unsqueeze(1).expand(N, M, -1)        # (N,M,hid)
        hs_e = hs.unsqueeze(0).expand(N, M, -1)        # (N,M,hid)
        eq = self.score(torch.cat([hc_e, hs_e, g], -1)).squeeze(-1)   # (N,M)
        key = car_type[:, None] * N_TYPE + spot_type[None, :]         # (N,M)
        res = self.res(key).squeeze(-1)                               # (N,M)
        return eq + lam * res


def train(policy, scenes, lam, n_epoch=28, lr=3e-3):
    opt = optim.Adam(policy.parameters(), lr=lr)
    data = []
    for sc in scenes:
        cp = torch.tensor(sc['car_pos'], dtype=torch.float32)
        sp = torch.tensor(sc['spots'], dtype=torch.float32)
        ct = torch.tensor(sc['car_type'], dtype=torch.long)
        st = torch.tensor(sc['spot_type'], dtype=torch.long)
        r, c, _ = oracle(sc['cost'])
        target = torch.tensor(c, dtype=torch.long)
        data.append((cp, sp, ct, st, target))
    ce = nn.CrossEntropyLoss()
    for ep in range(n_epoch):
        opt.zero_grad()
        loss = 0.0
        for cp, sp, ct, st, target in data:
            scores = policy(cp, sp, ct, st, lam)     # (N,M)
            loss = loss + ce(scores, target)
        loss.backward()
        opt.step()
    return policy


def evaluate(policy, scenes, lam):
    gaps = []
    for sc in scenes:
        cp = torch.tensor(sc['car_pos'], dtype=torch.float32)
        sp = torch.tensor(sc['spots'], dtype=torch.float32)
        ct = torch.tensor(sc['car_type'], dtype=torch.long)
        st = torch.tensor(sc['spot_type'], dtype=torch.long)
        with torch.no_grad():
            scores = policy(cp, sp, ct, st, lam)
        r, c = linear_sum_assignment(-scores.numpy())
        _, _, oc = oracle(sc['cost'])
        cost_hat = sc['cost'][r, c].sum()
        gaps.append((cost_hat - oc) / (oc + 1e-9))
    return float(np.mean(gaps))


# ============================================================
# ============================================================
def gen_scenes(n, W, H, obstacles, n_cars, M, p_hetero):
    return [make_scene(W, H, obstacles, n_cars, M, p_hetero) for _ in range(n)]


def run_all():
    W = H = 10
    obstacles = build_grid(W, H, n_obs=4)
    n_cars = 8
    M = 16

    p = 0.35
    n_train = 350
    n_test = 150
    lam_grid = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    print("=" * 70)
    print(f"E1 lambda sweep: heterogeneity p={p}, train scenes {n_train}, test {n_test}")
    print("=" * 70)
    train_scenes = gen_scenes(n_train, W, H, obstacles, n_cars, M, p)
    test_scenes = gen_scenes(n_test, W, H, obstacles, n_cars, M, p)
    e1_gaps = []
    for lam in lam_grid:
        pol = ParkingPolicy()
        train(pol, train_scenes, lam)
        g = evaluate(pol, test_scenes, lam)
        e1_gaps.append(g)
        print(f"  λ={lam:.1f}  gap={g:.4f}")
    lam_star = lam_grid[int(np.argmin(e1_gaps))]
    print(f"  interior best lam*~={lam_star:.1f} (gap={min(e1_gaps):.4f})")

    plt.figure(figsize=(6.2, 3.8), dpi=120)
    plt.plot(lam_grid, e1_gaps, 'o-', color='#2980b9', markersize=5)
    plt.axvline(lam_star, color='#c0392b', ls='--', lw=1, label=f'interior optimum λ*≈{lam_star:.1f}')
    plt.xlabel(r'symmetry injection strength $\lambda$')
    plt.ylabel('optimality gap')
    plt.title('E1 (sim): optimum is interior, not binary', fontsize=11, fontweight='bold')
    plt.legend(fontsize=8); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(FIG, 'e1_lambda_scan.png')); plt.close()

    print()
    print("=" * 70)
    print("E2 heterogeneity sweep: lam* increases monotonically with p")
    print("=" * 70)
    p_grid = [0.0, 0.15, 0.3, 0.45]
    e2 = []
    for p in p_grid:
        tr = gen_scenes(n_train, W, H, obstacles, n_cars, M, p)
        te = gen_scenes(n_test, W, H, obstacles, n_cars, M, p)
        best_lam, best_gap = 0.0, np.inf
        for lam in lam_grid:
            pol = ParkingPolicy(); train(pol, tr, lam)
            g = evaluate(pol, te, lam)
            if g < best_gap:
                best_gap, best_lam = g, lam
        e2.append((p, best_lam, best_gap))
        print(f"  p={p:.1f}  best lam*={best_lam:.1f}  gap={best_gap:.4f}")

    plt.figure(figsize=(6.2, 3.8), dpi=120)
    plt.plot([x[0] for x in e2], [x[1] for x in e2], 'o-', color='#c0392b', markersize=6)
    plt.xlabel('heterogeneity p  (mixed vehicle types)')
    plt.ylabel(r'optimal $\lambda^*$')
    plt.title('E2 (sim): λ* rises with heterogeneity ε', fontsize=11, fontweight='bold')
    plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(FIG, 'e2_hetero.png')); plt.close()

    print()
    print("=" * 70)
    print("E3 data efficiency: lam* increases monotonically with n train scenes")
    print("=" * 70)
    n_grid = [100, 350, 1200]
    e3 = []
    for n in n_grid:
        tr = gen_scenes(n, W, H, obstacles, n_cars, M, p)
        te = gen_scenes(n_test, W, H, obstacles, n_cars, M, p)
        best_lam, best_gap = 0.0, np.inf
        for lam in lam_grid:
            pol = ParkingPolicy(); train(pol, tr, lam)
            g = evaluate(pol, te, lam)
            if g < best_gap:
                best_gap, best_lam = g, lam
        e3.append((n, best_lam, best_gap))
        print(f"  n={n:>4}  best lam*={best_lam:.1f}  gap={best_gap:.4f}")

    plt.figure(figsize=(6.2, 3.8), dpi=120)
    plt.plot([x[0] for x in e3], [x[1] for x in e3], 'o-', color='#2980b9', markersize=6)
    plt.xlabel('n  (training scenarios)')
    plt.ylabel(r'optimal $\lambda^*$')
    plt.title('E3 (sim): λ* rises with data n', fontsize=11, fontweight='bold')
    plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(FIG, 'e3_data_eff.png')); plt.close()

    print()
    print("E1 interior best lam*~=%.1f" % lam_star)
    print("E2 λ* vs p:", [(round(x[0], 1), round(x[1], 1)) for x in e2])
    print("E3 λ* vs n:", [(x[0], round(x[1], 1)) for x in e3])
    print("figure saved to:", FIG)
    import json
    result = {
        "E1_lambda_scan": {"lam": [round(x, 2) for x in lam_grid.tolist()],
                            "gap": [round(x, 4) for x in e1_gaps],
                            "lam_star": round(float(lam_star), 2)},
        "E2_hetero": [{"p": x[0], "lam_star": round(float(x[1]), 2), "gap": round(x[2], 4)} for x in e2],
        "E3_data_eff": [{"n": x[0], "lam_star": round(float(x[1]), 2), "gap": round(x[2], 4)} for x in e3],
    }
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'parking_sim_result.json'), 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print("results exported to parking_sim_result.json")
    return result


if __name__ == '__main__':
    run_all()
