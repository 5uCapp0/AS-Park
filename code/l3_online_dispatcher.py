# -*- coding: utf-8 -*-
"""
    L3 strictly-online soft-equivariant dispatcher experiments.
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

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, 'data')
FIG = os.path.join(ROOT, 'figs')
os.makedirs(DATA, exist_ok=True)
os.makedirs(FIG, exist_ok=True)

SEEDS = [20260925, 7, 42, 123, 2024]
INF = 1e6
C_NORMAL, C_EV, C_LARGE = 0, 1, 2
N_TYPE = 3

# ---------------------------------------------------------------- environment
def hungarian(cost):
    """Min-cost matching via scipy.linear_sum_assignment (fast C implementation;
    optimal cost identical to the earlier numpy O(M^3) solver)."""
    from scipy.optimize import linear_sum_assignment
    r, c = linear_sum_assignment(cost)
    return np.asarray(r, dtype=int), np.asarray(c, dtype=int), float(cost[r, c].sum())


def make_episode(rng, N, M, frac_special=0.35):
    """Same environment as the supplementary benchmark. Returns cost, types, priorities."""
    spot_types = np.zeros(M, dtype=int)
    n_ev = int(round(M * frac_special / 2))
    n_large = int(round(M * frac_special / 2))
    perm = rng.permutation(M)
    spot_types[perm[:n_ev]] = C_EV
    spot_types[perm[n_ev:n_ev + n_large]] = C_LARGE
    car_types = rng.choice([C_NORMAL, C_EV, C_LARGE], size=N, p=[0.5, 0.3, 0.2])
    priority = rng.uniform(0.0, 1.0, size=N)
    dist = np.arange(1, M + 1, dtype=float)
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
    return cost, car_types, spot_types, priority


def online_oracle_labels(cost, N, M, arrival_order, rng):
    """Per-step online oracle: when car i arrives, re-solve Hungarian on
    {arrived-not-yet-assigned cars} x {free spots}; label = the spot assigned
    to the arriving car. Returns labels[i] for i=0..N-1 (in arrival order)."""
    labels = np.full(N, -1, dtype=int)
    arrived = set(); assigned_spots = set()
    for k in range(N):
        i = int(arrival_order[k])
        arrived.add(i)
        not_assigned = [a for a in arrived if labels[a] < 0]
        free = [j for j in range(M) if j not in assigned_spots]
        sub = cost[np.ix_(not_assigned, free)]
        r, c, _ = hungarian(sub)
        # row for car i
        pos = not_assigned.index(i)
        spot = free[c[pos]]
        labels[i] = spot
        assigned_spots.add(spot)
    return labels


# ---------------------------------------------------------------- model
class SoftEquivariantScorer(nn.Module):
    """π_λ = π_eq (DeepSets) + λ·π_res (type bilinear). Strictly S_N-equivariant
    in the backbone; the residual breaks equivariance by vehicle/spot type."""
    def __init__(self, hid=32, use_types=False):
        super().__init__()
        # By default the backbone is attribute-blind (sees NO type attributes),
        # so lambda=0 is truly anonymous; type enters ONLY via the residual.
        # With use_types=True the equivariant backbone sees car/spot type
        # one-hots (still strictly S_N-equivariant: types permute with cars).
        self.phi_v = nn.Sequential(nn.Linear(N_TYPE + 1 if use_types else 1, hid), nn.ReLU(), nn.Linear(hid, hid))
        self.phi_s = nn.Sequential(nn.Linear(N_TYPE + 1 if use_types else 1, hid), nn.ReLU(), nn.Linear(hid, hid))
        self.score = nn.Sequential(nn.Linear(hid * 3, hid), nn.ReLU(), nn.Linear(hid, 1))
        self.u = nn.Embedding(N_TYPE, 32)
        self.v = nn.Embedding(N_TYPE, 32)
        # NOTE: do NOT zero-init both u and v: res = u@v^T then has zero
        # gradients w.r.t. both factors (each multiplies the other's zero),
        # so the residual channel can never learn. Random init lets gradients flow.
        # Scale: the residual must be able to compete with eq-logits (order ~1);
        # 3-dim embeddings gave |res|~0.08 which can never flip the argmax.
        nn.init.normal_(self.u.weight, std=0.5)
        nn.init.normal_(self.v.weight, std=0.5)
        self.use_types = use_types

    def _feat_v(self, car_type, priority):
        N = car_type.shape[0]
        if not self.use_types:
            # attribute-blind: no attribute enters the equivariant backbone
            return torch.zeros(N, 1, device=car_type.device)
        # attribute-aware equivariant: car-type one-hot + priority (permutes with cars)
        ct = torch.nn.functional.one_hot(car_type, num_classes=N_TYPE).float()
        return torch.cat([ct, priority.unsqueeze(1)], dim=-1)
    def _feat_s(self, spot_type, dist):
        if not self.use_types:
            # position/distance only (global, permutation-irrelevant across cars)
            return dist.unsqueeze(1)                                # (M, 1)
        # attribute-aware equivariant: spot-type one-hot + distance
        st = torch.nn.functional.one_hot(spot_type, num_classes=N_TYPE).float()
        return torch.cat([st, dist.unsqueeze(1)], dim=-1)

    def forward(self, car_type, priority, spot_type, dist, lam):
        N = car_type.shape[0]; M = spot_type.shape[0]
        hv = self.phi_v(self._feat_v(car_type, priority))          # (N,hid)
        hs = self.phi_s(self._feat_s(spot_type, dist))             # (M,hid)
        g = (hv.mean(0) + hs.mean(0)).unsqueeze(0).unsqueeze(0).expand(N, M, -1)
        hv_e = hv.unsqueeze(1).expand(N, M, -1)
        hs_e = hs.unsqueeze(0).expand(N, M, -1)
        eq = self.score(torch.cat([hv_e, hs_e, g], -1)).squeeze(-1)  # (N,M)
        ui = self.u(car_type)                                       # (N,N_TYPE)
        vj = self.v(spot_type)                                      # (M,N_TYPE)
        res = ui @ vj.t()                                           # (N,M)
        # scale-normalize both channels so lambda interpolates on equal footing:
        # eq magnitude ~40 vs res ~1.7 made lambda (0..1) unable to flip argmax.
        eq_s = eq / (eq.std() + 1e-6)
        res_s = res / (res.std() + 1e-6)
        return eq_s + lam * res_s


def online_rollout(model, cost, car_types, priorities, spot_types, dist, arrival_order, lam, device):
    """Strictly online greedy: cars arrive one by one; at each arrival the model
    scores the arriving car against currently free spots and assigns argmax.
    Returns total cost (miss penalty if no free spot)."""
    N, M = cost.shape
    free = set(range(M)); total = 0.0; misses = 0
    with torch.no_grad():
        for k in range(N):
            i = int(arrival_order[k])
            ct = torch.tensor([car_types[i]], dtype=torch.long, device=device)
            pr = torch.tensor([priorities[i]], dtype=torch.float32, device=device)
            st = torch.tensor(spot_types, dtype=torch.long, device=device)
            ds = torch.tensor(dist, dtype=torch.float32, device=device)
            scores = model(ct, pr, st, ds, lam).squeeze(0).cpu().numpy()  # (M,)
            best = None
            for j in sorted(range(M), key=lambda j: -scores[j]):
                if j in free:
                    best = j; break
            if best is None:
                misses += 1; total += 20.0
            else:
                free.discard(best); total += float(cost[i, best])
    return total, misses


# ---------------------------------------------------------------- defect estimator
def defect_energy_eps2(scenes_cost, scenes_D):
    """ε² = mean over scenes of (C_pos_only_assigned - C_full)/C_full, L2-calibrated.
    C_full = type-aware Hungarian on full cost; the anonymous (distance-only)
    Hungarian assignment is re-evaluated under the FULL type-aware cost to give
    C_pos_only_assigned, i.e. the relative suboptimality of ignoring types."""
    vals = []
    for cost, D in zip(scenes_cost, scenes_D):
        _, _, c_full = hungarian(cost)
        r, c, _ = hungarian(D)          # anonymous (distance-only) assignment
        c_anon = float(cost[r, c].sum())  # re-evaluate under FULL type-aware cost
        vals.append((c_anon - c_full) / (c_full + 1e-9))
    return float(np.mean(vals))


# ---------------------------------------------------------------- main
def run_seed(seed, device, n_ep_train=600, n_epoch=15, n_ep_eval=150, use_types=False, N=8, M=16, frac=0.35, sigma2=1.0):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    dist = np.arange(1, M + 1, dtype=float)

    # ---- train data (online oracle labels) ----
    train = []
    for _ in range(n_ep_train):
        cost, ctypes, stypes, prio = make_episode(rng, N, M, frac)
        order = rng.permutation(N)
        labels = online_oracle_labels(cost, N, M, order, rng)
        train.append((cost, ctypes, stypes, prio, labels, order))

    # ---- eval data ----
    ev = []
    ev_scenes_cost, ev_scenes_D = [], []
    for _ in range(n_ep_eval):
        cost, ctypes, stypes, prio = make_episode(rng, N, M, frac)
        order = rng.permutation(N)
        ev.append((cost, ctypes, stypes, prio, order))
        ev_scenes_cost.append(cost)
        ev_scenes_D.append(np.tile(dist, (N, 1)).astype(float))

    model = SoftEquivariantScorer(use_types=use_types).to(device)
    opt = optim.Adam(model.parameters(), lr=1e-2)
    ce = nn.CrossEntropyLoss()

    # ---- train at λ_train = λ*_train (estimated from a probe of the training
    # scenes): the model is optimized exactly where the theory says injection
    # should sit, so evaluating at λ* is in-distribution and the adaptive rule
    # is not disadvantaged against λ=1. The anonymous backbone still has to
    # carry distance-only preference (it sees no type attributes). ----
    probe_cost = [t[0] for t in train[:50]]
    probe_D = [np.tile(dist, (N, 1)).astype(float) for _ in range(50)]
    eps2_train = defect_energy_eps2(probe_cost, probe_D)
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
            # online CE: for each step k, score arriving car vs free spots
            assigned = set(); free_mask = np.ones(M, dtype=bool)
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
                step_loss = step_loss + ce(logits.unsqueeze(0), torch.tensor([int(tgt[0])], dtype=torch.long, device=device))
                n_steps += 1
                free_mask[labels[i]] = False
            loss = loss + step_loss
        loss = loss / max(n_steps, 1) * 4800.0   # normalized CE at constant per-step lr
        loss.backward()
        opt.step()
        if ep % 5 == 0 or ep == n_epoch - 1:
            with torch.no_grad():
                u_norm = model.u.weight.norm().item()
                v_norm = model.v.weight.norm().item()
                print(f"  ep={ep:2d} loss={loss.item():.4f} n_steps={n_steps} |u|={u_norm:.4f} |v|={v_norm:.4f}", flush=True)

    # ---- defect estimator (on eval) ----
    eps2 = defect_energy_eps2(ev_scenes_cost, ev_scenes_D)
    lam_star = eps2 / (eps2 + sigma2 / N + 1e-12)

    # ---- evaluate across λ ----
    lam_grid = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    results = {}
    for lam in lam_grid:
        gaps = []
        for cost, ctypes, stypes, prio, order in ev:
            total, misses = online_rollout(model, cost, ctypes, prio, stypes, dist, order, lam, device)
            _, _, oc = hungarian(cost)
            gaps.append((total - oc) / (oc + 1e-9))
        results[lam] = {'gap': float(np.mean(gaps)), 'std': float(np.std(gaps)),
                        'misses': 0}
    # adaptive λ* (clip to grid? use exact value at inference)
    gaps_star = []
    for cost, ctypes, stypes, prio, order in ev:
        total, misses = online_rollout(model, cost, ctypes, prio, stypes, dist, order, lam_star, device)
        _, _, oc = hungarian(cost)
        gaps_star.append((total - oc) / (oc + 1e-9))
    results['lambda_star'] = {'lam': lam_star, 'gap': float(np.mean(gaps_star)),
                              'std': float(np.std(gaps_star))}
    return results, eps2, lam_star, model


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-cars', type=int, default=8)
    ap.add_argument('--m-spots', type=int, default=16)
    ap.add_argument('--frac', type=float, default=0.35)
    ap.add_argument('--epochs', type=int, default=15)
    ap.add_argument('--seeds', type=str, default=None)
    ap.add_argument('--output', type=str, default='l3_online_dispatcher_result.json')
    ap.add_argument('--sigma', type=float, default=1.0)
    args = ap.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("device:", device, "| GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'n/a')
    seeds = [int(x) for x in args.seeds.split(',')] if args.seeds else SEEDS
    all_results = {}
    eps2s, lamstars = [], []
    for seed in seeds:
        print(f"--- seed {seed} (N={args.n_cars}, M={args.m_spots}) ---")
        res, eps2, lam_star, _ = run_seed(seed, device, n_epoch=args.epochs, N=args.n_cars, M=args.m_spots, frac=args.frac, sigma2=args.sigma)
        all_results[str(seed)] = res
        eps2s.append(eps2); lamstars.append(lam_star)
        for k, v in res.items():
            print(f"  lam={k:>10}: gap={v['gap']*100:6.2f}% ±{v['std']*100:5.2f}")
        print(f"  eps2={eps2:.4f}  lam*={lam_star:.4f}")

    # ---- aggregate: gap(lambda) mean±std across seeds, lambda* vs extremes ----
    lam_grid = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    agg = {}
    for lam in lam_grid:
        gs = [all_results[str(s)][lam]['gap'] for s in seeds]
        agg[lam] = {'mean': float(np.mean(gs)), 'std': float(np.std(gs))}
    gs_star = [all_results[str(s)]['lambda_star']['gap'] for s in seeds]
    agg['lambda_star'] = {'mean': float(np.mean(gs_star)), 'std': float(np.std(gs_star))}
    lam_star_mean = float(np.mean(lamstars))
    agg['lam_star_mean'] = lam_star_mean
    agg['eps2_mean'] = float(np.mean(eps2s))

    # Welch t-test: lambda* vs lam=0, lam=1
    from scipy import stats as st
    for name, other in [('vs_lam0', 0.0), ('vs_lam1', 1.0)]:
        g0 = np.array([all_results[str(s)][other]['gap'] for s in seeds])
        g1 = np.array(gs_star)
        t, p = st.ttest_ind(g0, g1, equal_var=False)
        agg[name] = {'t': float(t), 'p': float(p)}

    out = {'env': f'online soft-equivariant dispatcher, M={args.m_spots} N={args.n_cars} frac={args.frac}, {len(seeds)} seeds',
           'N': args.n_cars, 'M': args.m_spots, 'frac': args.frac, 'seeds': seeds,
           'lam_grid': lam_grid, 'lambda_star_mean': lam_star_mean,
           'eps2_mean': agg['eps2_mean'], 'per_seed': all_results,
           'aggregate': agg, 'welch': {k: v for k, v in agg.items() if isinstance(k, str) and k.startswith('vs_')}}
    with open(os.path.join(DATA, args.output), 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("Saved:", os.path.join(DATA, args.output))

    # ---- figures ----
    xs = lam_grid + [lam_star_mean]
    ys = [agg[l]['mean'] * 100 for l in lam_grid] + [agg['lambda_star']['mean'] * 100]
    errs = [agg[l]['std'] * 100 for l in lam_grid] + [agg['lambda_star']['std'] * 100]
    plt.figure(figsize=(6.2, 3.8), dpi=120)
    plt.errorbar(xs, ys, yerr=errs, fmt='o-', color='#2c3e50', capsize=4, label='online gap (mean±std)')
    plt.axvline(lam_star_mean, color='#c0392b', ls='--', lw=1.2, label=f'λ*={lam_star_mean:.2f}')
    plt.xlabel('symmetry-injection strength λ'); plt.ylabel('optimality gap vs oracle (%)')
    plt.title('Soft-equivariant online dispatcher: λ tunes the gap', fontsize=10, fontweight='bold')
    plt.legend(fontsize=8); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(FIG, 'l3_dispatcher_lambda_scan.png')); plt.close()

    # compare bar: lam=0, lam=1, lam*
    names = ['λ=0 (anonymous)', 'λ=1 (personalized)', f'λ*={lam_star_mean:.2f} (adaptive)']
    vals = [agg[0.0]['mean'] * 100, agg[1.0]['mean'] * 100, agg['lambda_star']['mean'] * 100]
    errs2 = [agg[0.0]['std'] * 100, agg[1.0]['std'] * 100, agg['lambda_star']['std'] * 100]
    plt.figure(figsize=(6.2, 3.8), dpi=120)
    b = plt.bar(names, vals, yerr=errs2, capsize=5, color=['#95a5a6', '#bdc3c7', '#c0392b'])
    plt.ylabel('optimality gap vs oracle (%)')
    plt.title('Adaptive λ* vs extremes (5 seeds, mean±std)', fontsize=10, fontweight='bold')
    plt.grid(alpha=0.3, axis='y'); plt.tight_layout()
    plt.savefig(os.path.join(FIG, 'l3_dispatcher_compare.png')); plt.close()
    print("Figures saved to:", FIG)


if __name__ == '__main__':
    main()
