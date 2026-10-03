# -*- coding: utf-8 -*-
"""
    Multi-seed scheduler benchmark (FCFS/priority/adaptive comparisons).
"""
import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, 'data')
FIG = os.path.join(ROOT, 'figs')
os.makedirs(DATA, exist_ok=True)
os.makedirs(FIG, exist_ok=True)

SEEDS = [20260925, 7, 42, 123, 2024]
INF = 1e6
MISS_PENALTY = 20.0
C_NORMAL, C_EV, C_LARGE = 0, 1, 2


def hungarian(cost):
    n, m = cost.shape
    size = m
    C = np.zeros((size, size), dtype=float)
    C[:n, :m] = cost
    u = np.zeros(size + 1); v = np.zeros(size + 1)
    p = np.zeros(size + 1, dtype=int); way = np.zeros(size + 1, dtype=int)
    for i in range(1, size + 1):
        p[0] = i; j0 = 0
        minv = np.full(size + 1, INF); used = np.zeros(size + 1, dtype=bool)
        while True:
            used[j0] = True
            i0 = p[j0]; delta = INF; j1 = -1
            for j in range(1, size + 1):
                if not used[j]:
                    cur = C[i0 - 1, j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j] = cur; way[j] = j0
                    if minv[j] < delta:
                        delta = minv[j]; j1 = j
            for j in range(size + 1):
                if used[j]:
                    u[p[j]] += delta; v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]; p[j0] = p[j1]; j0 = j1
            if j0 == 0:
                break
    rows, cols, total = [], [], 0.0
    for j in range(1, size + 1):
        i = p[j]
        if 1 <= i <= n:
            rows.append(i - 1); cols.append(j - 1); total += cost[i - 1, j - 1]
    return np.array(rows), np.array(cols), float(total)


def make_episode(rng, N, M, frac_special=0.35):
    spot_types = np.zeros(M, dtype=int)
    n_ev = int(round(M * frac_special / 2)); n_large = int(round(M * frac_special / 2))
    perm = rng.permutation(M)
    spot_types[perm[:n_ev]] = C_EV; spot_types[perm[n_ev:n_ev + n_large]] = C_LARGE
    car_types = rng.choice([C_NORMAL, C_EV, C_LARGE], size=N, p=[0.5, 0.3, 0.2])
    priority = rng.uniform(0.0, 1.0, size=N)
    dist = np.arange(1, M + 1, dtype=float)
    cost = np.zeros((N, M), dtype=float)
    for i in range(N):
        ct = car_types[i]
        for j in range(M):
            st = spot_types[j]; c = dist[j]
            if ct == C_EV:
                c += {C_EV: 0.0, C_NORMAL: 2.0, C_LARGE: 1.0}[st]
            elif ct == C_LARGE:
                c += {C_LARGE: 0.0, C_NORMAL: 3.0, C_EV: 3.0}[st]
            cost[i, j] = c
    return cost, car_types, spot_types, priority


def greedy_match(cost, order):
    N, M = cost.shape
    free = np.ones(M, dtype=bool); total = 0.0; misses = 0
    for i in order:
        row = cost[i].copy(); row[~free] = INF
        j = int(np.argmin(row))
        if row[j] >= INF:
            misses += 1; total += MISS_PENALTY
            continue
        free[j] = False; total += float(cost[i, j])
    return total, misses


def run_seed(seed, M=16, frac=0.35, n_ep=200):
    rng = np.random.default_rng(seed)
    Ns = [4, 6, 8, 10, 12]
    gap_vs_N = {n: {'fcfs': [], 'random': [], 'priority': []} for n in Ns}
    for N in Ns:
        for _ in range(n_ep):
            cost, ctypes, stypes, prio = make_episode(rng, N, M, frac)
            _, _, oc = hungarian(cost)
            c_f, _ = greedy_match(cost, list(range(N)))
            c_r, _ = greedy_match(cost, list(rng.permutation(N)))
            c_p, _ = greedy_match(cost, list(np.argsort(-prio)))
            gap_vs_N[N]['fcfs'].append((c_f - oc) / (oc + 1e-9))
            gap_vs_N[N]['random'].append((c_r - oc) / (oc + 1e-9))
            gap_vs_N[N]['priority'].append((c_p - oc) / (oc + 1e-9))
    # heterogeneity sweep at N=8
    het = {f: {'fcfs': [], 'priority': []} for f in [0.15, 0.35, 0.5, 0.6]}
    for f in het:
        for _ in range(n_ep):
            cost, ctypes, stypes, prio = make_episode(rng, 8, M, f)
            _, _, oc = hungarian(cost)
            c_f, _ = greedy_match(cost, list(range(8)))
            c_p, _ = greedy_match(cost, list(np.argsort(-prio)))
            het[f]['fcfs'].append((c_f - oc) / (oc + 1e-9))
            het[f]['priority'].append((c_p - oc) / (oc + 1e-9))
    return gap_vs_N, het


def main():
    Ns = [4, 6, 8, 10, 12]
    fracs = [0.15, 0.35, 0.5, 0.6]
    # accumulate per-seed per-(N) mean gaps: shape {seed: {N: {policy: mean}}}
    per_seed_N = {s: {} for s in SEEDS}
    per_seed_het = {s: {} for s in SEEDS}
    for s in SEEDS:
        gN, het = run_seed(s)
        for n in Ns:
            per_seed_N[s][n] = {k: float(np.mean(v)) for k, v in gN[n].items()}
        for f in fracs:
            per_seed_het[s][f] = {k: float(np.mean(v)) for k, v in het[f].items()}

    rows_N, rows_het = [], []
    for n in Ns:
        row = {'N': n}
        for pol in ['fcfs', 'random', 'priority']:
            vals = [per_seed_N[s][n][pol] for s in SEEDS]
            row[f'gap_{pol}_mean'] = round(float(np.mean(vals)), 4)
            row[f'gap_{pol}_std'] = round(float(np.std(vals)), 4)
        # Welch-t fcfs vs priority
        a = np.array([per_seed_N[s][n]['fcfs'] for s in SEEDS])
        b = np.array([per_seed_N[s][n]['priority'] for s in SEEDS])
        t, p = stats.ttest_ind(a, b, equal_var=False)
        row['welch_fcfs_vs_priority_t'] = round(float(t), 3)
        row['welch_fcfs_vs_priority_p'] = round(float(p), 4)
        rows_N.append(row)
        print(f"N={n:>2}  fcfs {row['gap_fcfs_mean']*100:6.2f}±{row['gap_fcfs_std']*100:4.2f}%  "
              f"random {row['gap_random_mean']*100:6.2f}±{row['gap_random_std']*100:4.2f}%  "
              f"priority {row['gap_priority_mean']*100:6.2f}±{row['gap_priority_std']*100:4.2f}%  "
              f"(Welch p={row['welch_fcfs_vs_priority_p']:.3f})")

    for f in fracs:
        row = {'frac_special': f}
        for pol in ['fcfs', 'priority']:
            vals = [per_seed_het[s][f][pol] for s in SEEDS]
            row[f'gap_{pol}_mean'] = round(float(np.mean(vals)), 4)
            row[f'gap_{pol}_std'] = round(float(np.std(vals)), 4)
        a = np.array([per_seed_het[s][f]['fcfs'] for s in SEEDS])
        b = np.array([per_seed_het[s][f]['priority'] for s in SEEDS])
        t, p = stats.ttest_ind(a, b, equal_var=False)
        row['welch_fcfs_vs_priority_p'] = round(float(p), 4)
        rows_het.append(row)
        print(f"frac={f:.2f}  fcfs {row['gap_fcfs_mean']*100:6.2f}±{row['gap_fcfs_std']*100:4.2f}%  "
              f"priority {row['gap_priority_mean']*100:6.2f}±{row['gap_priority_std']*100:4.2f}%  "
              f"(Welch p={row['welch_fcfs_vs_priority_p']:.3f})")

    # figures
    plt.figure(figsize=(6.2, 3.8), dpi=120)
    for pol, mk, col in [('fcfs', 's', '#e67e22'), ('random', '^', '#7f8c8d'), ('priority', 'o', '#c0392b')]:
        ys = [r[f'gap_{pol}_mean'] * 100 for r in rows_N]
        errs = [r[f'gap_{pol}_std'] * 100 for r in rows_N]
        plt.errorbar(Ns, ys, yerr=errs, fmt=mk + '-', color=col, capsize=4, label=pol)
    plt.xlabel('number of arriving cars N'); plt.ylabel('optimality gap vs oracle (%)')
    plt.title('Greedy ordering gap grows with load (5 seeds, mean±std)', fontsize=10, fontweight='bold')
    plt.legend(fontsize=8); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(FIG, 'sched_gap_vs_N_multiseed.png')); plt.close()

    plt.figure(figsize=(6.2, 3.8), dpi=120)
    for pol, mk, col in [('fcfs', 's', '#e67e22'), ('priority', 'o', '#c0392b')]:
        ys = [r[f'gap_{pol}_mean'] * 100 for r in rows_het]
        errs = [r[f'gap_{pol}_std'] * 100 for r in rows_het]
        plt.errorbar([r['frac_special'] for r in rows_het], ys, yerr=errs,
                     fmt=mk + '-', color=col, capsize=4, label=pol)
    plt.xlabel('fraction of specialized (EV/large) spots'); plt.ylabel('optimality gap vs oracle (%)')
    plt.title('More type heterogeneity -> ordering matters more (5 seeds)', fontsize=10, fontweight='bold')
    plt.legend(fontsize=8); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(FIG, 'sched_gap_vs_heterogeneity_multiseed.png')); plt.close()

    out = {'env': 'M=16 1-D aisle, 5 seeds, n_ep=200 per cell',
           'gap_vs_N': rows_N, 'gap_vs_heterogeneity': rows_het, 'seeds': SEEDS}
    with open(os.path.join(DATA, 'sched_benchmark_multiseed_result.json'), 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("\nSaved:", os.path.join(DATA, 'sched_benchmark_multiseed_result.json'))


if __name__ == '__main__':
    main()
