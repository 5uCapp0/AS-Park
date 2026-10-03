# -*- coding: utf-8 -*-
"""
    Measure the effective orbit size M_eff from a permutation probe.
"""
import sys, numpy as np
sys.path.insert(0, r'PROJECT_ROOT\code')
from l3_online_dispatcher import make_episode  # same generator as the paper's L3

rng = np.random.default_rng(20260925)
N, M, frac = 8, 16, 0.35
K = 600          # scenes
P = 20           # permutations per scene

def feat(cost, ctypes, stypes, prio):
    """Feature vector: type-count histogram (permutation-invariant by
    construction) PLUS type x stall-type co-occurrence (sensitive to
    type-position coupling, which is the only thing that can break
    permutation invariance of the scene distribution)."""
    tc = np.bincount(ctypes, minlength=3).astype(float)
    tc /= tc.sum()
    co = np.zeros((3, 3))
    for i in range(N):
        co[ctypes[i], stypes[np.argmin(cost[i])]] += 1   # type -> preferred stall type
    co /= co.sum()
    return np.concatenate([tc, co.ravel()])

diffs = []
for k in range(K):
    cost, ctypes, stypes, prio = make_episode(rng, N, M, frac)
    f0 = feat(cost, ctypes, stypes, prio)
    for _ in range(P):
        perm = rng.permutation(N)
        f1 = feat(cost[perm], ctypes[perm], stypes, prio[perm])
        diffs.append(0.5 * np.abs(f0 - f1).sum())  # total variation
rho = float(np.mean(diffs))
rho_std = float(np.std(diffs))
M_eff = 40320.0 / (1 + 40319.0 * rho)
print(f'rho(TV vs orbit-uniform) = {rho:.4f} ± {rho_std:.4f}  (N=8, frac={frac})')
print(f'M_eff = N!/(1+(N!-1)rho) = {M_eff:.1f}  (N!=40320)')
print(f'log10(M_eff) = {np.log10(M_eff):.2f}')
