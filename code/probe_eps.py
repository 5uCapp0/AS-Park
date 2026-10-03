# -*- coding: utf-8 -*-
"""
    Probe the heterogeneity defect epsilon^2 from cost matrices.
"""
import numpy as np
from scipy.optimize import linear_sum_assignment
import l2_common as L


def defect_energy(scenes):
    out = []
    for sc in scenes:
        r_full, c_full, c_full_cost = L.ps.oracle(sc['cost'])
        r_pos, c_pos = linear_sum_assignment(sc['D'])
        c_pos_cost = sc['D'][r_pos, c_pos].sum()
        eps_s = (c_pos_cost - c_full_cost) / (c_full_cost + 1e-9)
        out.append(eps_s)
    return out


for fam in (L.FAMILY_A, L.FAMILY_B):
    print(f"=== {fam['name']} ===")
    for p in [0.0, 0.15, 0.3, 0.45, 0.6]:
        rng = np.random.default_rng(20260925)
        obstacles, W, H = L.build_family(fam, rng)
        te = L.gen_family(60, fam, obstacles, p, rng)
        e = np.array(defect_energy(te))
        eps2 = float(np.mean(e ** 2))
        sig2 = float(np.var(e, ddof=1))
        kappa = L.kappa_theorem61(9, L.N_TYPE, 1, L.N_TRAIN, fam['M'])
        lam_hat = L.lambda_star_closed(eps2, sig2, L.N_TRAIN, kappa)
        print(f"  p={p:.2f}  mean_eps={np.mean(e):.4f}  eps2={eps2:.5f}  "
              f"sig2={sig2:.6f}  kappa={kappa:.4f}  lam_hat={lam_hat:.3f}")
