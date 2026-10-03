# -*- coding: utf-8 -*-
"""
    L2 statistics: Welch t-test and percentile bootstrap CIs.
"""
import numpy as np
from scipy import stats


def welch_t(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) < 2 or len(b) < 2:
        return dict(t=float('nan'), p=float('nan'), mean_diff=float(np.mean(a) - np.mean(b)),
                    significant=None, note='n<2')
    t, p = stats.ttest_ind(a, b, equal_var=False)
    return dict(t=float(t), p=float(p), mean_diff=float(np.mean(a) - np.mean(b)),
                significant=bool(p < 0.05), note='')


def bootstrap_ci(a, n_boot=2000, alpha=0.05, seed=12345):
    a = np.asarray(a, dtype=float)
    rng = np.random.default_rng(seed)
    n = len(a)
    if n == 0:
        return dict(lo=float('nan'), hi=float('nan'), mean=float('nan'))
    means = np.array([np.mean(a[rng.integers(0, n, n)]) for _ in range(n_boot)])
    lo = float(np.percentile(means, 100 * alpha / 2))
    hi = float(np.percentile(means, 100 * (1 - alpha / 2)))
    return dict(lo=lo, hi=hi, mean=float(np.mean(a)))


def compare_group(name, best_vals, ref_vals, seed=12345):
    wt = welch_t(best_vals, ref_vals)
    ci_best = bootstrap_ci(best_vals, seed=seed)
    ci_ref = bootstrap_ci(ref_vals, seed=seed + 1)
    return dict(
        name=name,
        best_mean=float(np.mean(best_vals)), best_std=float(np.std(best_vals, ddof=1)) if len(best_vals) > 1 else 0.0,
        best_ci=ci_best,
        ref_mean=float(np.mean(ref_vals)), ref_std=float(np.std(ref_vals, ddof=1)) if len(ref_vals) > 1 else 0.0,
        ref_ci=ci_ref,
        t=wt['t'], p=wt['p'], significant=wt['significant'],
        mean_diff=wt['mean_diff'],
    )
