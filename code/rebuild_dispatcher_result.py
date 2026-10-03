# -*- coding: utf-8 -*-
"""
    Rebuild the dispatcher result JSON from raw logs.
"""
import json, os, re

BASE = r"PROJECT_ROOT"
LOG = os.path.join(BASE, "logs_l3_final.log")
DATA = os.path.join(BASE, "data")

lines = open(LOG, encoding="utf-8", errors="replace").read().splitlines()

seeds = ["20260925", "7", "42", "123", "2024"]
all_results = {}
lam_grid = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

cur_seed = None
per = {}
for ln in lines:
    ls = ln.strip()
    m = re.match(r"--- seed (\S+) ---", ls)
    if m:
        if cur_seed is not None and per:
            all_results[cur_seed] = per
        cur_seed = m.group(1)
        per = {}
        continue
    if cur_seed is None:
        continue
    m = re.match(r"lam=\s*([\d.]+):\s*gap=\s*([\d.]+)% ?± ?([\d.]+)", ls)
    if m:
        lam = m.group(1)
        per[lam] = {'gap': float(m.group(2)) / 100.0, 'gap_pct': float(m.group(2)), 'std_pct': float(m.group(3))}
        continue
    m = re.match(r"lam=lambda_star:\s*gap=\s*([\d.]+)% ?± ?([\d.]+)", ls)
    if m:
        per['lambda_star'] = {'gap': float(m.group(1)) / 100.0, 'gap_pct': float(m.group(1)), 'std_pct': float(m.group(2))}
        continue
    m = re.match(r"eps2=([\d.]+)\s+lam\*=([\d.]+)", ls)
    if m:
        per['eps2'] = float(m.group(1))
        per['lam_star'] = float(m.group(2))
if cur_seed is not None and per:
    all_results[cur_seed] = per

missing = [s for s in seeds if s not in all_results]
print("parsed seeds:", list(all_results.keys()))
if missing:
    print("MISSING:", missing)

# aggregate
agg = {}
for lam in lam_grid:
    key = str(lam)
    gs = [all_results[str(s)][key]['gap'] for s in seeds]
    agg[key] = {'mean': float(sum(gs) / len(gs)), 'std': float((sum((g - sum(gs)/len(gs))**2 for g in gs) / len(gs)) ** 0.5)}
gs_star = [all_results[str(s)]['lambda_star']['gap'] for s in seeds]
star_mean = float(sum(gs_star) / len(gs_star))
star_std = float((sum((g - star_mean)**2 for g in gs_star) / len(gs_star)) ** 0.5)
agg['lambda_star'] = {'mean': star_mean, 'std': star_std}
lam_stars = [all_results[str(s)]['lam_star'] for s in seeds]
eps2s = [all_results[str(s)]['eps2'] for s in seeds]
agg['lam_star_mean'] = float(sum(lam_stars) / len(lam_stars))
agg['eps2_mean'] = float(sum(eps2s) / len(eps2s))

try:
    from scipy import stats as st
    welch = {}
    for name, other in [('vs_lam0', '0.0'), ('vs_lam1', '1.0')]:
        g0 = [all_results[str(s)][other]['gap'] for s in seeds]
        t, p = st.ttest_ind(g0, gs_star, equal_var=False)
        welch[name] = {'t': float(t), 'p': float(p)}
except Exception as e:
    welch = {'error': str(e)}

out = {'env': 'online soft-equivariant dispatcher, M=16 N=8 frac=0.35, 5 seeds (rebuilt from logs_l3_final.log)',
       'lam_grid': lam_grid, 'lambda_star_mean': agg['lam_star_mean'],
       'eps2_mean': agg['eps2_mean'], 'per_seed': all_results,
       'aggregate': agg, 'welch': welch}

os.makedirs(DATA, exist_ok=True)
dst = os.path.join(DATA, 'l3_online_dispatcher_result.json')
with open(dst, 'w', encoding='utf-8') as f:
    json.dump(out, f, ensure_ascii=False, indent=2)
print("Saved:", dst)
print("lam0 mean %.2f%%  lam1 mean %.2f%%  star mean %.2f%%±%.2f  lam*_mean %.3f  eps2 %.3f" % (
    agg['0.0']['mean'] * 100, agg['1.0']['mean'] * 100, star_mean * 100, star_std * 100,
    agg['lam_star_mean'], agg['eps2_mean']))
for s in seeds:
    r = all_results[s]
    print("seed %-8s lam0=%.2f lam1=%.2f star=%.2f lam*=%.3f eps2=%.3f" % (
        s, r['0.0']['gap_pct'], r['1.0']['gap_pct'], r['lambda_star']['gap_pct'], r['lam_star'], r['eps2']))
