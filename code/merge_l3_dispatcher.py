# -*- coding: utf-8 -*-
"""
    Merge dispatcher 5-seed results.
"""
import os, re, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
LOG = os.path.join(ROOT, 'logs_l3_final.log')
SEED24 = os.path.join(ROOT, 'data', 'l3_dispatcher_seed2024.json')
OUT = os.path.join(ROOT, 'data', 'l3_online_dispatcher_result.json')
FIG = os.path.join(ROOT, 'figs')

def parse_log(path):
    text = open(path, encoding='utf-8', errors='replace').read()
    blocks = re.split(r'--- seed (\d+) ---', text)
    out = {}
    for i in range(1, len(blocks), 2):
        seed = int(blocks[i]); body = blocks[i + 1]
        res = {}
        eps2 = lam_star = None
        for line in body.splitlines():
            m = re.match(r'\s*lam=\s*([\w.]+): gap=\s*([\d.]+)% ±\s*([\d.]+)', line)
            if m:
                key = m.group(1)
                res[key] = {'gap': float(m.group(2)) / 100.0, 'std': float(m.group(3)) / 100.0}
            m2 = re.match(r'\s*eps2=([\d.]+)\s+lam\*=([\d.]+)', line)
            if m2:
                eps2 = float(m2.group(1)); lam_star = float(m2.group(2))
        if res:
            out[seed] = {'res': res, 'eps2': eps2, 'lam_star': lam_star}
    return out

def main():
    from_log = parse_log(LOG)
    # 4 seeds from log
    seeds_from_log = [s for s in (20260925, 7, 42, 123) if s in from_log]
    per_seed = {}
    for s in seeds_from_log:
        per_seed[str(s)] = from_log[s]['res']
    # seed 2024 from json
    with open(SEED24, encoding='utf-8') as f:
        d24 = json.load(f)
    per_seed['2024'] = d24['per_seed']['2024']
    SEEDS = [20260925, 7, 42, 123, 2024]
    lam_grid = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    agg = {}
    for lam in lam_grid:
        gs = [per_seed[str(s)][str(lam)]['gap'] for s in SEEDS]
        agg[lam] = {'mean': float(np.mean(gs)), 'std': float(np.std(gs))}
    gs_star = [per_seed[str(s)]['lambda_star']['gap'] for s in SEEDS]
    agg['lambda_star'] = {'mean': float(np.mean(gs_star)), 'std': float(np.std(gs_star))}
    lamstars = [from_log[s]['lam_star'] for s in seeds_from_log] + [d24['lam_star']]
    eps2s = [from_log[s]['eps2'] for s in seeds_from_log] + [d24['eps2']]
    agg['lam_star_mean'] = float(np.mean(lamstars))
    agg['eps2_mean'] = float(np.mean(eps2s))
    from scipy import stats as st
    for name, other in [('vs_lam0', 0.0), ('vs_lam1', 1.0)]:
        g0 = np.array([per_seed[str(s)][str(other)]['gap'] for s in SEEDS])
        g1 = np.array(gs_star)
        t, p = st.ttest_ind(g0, g1, equal_var=False)
        agg[name] = {'t': float(t), 'p': float(p)}
    out = {
        'env': 'online soft-equivariant dispatcher, M=16 N=8 frac=0.35, 5 seeds',
        'lam_grid': lam_grid, 'lambda_star_mean': agg['lam_star_mean'],
        'eps2_mean': agg['eps2_mean'], 'per_seed': per_seed,
        'aggregate': agg,
        'welch': {k: v for k, v in agg.items() if isinstance(k, str) and k.startswith('vs_')},
    }
    with open(OUT, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("SAVED:", OUT)
    print("lam* = %.4f  eps2 = %.4f" % (agg['lam_star_mean'], agg['eps2_mean']))
    print("λ0 mean %.2f%% | λ* mean %.2f%% | λ1 mean %.2f%%" % (
        agg[0.0]['mean'] * 100, agg['lambda_star']['mean'] * 100, agg[1.0]['mean'] * 100))
    for k, v in agg.items():
        if isinstance(k, str) and k.startswith('vs_'):
            print("%s: t=%.3f p=%.4f" % (k, v['t'], v['p']))
    # figures
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    lam_star_mean = agg['lam_star_mean']
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
    names = ['λ=0 (anonymous)', 'λ=1 (personalized)', f'λ*={lam_star_mean:.2f} (adaptive)']
    vals = [agg[0.0]['mean'] * 100, agg[1.0]['mean'] * 100, agg['lambda_star']['mean'] * 100]
    errs2 = [agg[0.0]['std'] * 100, agg[1.0]['std'] * 100, agg['lambda_star']['std'] * 100]
    plt.figure(figsize=(6.2, 3.8), dpi=120)
    plt.bar(names, vals, yerr=errs2, capsize=5, color=['#95a5a6', '#bdc3c7', '#c0392b'])
    plt.ylabel('optimality gap vs oracle (%)')
    plt.title('Adaptive λ* vs extremes (5 seeds, mean±std)', fontsize=10, fontweight='bold')
    plt.grid(alpha=0.3, axis='y'); plt.tight_layout()
    plt.savefig(os.path.join(FIG, 'l3_dispatcher_compare.png')); plt.close()
    print("Figures saved to:", FIG)

if __name__ == '__main__':
    main()
