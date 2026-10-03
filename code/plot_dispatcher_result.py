# -*- coding: utf-8 -*-
"""
    Plot lambda-scan curves and comparison bar charts from dispatcher JSON.
"""
import json, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = r"PROJECT_ROOT"
DATA = os.path.join(BASE, "data")
FIG = os.path.join(BASE, "figs")
d = json.load(open(os.path.join(DATA, 'l3_online_dispatcher_result.json'), encoding='utf-8'))

lam_grid = d['lam_grid']
agg = d['aggregate']
lam_star_mean = d['lambda_star_mean']

xs = lam_grid + [lam_star_mean]
ys = [agg[str(l)]['mean'] * 100 for l in lam_grid] + [agg['lambda_star']['mean'] * 100]
errs = [agg[str(l)]['std'] * 100 for l in lam_grid] + [agg['lambda_star']['std'] * 100]

plt.figure(figsize=(6.2, 3.8), dpi=120)
plt.errorbar(xs, ys, yerr=errs, fmt='o-', color='#2c3e50', capsize=4, label='online gap (mean±std)')
plt.axvline(lam_star_mean, color='#c0392b', ls='--', lw=1.2, label=f'λ*={lam_star_mean:.2f}')
plt.xlabel('symmetry-injection strength λ'); plt.ylabel('optimality gap vs oracle (%)')
plt.title('Soft-equivariant online dispatcher: λ tunes the gap (5 seeds)', fontsize=10, fontweight='bold')
plt.legend(fontsize=8); plt.grid(alpha=0.3); plt.tight_layout()
plt.savefig(os.path.join(FIG, 'l3_dispatcher_lambda_scan.png')); plt.close()

names = ['λ=0 (anonymous)', 'λ=1 (personalized)', f'λ*={lam_star_mean:.2f} (adaptive)']
vals = [agg['0.0']['mean'] * 100, agg['1.0']['mean'] * 100, agg['lambda_star']['mean'] * 100]
errs2 = [agg['0.0']['std'] * 100, agg['1.0']['std'] * 100, agg['lambda_star']['std'] * 100]
plt.figure(figsize=(6.2, 3.8), dpi=120)
b = plt.bar(names, vals, yerr=errs2, capsize=5, color=['#95a5a6', '#bdc3c7', '#c0392b'])
plt.ylabel('optimality gap vs oracle (%)')
plt.title('Adaptive λ* vs extremes (5 seeds, mean±std)', fontsize=10, fontweight='bold')
plt.grid(alpha=0.3, axis='y'); plt.tight_layout()
plt.savefig(os.path.join(FIG, 'l3_dispatcher_compare.png')); plt.close()
print("Figures saved:", FIG)
