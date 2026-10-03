# -*- coding: utf-8 -*-
"""
    Generate the phase-diagram figure (lambda* iso-contours).
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

kappa = 0.05
eps2 = np.linspace(0.005, 0.5, 400)
sn = np.linspace(0.005, 0.5, 400)
EE, SS = np.meshgrid(eps2, sn)
LAM = EE / (EE + SS + kappa)

fig, ax = plt.subplots(figsize=(7.2, 5.4))
levels = [0.2, 0.4, 0.6, 0.8]
cf = ax.contourf(EE, SS, LAM, levels=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
                 cmap='RdYlBu_r', alpha=0.55, extend='both')
cs = ax.contour(EE, SS, LAM, levels=levels, colors='k', linewidths=1.2)
ax.clabel(cs, inline=True, fontsize=9, fmt='$\\lambda^*$=%.1f')

# region labels
ax.text(0.42, 0.06, 'defect-dominated\n$\\lambda^*\\to 1$', ha='center', fontsize=9, color='darkred')
ax.text(0.07, 0.42, 'symmetry-dominated\n$\\lambda^*\\to 0$', ha='center', fontsize=9, color='darkblue')
ax.text(0.22, 0.24, 'balancing regime\n$\\lambda^*\\in(0.2,0.8)$', ha='center', fontsize=10, color='black', fontweight='bold')

# experimental points
pts = [
    (0.104, 0.125, '$\\lambda^*$=0.455 (L3 online)', 'o', 'C1'),
    (0.104, 0.03125, '$\\lambda^*$=0.77 ($\\sigma^2$=0.25)', 's', 'C2'),
    (0.104, 0.0625,  '$\\lambda^*$=0.63 ($\\sigma^2$=0.5)', 's', 'C2'),
    (0.104, 0.25,   '$\\lambda^*$=0.30 ($\\sigma^2$=2.0)', 's', 'C2'),
]
for x, y, lab, mk, col in pts:
    ax.plot(x, y, mk, ms=9, mfc='none', mec=col, mew=2)
    dx = 0.012 if y < 0.2 else -0.012
    ax.annotate(lab, (x, y), xytext=(x + dx, y + 0.012), fontsize=8.5, color=col)

ax.set_xlabel('heterogeneity defect  $\\varepsilon^2$')
ax.set_ylabel('noise per sample  $\\sigma^2/n$')
ax.set_title('Closed-form $\\lambda^*$ and the balancing regime ($\\kappa=0.05$)')
ax.set_xlim(0.005, 0.5); ax.set_ylim(0.005, 0.5)
fig.tight_layout()
out = r'PROJECT_ROOT\paper\figs\phase_diagram.png'
fig.savefig(out, dpi=200)
print('saved', out)
