# -*- coding: utf-8 -*-
"""
    Plot L3 5-seed results.
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.normpath(os.path.join(HERE, "..", "data"))
FIGS = os.path.normpath(os.path.join(HERE, "..", "figs"))
os.makedirs(FIGS, exist_ok=True)

def load(name):
    with open(os.path.join(BASE, name + ".json"), "r", encoding="utf-8") as f:
        return json.load(f)

plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "figure.dpi": 120})

# ----------------------------------------------------------------------
# 1) MAPPO (RL) vs dispatcher arms -- 5 seeds, closed-loop gap vs oracle
# ----------------------------------------------------------------------
disp = load("l3_online_dispatcher_result")
mappo = load("l3_mappo_result_cloud5seed")

labels = [r"$\lambda=0$ (anonymous)", r"$\lambda=1$ (personalized)",
          r"$\lambda^{*}$ (adaptive)", "MAPPO (shared actor)"]
means = [disp["aggregate"]["0.0"]["mean"] * 100,
         disp["aggregate"]["1.0"]["mean"] * 100,
         disp["aggregate"]["lambda_star"]["mean"] * 100,
         mappo["aggregate"]["gap_mean"] * 100]
stds = [disp["aggregate"]["0.0"]["std"] * 100,
        disp["aggregate"]["1.0"]["std"] * 100,
        disp["aggregate"]["lambda_star"]["std"] * 100,
        mappo["aggregate"]["gap_std"] * 100]
colors = ["tab:orange", "tab:red", "tab:green", "tab:blue"]

fig, ax = plt.subplots(figsize=(6.0, 4.0))
x = np.arange(len(labels))
ax.bar(x, means, yerr=stds, capsize=4, color=colors, alpha=0.85,
       error_kw={"lw": 1.2, "ecolor": "black"})
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=9)
ax.set_ylabel("online optimality gap vs oracle (%)")
ax.set_title("Dispatcher (imitation) vs MAPPO (RL): 5 seeds, mean$\\pm$std")
for xi, m, s in zip(x, means, stds):
    ax.text(xi, m + s + 0.6, f"{m:.2f}", ha="center", fontsize=8.5)
ax.set_ylim(0, 24)
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(FIGS, "l3_mappo_compare_5seed.png"))
plt.close(fig)
print("saved l3_mappo_compare_5seed.png  means=%s" % [round(m, 2) for m in means])

# ----------------------------------------------------------------------
# 2) Sample-efficiency learning curves -- 5 seeds, 30 epochs
# ----------------------------------------------------------------------
se = load("l3_sample_efficiency_result_cloud5seed")
epochs = np.arange(len(se["aggregate"]["lam0"]["curve_mean"]))
fig, ax = plt.subplots(figsize=(6.0, 4.0))
for key, color, lab in [("lam0", "tab:orange", r"$\lambda=0$"),
                        ("lam1", "tab:red", r"$\lambda=1$"),
                        ("star", "tab:green", r"$\lambda^{*}$")]:
    agg = se["aggregate"][key]
    m = np.array(agg["curve_mean"]) * 100
    s = np.array(agg["curve_std"]) * 100
    ax.plot(epochs, m, color=color, label=lab, lw=1.8)
    ax.fill_between(epochs, m - s, m + s, color=color, alpha=0.15)
ax.set_xlabel("training epoch")
ax.set_ylabel("online optimality gap vs oracle (%)")
ax.set_title("Sample efficiency (5 seeds, mean$\\pm$std)")
ax.legend(frameon=False, loc="upper right")
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(FIGS, "l3_sample_efficiency_5seed.png"))
plt.close(fig)
print("saved l3_sample_efficiency_5seed.png  final lam0/star/lam1 = %.2f / %.2f / %.2f" % (
    se["aggregate"]["lam0"]["final_mean"] * 100,
    se["aggregate"]["star"]["final_mean"] * 100,
    se["aggregate"]["lam1"]["final_mean"] * 100))

# ----------------------------------------------------------------------
# 3) Fleet-size generalization -- 5 seeds, train N=8, test N=8/12/16
# ----------------------------------------------------------------------
gen = load("l3_generalization_result_cloud5seed")
Ns = ["8", "12", "16"]
fig, ax = plt.subplots(figsize=(6.0, 4.0))
for key, color, lab, mk in [("lam0", "tab:orange", r"$\lambda=0$", "o"),
                            ("lam1", "tab:red", r"$\lambda=1$", "s"),
                            ("star", "tab:green", r"$\lambda^{*}$", "^")]:
    m = [gen["aggregate"][n][key]["mean"] * 100 for n in Ns]
    s = [gen["aggregate"][n][key]["std"] * 100 for n in Ns]
    ax.errorbar(Ns, m, yerr=s, marker=mk, capsize=4, color=color,
                label=lab, lw=1.6, ms=6)
ax.set_xlabel("number of cars $N$ (trained at $N=8$)")
ax.set_ylabel("online optimality gap vs oracle (%)")
ax.set_title("Fleet-size generalization (5 seeds, mean$\\pm$std)")
ax.legend(frameon=False)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(FIGS, "l3_generalization_5seed.png"))
plt.close(fig)
print("saved l3_generalization_5seed.png  lam_star_mean =",
      [round(gen["aggregate"][n]["lam_star_mean"], 3) for n in Ns])
