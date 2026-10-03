# -*- coding: utf-8 -*-
"""
    Robustness of the closed-form lambda* under non-Gaussian noise and non-quadratic error structures.
"""
import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, 'data')
FIG = os.path.join(ROOT, 'figs')
os.makedirs(DATA, exist_ok=True)
os.makedirs(FIG, exist_ok=True)

plt.rcParams['axes.unicode_minus'] = False
TITLE_FONT = {'fontsize': 11, 'fontweight': 'bold'}
LAM_GRID = np.linspace(0, 1, 201)


def lambda_star(eps2, sigma2, n, kappa=0.0):
    return eps2 / (eps2 + sigma2 / n + kappa + 1e-12)


# ------------------------------------------------------------ A. non-Gaussian
def run_nonGaussian(eps2=0.16, sigma2=0.25, n_train=8, n_test=300000,
                    laplace_scale=None, seed=20260925):
    rng = np.random.default_rng(seed)
    eps = np.sqrt(eps2)
    sig = np.sqrt(sigma2)
    b = laplace_scale if laplace_scale is not None else sig / np.sqrt(2)

    delta = rng.normal(0, eps, n_test)
    z = delta[:, None] + rng.laplace(0, b, size=(n_test, n_train))
    dhat = z.mean(axis=1)

    losses = [float(np.mean((delta - lam * dhat) ** 2)) for lam in LAM_GRID]
    lopt_grid = LAM_GRID[int(np.argmin(losses))]
    th = lambda_star(eps2, sigma2, n_train)
    loss_th = float(np.mean((delta - th * dhat) ** 2))
    l0, l1 = losses[0], losses[-1]
    dev = abs(lopt_grid - th)
    return {
        'noise': 'Laplace(0, b), b=sigma/sqrt(2) same-variance',
        'interior_min': float(lopt_grid), 'closed_form': float(th),
        'argmin_closed_dev': float(dev),
        'loss_at_lam0': l0, 'loss_at_lam1': l1, 'loss_at_closed': loss_th,
        'closed_dominates_both': bool(loss_th < l0 and loss_th < l1),
        'losses': losses,
    }


# ------------------------------------------------------------ B. non-quadratic
def run_nonQuadratic(eps2=0.16, sigma2=0.25, n_train=8, n_test=300000,
                     c_list=(0.0, 0.5, 1.0, 2.0), seed=20260925):
    rng = np.random.default_rng(seed)
    eps = np.sqrt(eps2); sig = np.sqrt(sigma2)
    delta = rng.normal(0, eps, n_test)
    z = delta[:, None] + rng.normal(0, sig, (n_test, n_train))
    dhat = z.mean(axis=1)
    err = delta[:, None] - LAM_GRID[None, :] * dhat[:, None]   # (n_test, L)
    rows = []
    th = lambda_star(eps2, sigma2, n_train)
    for c in c_list:
        phi = np.mean(err ** 2, axis=0) + c * np.mean(err ** 4, axis=0)
        lopt = LAM_GRID[int(np.argmin(phi))]
        phi0, phi1, phith = phi[0], phi[-1], float(np.interp(th, LAM_GRID, phi))
        rows.append({
            'c': float(c), 'grid_min': float(lopt), 'closed_form': float(th),
            'dev_from_closed': float(abs(lopt - th)),
            'dominates_both': bool(phith < phi0 and phith < phi1),
            'phi_at_closed': float(phith), 'phi_at_0': float(phi0), 'phi_at_1': float(phi1),
        })
    return rows


def main():
    print("=" * 72)
    print("L1-robustness: does closed-form λ* survive non-Gaussian / non-quadratic?")
    print("=" * 72)

    # ---- A. non-Gaussian ----
    resA = run_nonGaussian()
    print("\n[A] Non-Gaussian (Laplace observation noise, same variance)")
    print(f"  interior grid argmin λ={resA['interior_min']:.3f}, "
          f"closed-form λ*={resA['closed_form']:.3f}, |dev|={resA['argmin_closed_dev']:.4f}")
    print(f"  loss λ*= {resA['loss_at_closed']:.4f}  vs  λ=0 {resA['loss_at_lam0']:.4f}  "
          f"λ=1 {resA['loss_at_lam1']:.4f}  -> dominates both: {resA['closed_dominates_both']}")

    plt.figure(figsize=(6.2, 3.8), dpi=120)
    plt.plot(LAM_GRID, resA['losses'], 'o-', color='#2980b9', markersize=3)
    plt.axvline(resA['closed_form'], color='#c0392b', ls='--', lw=1.2,
                label=rf"closed-form $\lambda^*$={resA['closed_form']:.2f}")
    plt.xlabel(r'symmetry injection strength $\lambda$')
    plt.ylabel('test MSE (Laplace noise)')
    plt.title('Robustness A: interior optimum survives heavy-tailed noise', **TITLE_FONT)
    plt.legend(fontsize=8); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(FIG, 'l1_nonGaussian.png')); plt.close()

    # ---- B. non-quadratic ----
    rowsB = run_nonQuadratic()
    print("\n[B] Non-quadratic loss  Φ = E[(δ−λδ̂)²] + c·E[(δ−λδ̂)⁴]")
    print(f"{'c':>6} {'grid min':>10} {'closed λ*':>10} {'|dev|':>8} {'dominates':>10}")
    for r in rowsB:
        print(f"{r['c']:>6.1f} {r['grid_min']:>10.3f} {r['closed_form']:>10.3f} "
              f"{r['dev_from_closed']:>8.4f} {str(r['dominates_both']):>10}")

    plt.figure(figsize=(6.2, 3.8), dpi=120)
    rng = np.random.default_rng(20260925)
    eps = np.sqrt(0.16); sig = np.sqrt(0.25)
    delta = rng.normal(0, eps, 300000)
    z = delta[:, None] + rng.normal(0, sig, (300000, 8))
    dhat = z.mean(axis=1)
    err = delta[:, None] - LAM_GRID[None, :] * dhat[:, None]
    for c, color, lab in [(0.0, '#7f8c8d', 'quadratic c=0'),
                          (0.5, '#2980b9', 'c=0.5'), (2.0, '#c0392b', 'c=2.0')]:
        phi = np.mean(err ** 2, axis=0) + c * np.mean(err ** 4, axis=0)
        phi = (phi - phi.min()) / (phi.max() - phi.min())
        plt.plot(LAM_GRID, phi, '-', color=color, lw=1.6, label=lab)
    th = lambda_star(0.16, 0.25, 8)
    plt.axvline(th, color='#c0392b', ls='--', lw=1.2, label=rf"closed-form $\lambda^*$={th:.2f}")
    plt.xlabel(r'$\lambda$'); plt.ylabel('normalized loss')
    plt.title('Robustness B: closed-form λ* stays near optimum under 4th-order loss', **TITLE_FONT)
    plt.legend(fontsize=8); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(FIG, 'l1_nonQuadratic.png')); plt.close()

    out = {'non_gaussian': resA, 'non_quadratic': rowsB}
    with open(os.path.join(DATA, 'l1_robustness_result.json'), 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("\nSaved:", os.path.join(DATA, 'l1_robustness_result.json'))
    print("Figures:", os.path.join(FIG, 'l1_nonGaussian.png'), os.path.join(FIG, 'l1_nonQuadratic.png'))


if __name__ == '__main__':
    main()
