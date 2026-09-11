"""
Lambda vs mixing fraction — theory (solid) vs simulation error bars (dashed).

2-by-2 layout: (a) Random (b) Predator-prey (c) Mutualistic (d) Mixed.
Theory: leading eigenvalue from ensemble statistics sigma_M, rho_M, drawn as a
        solid curve over the mixing fraction alpha.
Simulation: direct diagonalization of M/N - dI (uniform mode removed for
        mutualistic) averaged over networks, drawn as error bars (mean +- std)
        joined by a dashed line.

Standalone file; does NOT modify fig3_theory_vs_simulation.py.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.container import ErrorbarContainer
from matplotlib.legend_handler import HandlerErrorbar
import utils

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'mathtext.fontset': 'cm',
    'font.size': 16, 'axes.labelsize': 18, 'axes.titlesize': 18,
    'xtick.labelsize': 16, 'ytick.labelsize': 16, 'legend.fontsize': 13,
    'axes.linewidth': 1.2, 'lines.linewidth': 1.8,
    'xtick.major.size': 5, 'ytick.major.size': 5,
    'xtick.major.width': 1.2, 'ytick.major.width': 1.2,
    'xtick.direction': 'in', 'ytick.direction': 'in',
    'xtick.top': True, 'ytick.right': True,
    'axes.grid': False, 'legend.frameon': False,
    'pdf.fonttype': 42, 'ps.fonttype': 42,
})

N_sim = 100; p_sim = 0.3; sigma_sim = 0.8
R0 = np.sqrt(N_sim) * sigma_sim * np.sqrt(p_sim)
d_sim = R0 / N_sim * 2.0
alpha_points = [0.0, 0.2, 0.4, 0.6, 0.8]        # where simulation error bars sit
alpha_curve = np.linspace(0.0, 0.8, 201)         # theory solid-curve sampling
n_real = 50; seed_base = 42

colors = {'Random':        '#2c7bb6',
          'Predator-prey': '#d7191c',
          'Mutualistic':   '#fdae61',
          'Mixed':         '#7b3294'}


def build_J_eff(A, B, av, N, d_val):
    s1 = np.sum(B, axis=2)
    s2 = np.sum(np.transpose(B, (0, 2, 1)), axis=2)
    M = (1 - av) * A + (av / N) * (s1 + s2)
    return M / N - d_val * np.eye(N)


def measure_lambda(A, B, av, d_val, N, tp):
    J = build_J_eff(A, B, av, N, d_val)
    eigs, evecs = np.linalg.eig(J)
    if tp == 'Mutualistic':
        u = np.ones(N) / np.sqrt(N)
        overlap = np.abs(np.dot(evecs.T.conj(), u))
        outlier_idx = np.argmax(overlap)
        mask = np.ones(N, dtype=bool); mask[outlier_idx] = False
        eigs = eigs[mask]
    lam_max = np.max(np.real(eigs))
    return np.nan if lam_max >= 0 else lam_max


def lambda_theory(tp, av, sigma, p, N, d):
    s2, sm, rho = utils.analytical_sigmaM_rhoM(tp, av, sigma, p, N)
    return np.sqrt(N) * sm * (1.0 + rho) / N - d


class HandlerCappedErrorbar(HandlerErrorbar):
    """Legend entry for an error bar WITH caps (default handler drops them)."""
    def create_artists(self, legend, orig_handle, xdescent, ydescent,
                       width, height, fontsize, trans):
        x0 = xdescent + width / 2.0
        y0 = ydescent + height / 2.0
        lw = orig_handle.elinewidth if getattr(orig_handle, 'elinewidth', None) is not None else 1.4
        cap = orig_handle.capsize if getattr(orig_handle, 'capsize', None) is not None else 3.5
        ec = orig_handle.lines[0].get_color()

        artists = []
        # vertical bar
        artists.append(Line2D([x0, x0], [ydescent, ydescent + height],
                              color=ec, lw=lw, transform=trans))
        # upper and lower caps
        for yy in (ydescent, ydescent + height):
            artists.append(Line2D([x0 - cap, x0 + cap], [yy, yy],
                                  color=ec, lw=lw, transform=trans))
        # central marker
        artists.append(Line2D([x0], [y0], marker='o', color=ec,
                              markerfacecolor=ec, markeredgecolor=ec,
                              linestyle='none', transform=trans))
        return artists


types  = ['Random', 'Predator-prey', 'Mutualistic', 'Mixed']
labels = ['a', 'b', 'c', 'd']
key_map = {'Random': 'random', 'Predator-prey': 'predprey',
           'Mutualistic': 'mutual', 'Mixed': 'mixed'}

fig, axes = plt.subplots(2, 2, figsize=(7, 6.2))
fig.subplots_adjust(wspace=0.36, hspace=0.34,
                    left=0.15, right=0.96, top=0.93, bottom=0.13)

for ax, tp, lab in zip(axes.flat, types, labels):
    # --- theory: solid curve over alpha ---
    tcurve = [lambda_theory(tp, a, sigma_sim, p_sim, N_sim, d_sim)
              for a in alpha_curve]
    ax.plot(alpha_curve, tcurve, '-', color='black', lw=2.2,
            zorder=2, label='Theory')

    # --- simulation: error bars (mean ± std) joined by a dashed line ---
    nets = [utils.GENERATORS[key_map[tp]](N_sim, p_sim, sigma_sim, seed=seed_base + rep)
            for rep in range(n_real)]
    means, stds = [], []
    for av in alpha_points:
        lams = [measure_lambda(A, B, av, d_sim, N_sim, tp) for A, B, _ in nets]
        lams = np.array([x for x in lams if not np.isnan(x)])
        means.append(lams.mean()); stds.append(lams.std())

    c = colors[tp]
    ax.errorbar(alpha_points, means, yerr=stds, fmt='o', markersize=5.5,
                color=c, ecolor=c, elinewidth=1.4, capsize=3.5, capthick=1.4,
                alpha=1, zorder=5, label='Simulation')
    ax.plot(alpha_points, means, '--', color=c, lw=1.6, zorder=4)

    ax.set_xlabel(r'Mixing fraction, $\alpha$', fontsize=18)
    ax.set_ylabel(r'$\max\ \Re(\lambda(\widetilde{J}))$', fontsize=18)
    ax.set_xlim(-0.02, 0.82)

    ax.text(-0.14, 1.04, lab, transform=ax.transAxes,
            fontsize=18, fontweight='bold', va='bottom', ha='left')
    ax.text(0.50, 1.04, tp, transform=ax.transAxes,
            fontsize=14, fontweight='bold', va='bottom', ha='center')
    ax.tick_params(top=True, right=True, direction='in',
                   labeltop=False, labelright=False)

    # --- legend: Simulation (dashed) above Theory (solid) ---
    handles, hlabels = ax.get_legend_handles_labels()
    order = [hlabels.index('Simulation'), hlabels.index('Theory')]
    handles = [handles[i] for i in order]
    hlabels = [hlabels[i] for i in order]
    ax.legend(handles, hlabels, fontsize=13, frameon=False, loc='best',
              handlelength=2.8,
              handler_map={ErrorbarContainer: HandlerCappedErrorbar()})

plt.savefig('/Users/xichenli/venv/bin/高阶加速恢复/fig3_lambda_vs_alpha_errorbar.pdf',
            dpi=600, bbox_inches='tight')
plt.close()
print("Done: fig3_lambda_vs_alpha_errorbar.pdf")