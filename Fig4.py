"""
Recovery time T vs alpha — theory curves + SIMULATED error bars.

Simulated values come from the EFFECTIVE-MATRIX linearization

    d(delta)/dt = (M/N - dI) delta,

    M = (1-alpha) A
        + (alpha/N) (sum_k B_ijk + sum_k B_ikj),

which is the exact object the theory (utils.compute_tau) is built on.

T is measured from the leading eigenvalue of M/N - dI in the tangential
(sum-zero) subspace:

    T = 1/|Re(lambda_leading)|.

The uniform (mean-field) direction is projected out, which for Mutualistic
also removes the isolated mean-field eigenvalue; this matches
fig3_theory_vs_simulation.py.

Does NOT modify fig4_tau_vs_alpha.py; standalone file.
"""

import numpy as np
import matplotlib.pyplot as plt
import utils


# ============================================================
# Plot settings
# ============================================================

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'mathtext.fontset': 'cm',

    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.top': True,
    'ytick.right': True,

    'axes.grid': False,

    'legend.frameon': False,

    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})


# ============================================================
# Colors and line styles
# ============================================================

colors = {
    'Random':        '#2c7bb6',
    'Predator-prey': '#d7191c',
    'Mutualistic':   '#fdae61',
    'Mixed':         '#7b3294',
}

linestyles = {
    'Random':        '-',
    'Predator-prey': '--',
    'Mutualistic':   ':',
    'Mixed':         '-.',
}

key_map = {
    'Random':        'random',
    'Predator-prey': 'predprey',
    'Mutualistic':  'mutual',
    'Mixed':         'mixed'
}

types = [
    'Random',
    'Predator-prey',
    'Mutualistic',
    'Mixed'
]


# ============================================================
# Parameters
# ============================================================

N0, p0, sigma0, d0 = 100, 0.3, 1.0, 1

alpha_arr = np.linspace(0, 0.95, 50)
alpha_points = np.linspace(0.0, 0.95, 20)

n_seed = 50
seed_base = 100


# ============================================================
# Effective Jacobian
# ============================================================

def build_J_M(A, B, alpha, N, d):
    """
    Effective-matrix linearization:

        J = M/N - dI

    where

        M = (1-alpha) A
            + (alpha/N) [sum_k B_ijk + sum_k B_ikj].
    """

    s1 = np.sum(B, axis=2)

    s2 = np.sum(
        np.transpose(B, (0, 2, 1)),
        axis=2
    )

    M = (
        (1 - alpha) * A
        + (alpha / N) * (s1 + s2)
    )

    return M / N - d * np.eye(N)


# ============================================================
# Tangential basis
# ============================================================

def tang_basis(N):
    """
    Deterministic orthonormal basis of the sum-zero subspace.
    """

    Q = np.zeros((N, N - 1))

    for k in range(N - 1):

        v = np.zeros(N)

        v[0] = 1.0
        v[k + 1] = -1.0

        for j in range(k):
            v -= np.dot(Q[:, j], v) * Q[:, j]

        Q[:, k] = v / np.linalg.norm(v)

    return Q


# ============================================================
# Recovery time
# ============================================================

def measure_tau(J, Q):
    """
    Recovery time from the leading tangential eigenvalue:

        T = 1 / |Re(lambda_leading)|.
    """

    evals = np.linalg.eigvals(Q.T @ J @ Q)

    lam = np.max(evals.real)

    return 1.0 / abs(lam) if lam < 0 else np.nan


# ============================================================
# Tangential basis
# ============================================================

Q_tang = tang_basis(N0)


# ============================================================
# Figure
# ============================================================

fig, ax = plt.subplots(figsize=(7, 5))


# ============================================================
# Theory curves
# ============================================================

for tp in types:

    tau_vals = [
        utils.compute_tau(
            tp,
            a,
            sigma0,
            p0,
            N0,
            d0
        )
        for a in alpha_arr
    ]

    ax.plot(
        alpha_arr,
        tau_vals,

        color=colors[tp],
        ls=linestyles[tp],
        lw=2.0,

        label=f'{tp}'
    )


# ============================================================
# Simulated values: mean ± standard deviation
# ============================================================

for tp in types:

    print(
        f"Simulating {tp} "
        f"(effective M, leading eigenvalue) ..."
    )

    # --------------------------------------------------------
    # Generate network realizations
    # --------------------------------------------------------

    nets = [
        utils.GENERATORS[key_map[tp]](
            N0,
            p0,
            sigma0,
            seed=seed
        )
        for seed in range(
            seed_base,
            seed_base + n_seed
        )
    ]

    # --------------------------------------------------------
    # Marker style
    # --------------------------------------------------------

    marker_style = (
        'o'
        
    )

    # --------------------------------------------------------
    # Loop over alpha
    # --------------------------------------------------------

    for a in alpha_points:

        taus = []

        T_theory = utils.compute_tau(
            tp,
            a,
            sigma0,
            p0,
            N0,
            d0
        )

        # ----------------------------------------------------
        # Calculate T for each network realization
        # ----------------------------------------------------

        for A, B, _ in nets:

            J = build_J_M(
                A,
                B,
                a,
                N0,
                d0
            )

            taus.append(
                measure_tau(
                    J,
                    Q_tang
                )
            )

        # ----------------------------------------------------
        # Remove unstable / invalid realizations
        # ----------------------------------------------------

        taus = np.array([
            t for t in taus
            if not np.isnan(t)
        ])

        # ----------------------------------------------------
        # Mean ± standard deviation
        # ----------------------------------------------------

        if len(taus):

            T_mean = taus.mean()
            T_std = taus.std()

            # ------------------------------------------------
            # Simulated error bars
            #
            # label='_nolegend_' ensures that the individual
            # simulation points do NOT appear in the legend.
            # ------------------------------------------------

            ax.errorbar(
                a,
                T_mean,

                yerr=T_std,

                fmt=marker_style,

                markersize=3.5,

                color=colors[tp],
                ecolor=colors[tp],

                elinewidth=1.0,
                capsize=1.8,
                capthick=1.0,

                linestyle=':',

                alpha=1.0,

                markerfacecolor=colors[tp],
                markeredgecolor=colors[tp],
                markeredgewidth=0.7,

                zorder=5,

                # Do NOT add individual simulation points
                # to the legend.
                label='_nolegend_'
            )

            # ------------------------------------------------
            # Print results
            # ------------------------------------------------

            print(
                f"    alpha={a:.1f}: "
                f"T_theory={T_theory:.3f}, "
                f"T_sim(mean±std)="
                f"{T_mean:.3f}±{T_std:.3f}"
            )


# ============================================================
# One unified legend handle for all simulations
# ============================================================
#
# This errorbar is NOT drawn in the actual figure because
# x and y are empty. It only creates a legend entry.
#
# fmt='none' means that the legend shows ONLY the error bar,
# without a marker.
# ============================================================

ax.errorbar(
    [],
    [],
    yerr=[1],

    fmt='o',
    markersize=2.5,

    color='grey',
    ecolor='grey',

    elinewidth=1.0,
    capsize=2.0,
    capthick=1.0,

    markerfacecolor='grey',
    markeredgecolor='grey',
    markeredgewidth=0.7,

    linestyle='none',

    label='Simulation'
)


# ============================================================
# Axis formatting
# ============================================================

ax.set_xlabel(
    r'Mixing fraction, $\alpha$',
    fontsize=18
)

ax.set_ylabel(
    r'Recovery time, $T$',
    fontsize=18
)
ax.set_ylim(1, 1.15)
ax.tick_params(
    labelsize=16,
    width=1.2,
    length=5
)

for spine in ax.spines.values():
    spine.set_linewidth(1.2)


# ============================================================
# Legend
# ============================================================

handles, labels = ax.get_legend_handles_labels()

ax.legend(
    handles=handles,
    labels=labels,

    fontsize=12,

    frameon=True,
    framealpha=0.9,
    facecolor='white',
    edgecolor='black',

    handlelength=3.0,
    handletextpad=0.6,
    labelspacing=0.5,

    borderpad=0.5,

    loc='best'
)


# ============================================================
# Layout
# ============================================================

fig.tight_layout(pad=0.5)


# ============================================================
# Save
# ============================================================

fig.savefig(
    '/Users/xichenli/venv/bin/高阶加速恢复/'
    'fig4_tau_vs_alpha_linearized_M_sim-2.pdf',

    dpi=600,
    bbox_inches='tight'
)

plt.close(fig)

print(
    "Done: "
    "fig4_tau_vs_alpha_linearized_M_sim-2.pdf"
)