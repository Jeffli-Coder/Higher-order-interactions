r"""
Recovery time T vs system complexity (N and p) — SIMULATION only.
Top: T vs N at alpha = 0, 0.4, 0.9.
Bottom: T vs p at alpha = 0, 0.4, 0.9.

Simulated T follows the effective-matrix linearization of
fig4_tau_vs_alpha_linearized_M_sim.py:
    J = M/N - d I,   M = (1-alpha) A + (alpha/N) (sum_k B_ijk + sum_k B_ikj),
    T = 1/|Re(lambda_leading)| in the tangential (sum-zero) subspace.

Each curve point is the MEAN over n_real network realizations. No theory curve,
no error bars.

NOTE on cost: utils' network generators build the rank-3 tensor B with a pure
Python O(N^3) loop (N=500 takes ~55 s per network). A scan to N=500 is therefore
slow; lower N_max to speed up debugging at ~N^3 run-time cost.

PRE / Chaos double-column style.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, LogLocator
import utils

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
key_map = {'Random': 'random', 'Predator-prey': 'predprey',
           'Mutualistic': 'mutual', 'Mixed': 'mixed'}
types = ['Random', 'Predator-prey', 'Mutualistic', 'Mixed']

alphas = [0.0, 0.4, 0.9]

# ---- simulation ensemble settings (tunable) ----
n_real = 5          # networks averaged per point
seed_base = 42
N_max = 150         # top-row upper N (original fig5 used 500)

# ---- top row: T vs N ----
N_arr = np.unique(
    np.round(
        np.logspace(np.log10(20), np.log10(N_max), 18)
    ).astype(int)
)
p_fixed, sigma_fixed, d_fixed = 0.5, 1.0, 1

# ---- bottom row: T vs p ----
p_arr = np.linspace(0.05, 0.95, 30)
N_fixed, sigma_fixed2, d_fixed2 = 100, 1.0, 1


def build_J_M(A, B, alpha, N, d):
    s1 = np.sum(B, axis=2)
    s2 = np.sum(np.transpose(B, (0, 2, 1)), axis=2)
    M = (1 - alpha) * A + (alpha / N) * (s1 + s2)
    return M / N - d * np.eye(N)


def tang_basis(N):
    """Orthonormal basis of the sum-zero subspace (columns perpendicular to 1)."""
    Q = np.zeros((N, N - 1))
    for k in range(N - 1):
        v = np.zeros(N)
        v[0] = 1.0
        v[k + 1] = -1.0
        for j in range(k):
            v -= np.dot(Q[:, j], v) * Q[:, j]
        Q[:, k] = v / np.linalg.norm(v)
    return Q


def measure_tau(J, Q):
    evals = np.linalg.eigvals(Q.T @ J @ Q)
    lam = np.max(evals.real)
    return 1.0 / abs(lam) if lam < 0 else np.nan


Q_cache = {}


def sim_taus(key, N, p, sigma, d, alphas, n_real, seed_base):
    """Mean T over n_real networks at once, for every alpha (nets reused)."""
    Q = Q_cache.setdefault(N, tang_basis(N))
    sums = np.zeros(len(alphas))
    cnts = np.zeros(len(alphas))

    for rep in range(n_real):
        A, B, _ = utils.GENERATORS[key](
            N, p, sigma, seed=seed_base + rep
        )

        for ia, a in enumerate(alphas):
            J = build_J_M(A, B, a, N, d)
            t = measure_tau(J, Q)

            if not np.isnan(t):
                sums[ia] += t
                cnts[ia] += 1

    return [
        sums[i] / cnts[i] if cnts[i] else np.nan
        for i in range(len(alphas))
    ]


fig, axes = plt.subplots(2, 3, figsize=(14, 8))
fig.subplots_adjust(
    wspace=0.30,
    hspace=0.40,
    left=0.06,
    right=0.98,
    top=0.95,
    bottom=0.09
)

# ============================================================
# Top row: T vs N
# ============================================================
top_curves = [[[] for _ in types] for _ in alphas]

for n in N_arr:
    Ni = int(n)

    for j, tp in enumerate(types):
        t = sim_taus(
            key_map[tp],
            Ni,
            p_fixed,
            sigma_fixed,
            d_fixed,
            alphas,
            n_real,
            seed_base
        )

        for ia in range(len(alphas)):
            top_curves[ia][j].append(t[ia])


for idx, av in enumerate(alphas):
    ax = axes[0, idx]

    for j, tp in enumerate(types):
        ax.plot(
            N_arr,
            top_curves[idx][j],
            color=colors[tp],
            ls=linestyles[tp],
            lw=2.2,
            label=tp
        )

    ax.set_xscale('log')
    ax.xaxis.set_major_locator(LogLocator(base=10, subs=[1, 2, 5]))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:g}'))

    ax.set_xlabel(r'Species richness, $N$', fontsize=20)

    if idx == 0:
        ax.set_ylabel(r'Recovery time, $T$', fontsize=20)

    ax.tick_params(
        labelsize=16,
        width=1.2,
        length=5
    )

    for spine in ax.spines.values():
        spine.set_linewidth(1.2)

    if idx == 2:
        ax.legend(
            fontsize=15,
            frameon=False,
            handlelength=2.8
        )


# ============================================================
# Bottom row: T vs p
# ============================================================
bot_curves = [[[] for _ in types] for _ in alphas]

for pv in p_arr:
    for j, tp in enumerate(types):
        t = sim_taus(
            key_map[tp],
            N_fixed,
            pv,
            sigma_fixed2,
            d_fixed2,
            alphas,
            n_real,
            seed_base
        )

        for ia in range(len(alphas)):
            bot_curves[ia][j].append(t[ia])


for idx, av in enumerate(alphas):
    ax = axes[1, idx]

    for j, tp in enumerate(types):
        ax.plot(
            p_arr,
            bot_curves[idx][j],
            color=colors[tp],
            ls=linestyles[tp],
            lw=2.2,
            label=tp
        )

    ax.set_xlabel(r'Connectivity, $p$', fontsize=20)

    if idx == 0:
        ax.set_ylabel(r'Recovery time, $T$', fontsize=20)

    ax.tick_params(
        labelsize=16,
        width=1.2,
        length=5
    )

    for spine in ax.spines.values():
        spine.set_linewidth(1.2)

    if idx == 2:
        ax.legend(
            fontsize=15,
            frameon=False,
            handlelength=2.8
        )


fig.savefig(
    '/Users/xichenli/venv/bin/高阶加速恢复/fig5_tau_vs_N_and_p_sim-1.pdf',
    dpi=600,
    bbox_inches='tight'
)

plt.close(fig)

print("Done: fig5_tau_vs_N_and_p_sim-1.pdf")