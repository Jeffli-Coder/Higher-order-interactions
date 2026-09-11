r"""
Eigenvalue spectrum validation of the effective-matrix approximation.
2-by-2 layout, PRE/Chaos single-column style.
Each panel: full Jacobian (blue) vs effective matrix (orange) + theory ellipse (black).
Vertical dashed lines mark the maximum real part of each simulated spectrum,
colored to match its scatter. For the mutualistic case the isolated Perron
(all-ones) outlier eigenvalue is excluded before taking the maximum real part.

Source: 7-by-7 in, scaled to ~3.3 in by LaTeX.
Axis labels 18pt, ticks 16pt, legend 14pt, panel labels 18pt bold.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.linalg import eigvals
import utils

# ==================== PRE single-column style ====================
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'mathtext.fontset': 'cm',
    'font.size': 16,
    'axes.labelsize': 18,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'legend.fontsize': 13,
    'axes.linewidth': 1.2,
    'xtick.major.width': 1.2,
    'ytick.major.width': 1.2,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.top': True,
    'ytick.right': True,
    'axes.grid': False,
    'legend.frameon': False,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})

# ==================== Parameters ====================
N = 100; p = 0.3; sigma = 0.5; alpha = 0.5; d = 15
repeats = 15; seed0 = 42
x_star = np.ones(N) / N

# ==================== Jacobian builders (specific to this validation) ====================
def build_J_full(A, B, alpha, N, x_star):
    xx = np.outer(x_star, x_star)
    t3 = -(1-alpha)*(A + A.T).dot(x_star)
    t4 = np.zeros(N)
    for j in range(N):
        t4[j] = -alpha*np.sum(xx*(B[j,:,:] + B[:,j,:] + B[:,:,j]))
    Jf = np.zeros((N,N))
    for i in range(N):
        B_i1 = B[i,:,:]
        B_i2 = np.transpose(B[i,:,:])
        t12 = (1-alpha)*A[i,:] + alpha*(B_i1 + B_i2).dot(x_star)
        Jf[i,:] = t12 + t3 + t4
        Jf[i,i] -= d
    return Jf

def build_M_simple(A, B, alpha, N):
    s1 = np.sum(B, axis=2)
    s2 = np.sum(np.transpose(B,(0,2,1)), axis=2)
    M = (1-alpha)*A + (alpha/N)*(s1+s2)
    np.fill_diagonal(M, np.diag(M)-d)
    return M

def eigvals_wo_outlier(M):
    """Eigenvalues of M with the isolated Perron (all-ones) outlier removed.

    For a non-negative mutualistic matrix the dominant eigenvalue is simple and
    aligned with the all-ones direction (Perron-Frobenius); identify it as the
    eigenvector with the largest overlap with ones(N)/sqrt(N).
    """
    N = M.shape[0]
    eigs, evecs = np.linalg.eig(M)
    u = np.ones(N) / np.sqrt(N)
    overlap = np.abs(evecs.T.conj().dot(u))
    idx = int(np.argmax(overlap))
    return np.delete(eigs, idx)

# ==================== Simulation ====================
type_keys = ['random', 'predprey', 'mutual', 'mixed']
type_labels  = ['a Random', 'b Predator-prey', 'c Mutualistic', 'd Mixed']

print("Simulating ...")
results = {}       # full spectrum, for the scatter
results_bulk = {}  # outlier-removed spectrum, for the max-Re annotation
for key in type_keys:
    evM, evJ = [], []
    bM, bJ = [], []
    for rep in range(repeats):
        A, B, _ = utils.GENERATORS[key](N, p, sigma, seed=seed0+rep)
        M = build_M_simple(A, B, alpha, N)
        J = build_J_full(A, B, alpha, N, x_star)
        evM.append(eigvals(M))
        evJ.append(eigvals(J))
        if key == 'mutual':
            bM.append(eigvals_wo_outlier(M))
            bJ.append(eigvals_wo_outlier(J))
    results[key] = (np.concatenate(evM), np.concatenate(evJ))
    if key == 'mutual':
        results_bulk[key] = (np.concatenate(bM), np.concatenate(bJ))
    print(f"  {key}: done")

# ==================== Plotting ====================
print("Plotting ...")
fig, axes = plt.subplots(2, 2, figsize=(7, 7))
fig.subplots_adjust(wspace=0.24, hspace=0.30,
                    left=0.11, right=0.97, top=0.96, bottom=0.09)

center = -d
all_a, all_b = [], []

# Theoretical ellipse dimensions
for key in type_keys:
    sigma_M2, sigma_M, rho_M = utils.ANALYTICAL[key](alpha, sigma, p, N)
    all_a.append(np.sqrt(N)*sigma_M*(1.0+rho_M))
    all_b.append(np.sqrt(N)*sigma_M*(1.0-rho_M))

max_a = max(all_a)
max_b = max(all_b)
# Plot
for ax, key, label in zip(axes.flat, type_keys, type_labels):
    evM, evJ = results[key]
    evM_ann, evJ_ann = results_bulk.get(key, (evM, evJ))  # outlier-free for mutual
    sigma_M2, sigma_M, rho_M = utils.ANALYTICAL[key](alpha, sigma, p, N)
    a = np.sqrt(N)*sigma_M*(1.0+rho_M)
    b = np.sqrt(N)*sigma_M*(1.0-rho_M)

    ax.scatter(np.real(evM), np.imag(evM), s=4.0, alpha=0.30,
               color='#d95f02', edgecolors='none', rasterized=True)
    ax.scatter(np.real(evJ), np.imag(evJ), s=4.0, alpha=0.30,
               color='#2c7bb6', edgecolors='none', rasterized=True)

    # Vertical dashed lines at the maximum real part of each spectrum
    # Maximum real part: Original system (solid)
    ax.axvline(
        np.max(np.real(evJ_ann)),
        color='#2c7bb6',
        lw=1.4,
        ls='-',
        alpha=0.85
    )

    # Maximum real part: Effective system (dashed)
    ax.axvline(
        np.max(np.real(evM_ann)),
        color='#d95f02',
        lw=1.4,
        ls='--',
        alpha=0.85
    )

    t = np.linspace(0, 2*np.pi, 400)
    ax.plot(center + a*np.cos(t), b*np.sin(t), 'k-', lw=2.2, alpha=0.80)
    ax.axvline(center, color='gray', lw=1.0, ls='--', alpha=0.45)

    le = [Line2D([0],[0], marker='o', color='w', markerfacecolor='#2c7bb6',
          markersize=7, label='Original system'),
          Line2D([0],[0], marker='o', color='w', markerfacecolor='#d95f02',
          markersize=7, label='Effective system')]
    ax.legend(handles=le, loc='upper left', fontsize=13,
              frameon=False, handlelength=0.8,
              borderpad=0.3, labelspacing=0.15)

    ax.text(-0.12, 1.04, label[0], transform=ax.transAxes,
            fontsize=18, fontweight='bold', va='bottom', ha='left')
    ax.text(0.50, 1.04, label[2:], transform=ax.transAxes,
            fontsize=14, fontweight='bold', va='bottom', ha='center')

    if ax in [axes[1,0], axes[1,1]]:
        ax.set_xlabel(r'Re($\lambda$)', labelpad=2)
    if ax in [axes[0,0], axes[1,0]]:
        ax.set_ylabel(r'Im($\lambda$)', labelpad=2)

    ax.tick_params(top=True, right=True, direction='in',
                   labeltop=False, labelright=False)

# Common axis limits
xpad = max(0.8, max_a*0.04)
ypad = max(0.8, max_b*0.04)
xlim = (center - max_a - xpad, center + max_a + xpad)
ylim = (-max_b - ypad, max_b + ypad)
for ax in axes.flat:
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect('equal', adjustable='box')

fig.savefig('/Users/xichenli/venv/bin/高阶加速恢复/fig2_eigenvalue_validation_maxreal.pdf',
            dpi=600, bbox_inches='tight')
plt.close(fig)
print("Saved: fig2_eigenvalue_validation_maxreal")
