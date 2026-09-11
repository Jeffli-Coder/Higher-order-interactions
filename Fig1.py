"""
Repeated perturbations under pairwise vs higher-order interactions.
2×2 layout (kept as before): top row x_i(t), bottom row ||x - x*||.
Every panel gets its own x- and y-labels; square axes; enlarged fonts
for cross-column downscaling in a larger composite figure.

Formatting follows PRE_figure_style_guide.md; fonts further enlarged for
composite-figure downscaling. Original: fig1_disaster_and_callback.py.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'mathtext.fontset': 'cm',
    'font.size': 18,
    'axes.labelsize': 24,
    'axes.titlesize': 22,
    'xtick.labelsize': 18,
    'ytick.labelsize': 18,
    'legend.fontsize': 17,
    'axes.linewidth': 1.2,
    'lines.linewidth': 2.2,
    'xtick.major.size': 5,
    'ytick.major.size': 5,
    'xtick.major.width': 1.2,
    'ytick.major.width': 1.2,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.top': True,
    'ytick.right': True,
    'axes.grid': False,
    'legend.frameon': False,
    'legend.handlelength': 2.8,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})

N = 50; p = 0.5; sigma = 0.8; seed = 42
R0 = np.sqrt(N)*sigma*np.sqrt(p)
d = R0/N + 0.003                      # near stability boundary
t_pert_interval = 50
n_perts = 4
T_total = t_pert_interval * n_perts + 60   # full recovery tail
dt = 0.05

np.random.seed(seed)
E = np.zeros((N,N),dtype=int)
for i in range(N):
    for j in range(i+1,N):
        if np.random.rand() < p: E[i,j] = E[j,i] = 1
A = E * np.random.normal(0, sigma, size=(N,N))
import interaction_utils as iu
T_ijk = iu.triples_T_from_E(E)
B = np.zeros((N,N,N))
for i in range(N):
    for j in range(N):
        if i==j or E[i,j]==0: continue
        for k in range(N):
            if k==i or k==j: continue
            if T_ijk[i,j,k]: B[i,j,k] = np.random.normal(0, sigma)

def rhs(t, x, av):
    pair = A.dot(x); hoi = np.zeros(N)
    for i_ in range(N): hoi[i_]=np.sum(B[i_,:,:]*x[np.newaxis,:]*x[:,np.newaxis])
    f = (1-av)*pair + av*hoi; fm = np.dot(x,f)
    return x*(f-fm) - d*(x - 1.0/N)

def apply_perturb(x, seed_p, amp=0.12):
    rng = np.random.RandomState(seed_p)
    dx = amp*(rng.rand(len(x))-0.5); xp=x+dx
    xp=np.maximum(xp,0.001); return xp/xp.sum()*np.sum(x)

fig, axes = plt.subplots(2, 2, figsize=(12, 13))
fig.subplots_adjust(wspace=0.30, hspace=0.40, left=0.09, right=0.97,
                    top=0.88, bottom=0.08)

colors_ts = plt.cm.tab20(np.linspace(0,1,N))

for col, (av, title) in enumerate([(0.0, r'$\alpha=0$ (Pairwise)'),
                                     (0.7, r'$\alpha=0.7$ (Higher-order)')]):
    print(f"α={av} ...")
    xs = np.ones(N) / N           # reference equilibrium x* = 1/N

    x_cur = xs.copy()
    all_segs_t, all_segs_y = [], []
    for pi in range(n_perts):
        x_pert = apply_perturb(x_cur, 99+pi)
        T_seg = t_pert_interval if pi < n_perts-1 else t_pert_interval+60
        sol = solve_ivp(rhs,[0,T_seg],x_pert,args=(av,),
                        rtol=1e-10,atol=1e-12,t_eval=np.arange(0,T_seg,dt))
        x_cur = sol.y[:,-1]
        offset = pi*t_pert_interval
        all_segs_t.append(sol.t+offset)
        all_segs_y.append(sol.y)

    t_full = np.concatenate(all_segs_t); y_full = np.concatenate(all_segs_y, axis=1)

    # top row: time series
    ax_ts = axes[0,col]
    for i in range(N):
        ax_ts.plot(t_full, y_full[i], lw=2, color=colors_ts[i], alpha=0.6)
    for pi in range(1,n_perts):
        ax_ts.axvline(pi*t_pert_interval, color='#d7191c', lw=1.8, ls='--', alpha=0.5)
    ax_ts.axhline(np.mean(xs), color='gray', lw=0.7, ls=':', alpha=0.4)
    ax_ts.set_xlabel(r'$t$')
    ax_ts.set_ylabel(r'$x_i(t)$')
    ax_ts.set_title(title, fontsize=22, fontweight='bold')
    ax_ts.set_xlim(0, T_total-20)

    # bottom row: deviation ||x - x*||
    ax_dev = axes[1,col]
    dev = np.sqrt(np.mean((y_full - xs[:,np.newaxis])**2, axis=0))
    ax_dev.plot(t_full, dev, 'k-', lw=2.2)
    for pi in range(1,n_perts):
        ax_dev.axvline(pi*t_pert_interval, color='#d7191c', lw=1.8, ls='--', alpha=0.5)
    ax_dev.set_xlabel(r'$t$')
    ax_dev.set_ylabel(r'$\|\mathbf{x}(t)-\mathbf{x}^*\|$')
    ax_dev.set_xlim(0, T_total-20)
    ax_dev.set_ylim(0, 0.08)
    ax_dev.set_yticks(np.linspace(0, 0.08, 5))
    ax_dev.ticklabel_format(axis='y', style='plain')

for ax in axes.flat:
    ax.set_box_aspect(1)          # square axes for the composite row

fig.suptitle(r'Repeated perturbations: collapse under pairwise vs.\ survival under higher-order',
             fontsize=16, y=0.995)

base = '/Users/xichenli/venv/bin/高阶加速恢复/fig1_recovery_dynamics_v2'
plt.savefig(base + '.pdf', dpi=600, bbox_inches='tight')
plt.savefig(base + '.png', dpi=600, bbox_inches='tight')
plt.close()
print("Saved: fig1_recovery_dynamics_v2.pdf / .png")
