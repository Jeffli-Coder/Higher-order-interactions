# Higher-Order Interactions Hasten Ecosystem Recovery — Figure Code

This directory contains the Python code that produces all figures of the paper
*Higher-Order Interactions Hasten the Recovery of Complex Ecosystems*.

The model is a replicator dynamics with pairwise and higher-order (third-body)
interactions,

```
dx_i/dt = x_i [ f_i(x) − Σ_j x_j f_j(x) ] − d (x_i − x_i*),
f_i(x)  = (1−α) Σ_j A_ij x_j + α Σ_jk B_ijk x_j x_k,
```

with interaction type entering through the statistics of `A` and `B`
(random, predator–prey, mutualistic/competitive, mixed). Recovery time is
obtained from the leading eigenvalue of the effective matrix
`J = (1−α)A + (α/N)(Σ_k B_ijk + Σ_k B_ikj) − dI` restricted to the
tangential (sum-zero) subspace.

## Files

| File | Purpose | Output |
|---|---|---|
| `Fig1.py` | Repeated-perturbation dynamics under pairwise vs higher-order interactions; 2×2 layout (top: species trajectories, bottom: distance to equilibrium) | `fig1_recovery_dynamics_v2.pdf/.png` |
| `Fig2.py` | Validation of the effective-matrix reduction: eigenvalue spectrum of the full Jacobian vs the effective matrix, with the theory (elliptic-law) boundary | `fig2_eigenvalue_validation_maxreal.pdf` |
| `Fig3.py` | Leading eigenvalue `max Re λ` vs higher-order mixing fraction `α`: analytical curve (solid) vs simulation error bars (dashed), for the four interaction types | `fig3_lambda_vs_alpha_errorbar.pdf` |
| `Fig4.py` | Recovery time `T` vs `α`: theory curves plus simulation error bars from the effective-matrix linearization | `fig4_tau_vs_alpha_linearized_M_sim-2.pdf` |
| `Fig5.py` | Recovery time `T` vs complexity (species richness `N` and connectance `p`) at `α = 0, 0.4, 0.9` (simulation only) | `fig5_tau_vs_N_and_p_sim-1.pdf` |
| `utils.py` | Shared utilities — network generators for the four interaction types and the analytical formulas for σ_M, ρ_M, and recovery time. **Do not rename.** | — |

## Dependencies

- Python 3
- NumPy
- SciPy (`scipy.linalg.eigvals`, `scipy.integrate.solve_ivp`)
- Matplotlib

## Usage

Run each script from this directory (it must find `utils.py` on the search
path):

```bash
python Fig1.py
python Fig2.py
python Fig3.py
python Fig4.py
python Fig5.py
```

Each script writes its PDF (and, for `Fig1.py`, a PNG) into this directory.

## Notes

- All scripts import `utils` from the same directory; keep the files together.
- `Fig5.py` builds the third-order tensor with a pure-Python O(N³) loop
  (N = 500 takes ≈ 55 s per network). A scan to N = 500 is therefore slow;
  lower `N_max` near the top of the script for quick tests.
- The `utils.py` analytical formulas use the exact finite-N factors from the
  paper's derivations, not a large-N approximation.
