"""
Consolidated utility functions for higher-order ecological networks.
Covers: random, predator-prey, mutualistic, mixed interactions.

All formulas match the corrected paper derivations with exact finite-N factors.
"""
import numpy as np
from scipy.linalg import eigvals

# ============================================================
# 1. Shared network generation
# ============================================================

def generate_E_er_binary(N, p, seed=None, directed=False):
    """ER random graph adjacency (0/1), no self-loops."""
    rng = np.random.RandomState(seed) if seed is not None else np.random
    E = np.zeros((N, N), dtype=int)
    if directed:
        for i in range(N):
            for j in range(N):
                if i == j: continue
                E[i, j] = 1 if rng.rand() < p else 0
    else:
        for i in range(N):
            for j in range(i + 1, N):
                if rng.rand() < p:
                    E[i, j] = 1
                    E[j, i] = 1
    return E


def generate_T_undirected(E, min_edges_in_triple=2):
    """Triplet indicator T_ijk for all ordered triples with distinct indices."""
    N = E.shape[0]
    T = np.zeros((N, N, N), dtype=int)
    adj = ((E != 0) | (E.T != 0)).astype(int)
    for i in range(N - 2):
        for j in range(i + 1, N - 1):
            for k in range(j + 1, N):
                edges = adj[i, j] + adj[j, k] + adj[i, k]
                if edges >= min_edges_in_triple:
                    for a in (i, j, k):
                        for b in (i, j, k):
                            for c in (i, j, k):
                                if a != b and b != c and a != c:
                                    T[a, b, c] = 1
    return T


# ============================================================
# 2. Random interactions
# ============================================================

def generate_random(N, p, sigma, seed=None):
    """Generate A (pairwise) and B (higher-order) for random interactions."""
    rng = np.random.RandomState(seed) if seed is not None else np.random
    E = generate_E_er_binary(N, p, seed=seed, directed=False)
    T = generate_T_undirected(E)
    A = E * rng.normal(0, sigma, size=(N, N))
    np.fill_diagonal(A, 0)
    B = np.zeros((N, N, N))
    for i in range(N):
        for j in range(N):
            if i == j: continue
            for k in range(N):
                if k == i or k == j: continue
                if T[i, j, k]:
                    B[i, j, k] = rng.normal(0, sigma)
    return A, B, E


def analytical_sigmaM_random(alpha, sigma, p, N):
    """σ_M² for random: Eq. (26). Returns (sigma_M2, sigma_M, rho_M=0)."""
    gamma = 2 * p**2 - p**3
    sigma_M2 = (1 - alpha)**2 * p * sigma**2 + (2 * alpha**2 / N) * gamma * sigma**2
    sigma_M2 = max(sigma_M2, 1e-30)
    sigma_M = np.sqrt(sigma_M2)
    return sigma_M2, sigma_M, 0.0


def compute_tau_random(alpha, sigma, p, N, d, return_growth=False):
    """Recovery time T for random interactions."""
    sigma_M2, sigma_M, _ = analytical_sigmaM_random(alpha, sigma, p, N)
    lam = np.sqrt(N) * sigma_M / N - d
    if lam < 0:
        return 1.0 / (-lam)
    return (1.0 / lam) if return_growth else np.nan


# ============================================================
# 3. Predator-prey interactions
# ============================================================

def generate_predprey(N, p, sigma, seed=None):
    """Generate A, B for predator-prey (E_ij = -E_ji, antisymmetric)."""
    rng = np.random.RandomState(seed) if seed is not None else np.random
    # Antisymmetric adjacency: E_ij = ±1, E_ji = ∓1
    E = np.zeros((N, N), dtype=int)
    for i in range(N):
        for j in range(i + 1, N):
            if rng.rand() < p:
                s = 1 if rng.rand() < 0.5 else -1
                E[i, j] = s
                E[j, i] = -s
    T = generate_T_undirected(E)
    A = E * np.abs(rng.normal(0, sigma, size=(N, N)))
    np.fill_diagonal(A, 0)
    # HO modulation only on predation edges (E_ij = 1)
    B = np.zeros((N, N, N))
    Z = rng.normal(0, sigma, size=(N, N, N))
    for i in range(N):
        for j in range(N):
            if i == j or E[i, j] != 1: continue
            for k in range(N):
                if k == i or k == j: continue
                if T[i, j, k]:
                    B[i, j, k] = Z[i, j, k]
    return A, B, E


def analytical_sigmaM_rhoM_predprey(alpha, sigma, p, N):
    """σ_M² and ρ_M for predator-prey. Eqs. (28)-(29)."""
    gamma_pp = p**2 - p**3 / 2
    sigma_M2 = (1 - alpha)**2 * p * sigma**2 + (2 * alpha**2 / N) * gamma_pp * sigma**2
    sigma_M2 = max(sigma_M2, 1e-30)
    cov = -(1 - alpha)**2 * p * sigma**2 * (2.0 / np.pi)
    rho_M = cov / sigma_M2
    sigma_M = np.sqrt(sigma_M2)
    return sigma_M2, sigma_M, rho_M


def compute_tau_predprey(alpha, sigma, p, N, d):
    """Recovery time T for predator-prey."""
    sigma_M2, sigma_M, rho_M = analytical_sigmaM_rhoM_predprey(alpha, sigma, p, N)
    lam = np.sqrt(N) * sigma_M * (1 + rho_M) / N - d
    if lam < 0:
        return 1.0 / (-lam)
    return np.nan


# ============================================================
# 4. Mutualistic interactions (corrected, exact finite-N)
# ============================================================

def generate_mutual(N, p, sigma, seed=None):
    """Generate A, B for mutualistic (half-normal, undirected)."""
    rng = np.random.RandomState(seed) if seed is not None else np.random
    E = generate_E_er_binary(N, p, seed=seed, directed=False)
    T = generate_T_undirected(E)
    X_raw = np.abs(rng.normal(0, sigma, size=(N, N)))
    np.fill_diagonal(X_raw, 0)
    A = E * X_raw
    B = np.zeros((N, N, N))
    Z = np.abs(rng.normal(0, sigma, size=(N, N, N)))
    for i in range(N):
        for j in range(N):
            if i == j: continue
            for k in range(N):
                if k == i or k == j: continue
                if T[i, j, k]:
                    B[i, j, k] = E[i, j] * Z[i, j, k]
    return A, B, E


def analytical_sigmaM_rhoM_mutual(alpha, sigma, p, N):
    """
    σ_M² and ρ_M for mutualistic — corrected formulas matching paper.
    Includes exact (N-1)/N factors and all cross-covariance channels.
    """
    mu = sigma * np.sqrt(2.0 / np.pi)
    mu2 = mu**2  # = 2σ²/π
    Nf = float(N)
    q = 2.0 * p - p**2
    gamma = p * q

    # ---- σ_M² ----
    # Part 1: pairwise
    part1 = (1 - alpha)**2 * (p * sigma**2 - p**2 * mu2)

    # Part 2: higher-order
    # 2a: Var(∑B_ijk)
    var_B = gamma * sigma**2 - gamma**2 * mu2
    cov_B_kl = p * (1 - p) * q**2 * mu2
    var_sum1 = Nf * var_B + Nf * (Nf - 1) * cov_B_kl

    # 2b: Var(∑B_ikj)
    cov_B2_kl = p**3 * (1 - p)**3 * mu2           # corrected
    var_sum2 = Nf * var_B + Nf * (Nf - 1) * cov_B2_kl

    # 2c: Cov(∑B_ijk, ∑B_ikj)
    cov_eq = p**2 * (1 - q**2) * mu2               # k = l
    cov_ne = p**2 * q * (1 - q) * mu2              # k ≠ l
    cov_cross = Nf * cov_eq + Nf * (Nf - 1) * cov_ne

    part2 = (alpha**2 / Nf**2) * (var_sum1 + var_sum2 + 2 * cov_cross)

    # Part 3: pairwise-HO cross
    cov_A_Bijk = p * (1 - p) * q * mu2
    cov_A_Bikj = p**2 * (1 - q) * mu2              # corrected (not equal)
    part3 = 2 * (1 - alpha) * alpha * (cov_A_Bijk + cov_A_Bikj)

    sigma_M2 = part1 + part2 + part3
    sigma_M2 = max(sigma_M2, 1e-30)

    # ---- ρ_M numerator (Cov(M̃_ij, M̃_ji)) ----
    # Part 1
    cov1 = (1 - alpha)**2 * p * (1 - p) * mu2

    # Part 2+3
    cov23 = 2 * (1 - alpha) * alpha * (cov_A_Bijk + cov_A_Bikj)

    # Part 4: HO-HO
    cov_C1_eq = gamma * (1 - gamma) * mu2
    cov_C1_ne = p * q**2 * (1 - p) * mu2
    C1 = Nf * cov_C1_eq + Nf * (Nf - 1) * cov_C1_ne

    cov_C2_eq = (p**2 - gamma**2) * mu2
    cov_C2_ne = p**2 * q * (1 - q) * mu2
    C2 = Nf * cov_C2_eq + Nf * (Nf - 1) * cov_C2_ne
    C3 = C2                                       # symmetric

    cov_C4_eq = (p**2 - gamma**2) * mu2
    cov_C4_ne = p**3 * (1 - p)**3 * mu2
    C4 = Nf * cov_C4_eq + Nf * (Nf - 1) * cov_C4_ne

    part4 = (alpha**2 / Nf**2) * (C1 + C2 + C3 + C4)

    numerator = cov1 + cov23 + part4
    rho_M = numerator / sigma_M2

    sigma_M = np.sqrt(sigma_M2)
    return sigma_M2, sigma_M, rho_M


def compute_tau_mutual(alpha, sigma, p, N, d):
    """Recovery time T for mutualistic."""
    sigma_M2, sigma_M, rho_M = analytical_sigmaM_rhoM_mutual(alpha, sigma, p, N)
    lam = np.sqrt(N) * sigma_M * (1 + rho_M) / N - d
    if lam < 0:
        return 1.0 / (-lam)
    return np.nan


# ============================================================
# 5. Mixed interactions (corrected, exact finite-N)
# ============================================================

def generate_mixed(N, p, sigma, seed=None):
    """Generate A, B for mixed (sign-symmetric E, half-normal weights)."""
    rng = np.random.RandomState(seed) if seed is not None else np.random
    mask_triu = np.triu(rng.rand(N, N) < p, 1)
    signs = rng.choice([1, -1], size=(N, N))
    E = (mask_triu.astype(float) * signs)
    E = E + E.T
    X_raw = np.abs(rng.normal(0, sigma, size=(N, N)))
    np.fill_diagonal(X_raw, 0)
    A = E * X_raw
    C = (E != 0)
    Z = np.abs(rng.normal(0, sigma, size=(N, N, N)))
    C_ik = C[:, np.newaxis, :]
    C_jk = C[np.newaxis, :, :]
    T_bool = np.logical_or(C_ik, C_jk)
    B = E[:, :, np.newaxis] * T_bool * Z
    return A, B, E


def analytical_sigmaM_rhoM_mixed(alpha, sigma, p, N):
    """
    σ_M² and ρ_M for mixed — corrected formulas matching paper.
    E[E_ij]=0 eliminates mean-subtractions and cross-covariances with E_ij E_ik.
    """
    mu = sigma * np.sqrt(2.0 / np.pi)
    mu2 = mu**2
    Nf = float(N)
    q = 2.0 * p - p**2
    gamma = p * q

    # ---- σ_M² ----
    # Part 1: pairwise (E[E]=0, no mean correction)
    part1 = (1 - alpha)**2 * p * sigma**2

    # Part 2: higher-order
    var_B = gamma * sigma**2                        # no mean correction
    cov_B_kl = p * q**2 * mu2                       # k≠l, only within ∑B_ijk
    var_sum1 = Nf * var_B + Nf * (Nf - 1) * cov_B_kl
    var_sum2 = Nf * var_B                            # ∑B_ikj: no k≠l cov
    # Cov(∑B_ijk, ∑B_ikj) = 0 (E[E_ij E_ik] = 0)
    part2 = (alpha**2 / Nf**2) * (var_sum1 + var_sum2)

    # Part 3: pairwise-HO cross (only E²_ij survives)
    cov_A_Bijk = p * q * mu2
    cov_A_Bikj = 0.0                                 # E[E_ij E_ik] = 0
    part3 = 2 * (1 - alpha) * alpha * (cov_A_Bijk + cov_A_Bikj)

    sigma_M2 = part1 + part2 + part3
    sigma_M2 = max(sigma_M2, 1e-30)

    # ---- ρ_M numerator ----
    # Part 1: Cov(A_ij, A_ji) = p μ²
    cov1 = (1 - alpha)**2 * p * mu2

    # Part 2+3: A-HO cross
    cov23 = 2 * (1 - alpha) * alpha * p * q * mu2    # only Cov(A_ij, B_jik)

    # Part 4: HO-HO
    # C1: Cov(∑B_ijk, ∑B_jik)
    cov_C1_eq = gamma * mu2                           # no mean correction
    cov_C1_ne = p * q**2 * mu2
    C1 = Nf * cov_C1_eq + Nf * (Nf - 1) * cov_C1_ne

    # C2: Cov(∑B_ijk, ∑B_jki), C3, C4 — all vanish at O(1) for mixed
    # (E[E_ij E_jk] = 0 for j≠k, and E[E_ik E_jk] = 0, etc.)
    # Only O(1/N) k=l terms survive
    cov_C2_eq = p**2 * mu2
    C2 = Nf * cov_C2_eq                             # C2 from ∑_k B_ijk ↔ ∑_k B_jki, only k=l
    C3 = Nf * cov_C2_eq                             # symmetric
    C4 = Nf * cov_C2_eq                             # same pattern

    part4 = (alpha**2 / Nf**2) * (C1 + C2 + C3 + C4)

    numerator = cov1 + cov23 + part4
    rho_M = numerator / sigma_M2

    sigma_M = np.sqrt(sigma_M2)
    return sigma_M2, sigma_M, rho_M


def compute_tau_mixed(alpha, sigma, p, N, d):
    """Recovery time T for mixed."""
    sigma_M2, sigma_M, rho_M = analytical_sigmaM_rhoM_mixed(alpha, sigma, p, N)
    lam = np.sqrt(N) * sigma_M * (1 + rho_M) / N - d
    if lam < 0:
        return 1.0 / (-lam)
    return np.nan


# ============================================================
# 6. Unified dispatch
# ============================================================

def generate(tp, N, p, sigma, seed=None):
    """Unified generator for any interaction type."""
    key = _normalize_key(tp)
    if key in GENERATORS:
        return GENERATORS[key](N, p, sigma, seed=seed)
    raise ValueError(f"Unknown type: {tp}")

# Map type names to functions
GENERATORS = {
    'random':       generate_random,
    'predprey':     generate_predprey,
    'mutual':       generate_mutual,
    'mixed':        generate_mixed,
}

ANALYTICAL = {
    'random':       analytical_sigmaM_random,
    'predprey':     analytical_sigmaM_rhoM_predprey,
    'mutual':       analytical_sigmaM_rhoM_mutual,
    'mixed':        analytical_sigmaM_rhoM_mixed,
}

TAU = {
    'random':       compute_tau_random,
    'predprey':     compute_tau_predprey,
    'mutual':       compute_tau_mutual,
    'mixed':        compute_tau_mixed,
}


def _normalize_key(tp):
    """Normalize type name to internal key: 'Random' → 'random', 'Predator-prey' → 'predprey', etc."""
    key = tp.lower().replace('-', '').replace(' ', '')
    # Handle common aliases
    aliases = {
        'random': 'random', 'predatorprey': 'predprey',
        'mutualistic': 'mutual', 'mutual': 'mutual',
        'mixed': 'mixed',
    }
    return aliases.get(key, key)


def compute_tau(tp, alpha, sigma, p, N, d):
    """Unified recovery time T for any interaction type."""
    key = _normalize_key(tp)
    if key in TAU:
        return TAU[key](alpha, sigma, p, N, d)
    raise ValueError(f"Unknown type: {tp}")


def analytical_sigmaM_rhoM(tp, alpha, sigma, p, N):
    """Unified σ_M², σ_M, ρ_M for any interaction type."""
    key = _normalize_key(tp)
    if key in ANALYTICAL:
        return ANALYTICAL[key](alpha, sigma, p, N)
    raise ValueError(f"Unknown type: {tp}")
