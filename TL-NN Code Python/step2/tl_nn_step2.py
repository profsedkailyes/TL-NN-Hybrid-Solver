# tl_nn_step2.py
# ==========================================
# TL-NN ALGORITHM: STEP 2
# Grid Generation & Vectorized Kernel Evaluation
# ==========================================
import numpy as np
import time

# ==========================================
# 1. PAPER NOTATION & PARAMETERS
# ==========================================
# Large interval: [0, τ]  (Eq. 1 in TL-NN / Eq. 2.1 in TLD)
TAU = 20.0

# N: Number of subintervals for domain decomposition (Transform phase)
N = 3

# n: Number of collocation points per subinterval (Discretization phase)
n = 5

# ==========================================
# 2. DOMAIN DECOMPOSITION (Eq. 2)
# ==========================================
# τ_p = p * τ / N, for p = 0, ..., N
# These are the boundaries of the N subintervals.
# In Python (0-based): tau_bounds[p] corresponds to τ_p in the paper.
tau_bounds = np.linspace(0, TAU, N + 1)  # Shape: (N+1,)

# ==========================================
# 3. COLLOCATION GRIDS Ω_l^n (Eq. 2.5)
# ==========================================
# Step size within each subinterval:
# ω_n = (τ_l - τ_{l-1}) / (n - 1) = τ / (N * (n - 1))
# Note: The paper writes ω_n = 1/(n-1) assuming unit-length subintervals.
# We use the actual geometric step size for correct quadrature.
omega_n = (TAU / N) / (n - 1)

# Generate grids for each subinterval l = 1, ..., N
# Ω_l^n = {t_{l,i} | i = 1, ..., n}
# t_{l,i} = τ_{l-1} + (i-1) * ω_n
# We store them in a list: grids[l-1] corresponds to Ω_l^n
grids = []
for l in range(1, N + 1):
    t_l = np.linspace(tau_bounds[l-1], tau_bounds[l], n)
    grids.append(t_l)

# ==========================================
# 4. EXACT SOLUTION & KERNEL DERIVATIVE (Example 1)
# ==========================================
def psi_ext(t):
    """Exact solution ψ_ext(t) = 3 + sin(t) + cos(4t)"""
    return 3.0 + np.sin(t) + np.cos(4.0 * t)

def d_kappa_d_psi(t, s, psi):
    """
    Fréchet derivative ∂κ/∂ψ for Example 1:
    κ(t,s,ψ) = (t+s)^3 ψ^2 / (4 + sin(s) - 2 sin^2(2s))^2
    ⇒ ∂κ/∂ψ = 2(t+s)^3 ψ / (4 + sin(s) - 2 sin^2(2s))^2
    
    This corresponds to D^{(k)}_{lp}(t, s) in Eq. (15) of the TL-NN paper.
    """
    denom = 4.0 + np.sin(s) - 2.0 * np.sin(2.0 * s)**2
    return 2.0 * (t + s)**3 * psi / denom**2

# ==========================================
# 5. VECTORIZED EVALUATION FOR BLOCK (l, p)
# ==========================================
# Pick a specific block pair (l, p) to demonstrate.
# Paper notation: l ∈ {1,...,N}, p ∈ {1,...,N}
l_block, p_block = 2, 1

# Extract local grids Ω_l^n and Ω_p^n
# Python uses 0-based indexing: grids[l_block-1] → Ω_{l_block}^n
t_grid_l = grids[l_block - 1]  # t_{l,1}, ..., t_{l,n}
s_grid_p = grids[p_block - 1]  # t_{p,1}, ..., t_{p,n}

# Create 2D coordinate arrays for vectorized evaluation.
# indexing='ij' ensures:
#   T[i, j] = t_{l, i+1}  (row index i corresponds to local collocation index i)
#   S[i, j] = t_{p, j+1}  (col index j corresponds to local collocation index j)
# This exactly matches the matrix layout H_{lp}(i, j) in Eq. (12) & (23).
T, S = np.meshgrid(t_grid_l, s_grid_p, indexing='ij')

# Evaluate target function D^{(k)}_{lp}(t, s) on the entire n×n grid at once.
# No Python loops are used. NumPy broadcasts the operation across all (i,j) pairs.
psi_s_grid = psi_ext(S)
D_block = d_kappa_d_psi(T, S, psi_s_grid)  # Shape: (n, n)

# ==========================================
# 6. VERIFICATION & INDEX MAPPING (Bridge to Step 3)
# ==========================================
print(f"📐 Domain: [0, {TAU}] split into N={N} subintervals")
print(f"📏 Boundaries τ_p: {tau_bounds}")
print(f"📊 Collocation step ω_n = {omega_n:.6f}")
print(f"🎯 Block (l={l_block}, p={p_block})")
print(f"   Ω_{l_block}^n = {t_grid_l}")
print(f"   Ω_{p_block}^n = {s_grid_p}")
print(f"   D_block.shape = {D_block.shape}")
print(f"   D_block[0,0] = {D_block[0,0]:.6f} → (t={t_grid_l[0]:.4f}, s={s_grid_p[0]:.4f})")
print(f"   D_block[-1,-1] = {D_block[-1,-1]:.6f} → (t={t_grid_l[-1]:.4f}, s={s_grid_p[-1]:.4f})")

# Prepare global index mapping for Step 3 (Matrix Assembly)
# Global row for local (l, i): R = n*(l-1) + (i-1)
# Global col for local (p, j): C = n*(p-1) + (j-1)
row_start = n * (l_block - 1)
col_start = n * (p_block - 1)
print(f"\n🔗 Global matrix indices for this block (0-based Python):")
print(f"   Rows: {row_start} to {row_start + n - 1}")
print(f"   Cols: {col_start} to {col_start + n - 1}")
print(f"   In Step 3, we will assign: H_NN[row_start:row_start+n, col_start:col_start+n] = ω_n * D_block")

# ==========================================
# 7. SPEED TEST (Vectorized vs Loop)
# ==========================================
def slow_loop_eval(t_arr, s_arr):
    result = np.zeros((len(t_arr), len(s_arr)))
    for i in range(len(t_arr)):
        for j in range(len(s_arr)):
            result[i, j] = d_kappa_d_psi(t_arr[i], s_arr[j], psi_ext(s_arr[j]))
    return result

t1 = time.perf_counter()
D_vec = d_kappa_d_psi(T, S, psi_ext(S))
t2 = time.perf_counter()
D_loop = slow_loop_eval(t_grid_l, s_grid_p)
t3 = time.perf_counter()

print(f"\n⚡ Vectorized time: {(t2-t1)*1000:.3f} ms")
print(f"🐌 Python loop time: {(t3-t2)*1000:.3f} ms")
print(f"✅ Results match: {np.allclose(D_vec, D_loop)}")