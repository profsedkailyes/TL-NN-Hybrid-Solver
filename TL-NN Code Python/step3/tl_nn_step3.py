# tl_nn_step3.py
# ==========================================
# TL-NN ALGORITHM: STEP 3
# Block Matrix Assembly & Linear System Solving
# ==========================================
import numpy as np
import time

# ==========================================
# 1. PARAMETERS & FUNCTIONS (Copy from Step 1)
# ==========================================
TAU = 20.0
N   = 3       # Number of subintervals
n   = 5       # Collocation points per subinterval
LAMBDA = 10.0

def psi_ext(t):
    return 3.0 + np.sin(t) + np.cos(4.0 * t)

def kappa(t, s, psi):
    denom = (4.0 + np.sin(s) - 2.0 * np.sin(2.0 * s)**2)**2
    return (t + s)**3 * psi**2 / denom

def g(t):
    return LAMBDA * psi_ext(t) - 0.25 * ((t + TAU)**4 - t**4)

def d_kappa_d_psi(t, s, psi):
    """Derivative of kernel wrt psi (Eq 15)"""
    denom = 4.0 + np.sin(s) - 2.0 * np.sin(2.0 * s)**2
    return 2.0 * (t + s)**3 * psi / denom**2

# ==========================================
# 2. DOMAIN DECOMPOSITION & GRIDS (From Step 2)
# ==========================================
tau_bounds = np.linspace(0, TAU, N + 1)
omega_n = (TAU / N) / (n - 1)

# Store grids for each subinterval l=0..N-1
local_grids = []
for l in range(N):
    t_l = np.linspace(tau_bounds[l], tau_bounds[l+1], n)
    local_grids.append(t_l)

# ==========================================
# 3. MATRIX ASSEMBLY: H_NN (Eq. 23)
# ==========================================
print("⏳ Assembling Global Matrix H_NN ...")
size_global = N * n
H_NN = np.zeros((size_global, size_global))

# Loop over Block Rows (l) and Block Cols (p)
for l in range(N):       # 0 to N-1 (Python indexing)
    for p in range(N):   # 0 to N-1
        
        # Extract grids for current block (l, p)
        # l is the t-dimension (row), p is the s-dimension (col)
        t_grid_l = local_grids[l] 
        s_grid_p = local_grids[p]
        
        # Create 2D mesh for vectorization
        T, S = np.meshgrid(t_grid_l, s_grid_p, indexing='ij')
        
        # Evaluate Psi at collocation nodes s (from the exact solution for testing)
        # In the full algorithm, this comes from the previous iteration Y(k)
        psi_nodes = psi_ext(S) 
        
        # Compute Kernel Derivative D(t, s) over the whole block at once
        D_block = d_kappa_d_psi(T, S, psi_nodes)
        
        # Apply quadrature weight ω_n
        # H[l, p](i, j) = ω_n * D(t_i, s_j)
        H_block = omega_n * D_block
        
        # Map to Global Matrix using slicing
        row_slice = slice(l * n, (l + 1) * n)
        col_slice = slice(p * n, (p + 1) * n)
        
        H_NN[row_slice, col_slice] = H_block

# ==========================================
# 4. BUILD THE SYSTEM MATRIX A = λI - H (Eq. 24)
# ==========================================
# Identity matrix of size (N*n x N*n)
I_NN = np.eye(size_global)

# The system matrix: A * Y = b
A_NN = LAMBDA * I_NN - H_NN

print(f"✅ Matrix H_NN assembled. Shape: {H_NN.shape}")
print(f"✅ System Matrix A_NN assembled. Shape: {A_NN.shape}")

# ==========================================
# 5. BUILD RHS VECTOR b (Eq. 25)
# ==========================================
# b depends on the current iteration's solution. 
# We use psi_ext here just to test the solver mechanism.
print("⏳ Assembling RHS Vector b ...")
b_NN = np.zeros(size_global)

for l in range(N):
    t_grid_l = local_grids[l]
    for i in range(n): # Local row index
        t_val = t_grid_l[i]
        
        # Eq 25 requires summing integrals (Nyström approximation)
        # b = - sum(sum(omega * D * psi)) + sum(integral(Kappa)) + g
        
        # This is a simplified version for the solver test. 
        # Full implementation requires the integral approximation loop.
        # For now, we just fill with g(t) to verify the linear solver works.
        # In Step 5 we will complete the full b vector logic.
        
        global_idx = l * n + i
        b_NN[global_idx] = g(t_val)

# ==========================================
# 6. SOLVE THE LINEAR SYSTEM
# ==========================================
print("⏳ Solving linear system (λI - H)Y = b ...")
t_start = time.perf_counter()

# Use NumPy's optimized solver
Y_solution = np.linalg.solve(A_NN, b_NN)

t_end = time.perf_counter()
print(f"✅ System Solved in {(t_end - t_start)*1000:.2f} ms")

# ==========================================
# 7. VERIFY & VISUALIZE
# ==========================================
# Reshape solution to match (N, n) structure for easy viewing
Y_matrix = Y_solution.reshape(N, n)

print("\n📊 Solution Y (First few values):")
print(f"Block 0 (t ∈ [0, 6.66]): {Y_matrix[0, :]}")
print(f"Block 1 (t ∈ [6.66, 13.33]): {Y_matrix[1, :]}")
print(f"Block 2 (t ∈ [13.33, 20.0]): {Y_matrix[2, :]}")

# Verification: Check if A @ Y ≈ b
residual = np.linalg.norm(A_NN @ Y_solution - b_NN)
print(f"\n🔍 Verification ||Ay - b||: {residual:.2e} (Should be ~1e-12 or smaller)")