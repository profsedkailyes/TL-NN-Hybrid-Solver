# tl_nn_step1.py
import numpy as np

# -------------------------
# 1. Parameters (from paper)
# -------------------------
LAMBDA = 10.0
TAU    = 20.0

# -------------------------
# 2. Mathematical Functions
# -------------------------
def psi_ext(t):
    """Exact solution for Example 1: ψ_ext(t) = 3 + sin(t) + cos(4t)"""
    return 3.0 + np.sin(t) + np.cos(4.0 * t)

def g(t, lam=LAMBDA, tau=TAU):
    """Right-hand side function g(t) for Example 1"""
    return lam * psi_ext(t) - 0.25 * ((t + tau)**4 - t**4)

def kappa(t, s, psi):
    """
    Nonlinear kernel κ(t, s, ψ) for Example 1:
    κ(t,s,ψ) = (t+s)^3 * ψ^2 / (4 + sin(s) - 2*sin^2(2s))^2
    """
    numerator   = (t + s)**3 * psi**2
    denominator = (4.0 + np.sin(s) - 2.0 * np.sin(2.0 * s)**2)**2
    return numerator / denominator

def residual(t):
    ''' function that computes |λψ_ext(t) - ∫₀^τ κ(t,s,ψ_ext(s))ds - g(t)|'''
    s_grid = np.linspace(0, TAU, 1000)
    psi_grid = psi_ext(s_grid)
    kappa_grid = kappa(t, s_grid, psi_grid)
    integral_approx = np.trapz(kappa_grid, s_grid)
    resudal_val = abs(LAMBDA * psi_ext(t) - integral_approx - g(t_val))
    return resudal_val


# -------------------------
# 3. Quick Validation
# -------------------------
if __name__ == "__main__":
    t_val = 1.5
    s_val = 2.0
    
    # Evaluate exact solution at s
    psi_s = psi_ext(s_val)
    
    # Evaluate kernel at (t, s, ψ(s))
    k_val = kappa(t_val, s_val, psi_s)

    # computes the residual value in t_val piont
    residual_val = residual(t_val)
    
    print(f"✅ ψ_ext({s_val}) = {psi_s:.6f}")
    print(f"✅ κ({t_val}, {s_val}, ψ_ext({s_val})) = {k_val:.6f}")
    print(f"✅ resudual(t_val) = {residual_val:.6f}")


