import numpy as np

# ---------------- Physical / model constants ----------------
SIGMA = 5.67e-8                  # Stefan-Boltzmann (W/m²/K⁴)
C = 1.046e9                      # Heat capacity (J m⁻² K⁻¹), 250 m mixed layer
SECONDS_PER_YEAR = 365 * 24 * 3600
R_EARTH = 6.371e6                # m

# greenhouse offset parameters
DELTA_T0 = 33.1                  # δT at reference
T00 = 287.5                      # K (reference)
DELTA_T_MIN = 10.0               # K (lower bound)

# initial profile amplitude (eq. 17)
A_PROFILE = 45.0                 # K

# ---------------- Tridiagonal solver ----------------
def thomas_solve(a, b, c, d): #If to slow, replace with scipy.linalg.solve_banded
    """Solve tridiagonal system Ax = d with A defined by diagonals a,b,c using Thomas algorithm.
    a: lower diagonal (length n but a[0] unused)
    b: main diagonal (length n)
    c: upper diagonal (length n but c[-1] unused)
    d: right-hand side (length n)"""
    n = len(d)
    ac, bc, cc, dc = map(np.array, (a, b, c, d))  # make copies to store modified coefficients
    # Forward elimination: modify coefficients
    for i in range(1, n):
        mc = ac[i] / bc[i-1]
        bc[i] -= mc * cc[i-1]
        dc[i] -= mc * dc[i-1]

    # Backward substitution: solve for solution vector
    x = np.zeros(n)
    x[-1] = dc[-1] / bc[-1]
    for i in range(n-2, -1, -1):
        x[i] = (dc[i] - cc[i] * x[i+1]) / bc[i]
    return x

# ---------------- Physics building blocks (PDF exact forms) ----------------
def T_init(x, T0):
    """Initial temperature profile (K) as function of x = sin(lat)."""
    return T0 + A_PROFILE * (1/3 - x**2)

def Q_x(x, S):
    """Annual-mean insolation (TOA) as function of x = sin(lat)."""
    dx = x[1] - x[0]
    x_left = x - 0.5 * dx
    x_right = x + 0.5 * dx
    return 0.25 * S * (1.0 - 0.241 * (x_right**3 - x_left**3 - (x_right - x_left)) / dx)

def albedo_from_T(T, x, k1):
    """Equation (12): effective albedo with ice fraction f_i = k1*(273-T) clipped to [0,1]."""
    alpha_a = 0.2 + 0.08 * x**2
    f_i = np.clip(k1 * (273.0 - T), 0.0, 1.0)
    alpha_s = 0.60 * f_i + (1.0 - f_i) * (0.1 + 0.15 * x**4)
    A_a = 0.32 * (1.0 - 0.85 * x**2)
    alpha = alpha_a + alpha_s - alpha_a * alpha_s - A_a * alpha_s
    return np.minimum(alpha, 0.7)

def deltaT_of_Ts(Ts, k3):
    """Equation (13): δT(Ts) = DELTA_T0 + k3 (Ts - T00), with lower bound."""
    return np.maximum(DELTA_T0 + k3 * (Ts - T00), DELTA_T_MIN)

# ---------------- Diffusion operator L ≈ ∂x[D(1-x²) ∂x] ----------------
def build_diffusion_tridiag(x, D):
    """Build tridiagonal representation of diffusion operator L with diffusivity D (W m⁻² K⁻¹) on borders with nx cell points at x = sin(lat).

    Parameters
        nx: number of cell points
        x: array of sin(lat) at cell centers (linear spacing)
        D: diffusivity (W m⁻² K⁻¹)

    Returns
        a, b, c: lower, main and upper diagonals of L (arrays of length nx)
            a[0] and c[-1] are unused (=0)
    """
    dx = x[1] - x[0]
    a, b, c = np.zeros_like(x), np.zeros_like(x), np.zeros_like(x)  # diagonals

    # half-point weights (length nx-1)
    x_half = 0.5 * (x[:-1] + x[1:])
    w_half = D * (1.0 - x_half**2) # diffusivity at cell borders
    
    # interior contributions
    a[1:] = w_half / dx**2     # lower diagonal
    c[:-1] = w_half / dx**2    # upper diagonal
    b = -(a + c)               # main diagonal
    return a, b, c

def apply_L_to_T(a, b, c, T):
    """Apply diffusion (tri-diagonal) operator L defined by diagonals a,b,c to temperature profile T."""
    out = b * T
    out[1:]  += a[1:] * T[:-1]   # lower diagonal contribution
    out[:-1] += c[:-1] * T[1:]   # upper diagonal contribution
    return out


# ---------------- Diagnostics helpers ----------------
def meridional_transport_PW(T, x, D):
    """Calculate meridional heat transport HMTrans (PW = 10¹⁵ W) at boundaries from temperature profile T (K) at x = sin(lat) with diffusivity D (W m⁻² K⁻¹)."""
    # dTdx = np.gradient(T, x)
    dTdx = np.r_[0, (T[1:] - T[:-1]) / (x[1] - x[0]), 0]
    x_borders = np.r_[-1, (x[1:] + x[:-1]) / 2, 1]
    flux = - D * (1.0 - x_borders**2) * dTdx                    # W/m² (per-area heat flux)
    HMTrans = - 2.0 * np.pi * R_EARTH**2 * flux                 # W (zonal integral around latitude circle)
    return np.arcsin(x_borders) / np.pi * 180, HMTrans / 1e15                            # PW

def global_mean(T):
    return np.mean(T)

def poles_temperature(T):
    """Return temperature at poles (K) by extrapolation since T is at cell centers. Extrapolates by a quadratic fit with 0 gradient at poles.
    
    Parameters
        T: array of temperatures at cell centers
        
    Returns
        (T_south, T_north) temperatures at south and north poles (K)"""
    return (9*T[0] - T[1]) / 8.0, (9*T[-1] - T[-2]) / 8.0 # Formula can be easily found be Taylor expansion (error is O(dx^3) actually O(dx^4) since function is even)
