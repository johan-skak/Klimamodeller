import numpy as np
from scipy.interpolate import CubicSpline


# ---------------- Physical / model constants ----------------
SIGMA = 5.67e-8                  # Stefan-Boltzmann (W/m²/K⁴)
C_M = 4184 * 999                   # Heat capacity per meter depth of water (J m⁻² K⁻¹ / m)
SECONDS_PER_YEAR = 365 * 24 * 3600
R_EARTH = 6.371e6                # m

# greenhouse offset parameters
DELTA_T0 = 33.1                  # δT at reference
T00 = 287.5                      # K (reference)
DELTA_T_MIN = 10.0               # K (lower bound)

# initial profile amplitude (eq. 17)
A_PROFILE = 45.0                 # K

# short description of the default parameters
PARAM_DESCS = {"k1": "ice temperature sensitivity", "k2": "diffusivity sensitivity", "k3": "longwave radiation sensitivity",
               "D0": "background diffusivity", "T0": "initial temperature", "SD": "mixed layer sea depth",
               "S0": "solar forcing initial", "S1": "solar forcing changed", "F": "additional forcing"}

# Wrapper to add model and i as optional input but ignore them
def Input(func):
    return lambda *args, model=None, i=None, **kwargs: func(*args, **kwargs)

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
@Input
def T_init(x, T0):
    """Initial temperature profile (K) as function of x = sin(lat)."""
    return T0 + A_PROFILE * (1/3 - x**2)

@Input
def Q_x(x, S):
    """Annual-mean insolation (TOA) as function of x = sin(lat)."""
    dx = x[1] - x[0]
    x_left = x - 0.5 * dx
    x_right = x + 0.5 * dx
    return 0.25 * S * (1.0 - 0.241 * (x_right**3 - x_left**3 - (x_right - x_left)) / dx)

def seasonal_Q(x, S, model, i):
    """Returns Solar Irradiance taking the earth inclination and season into account. See Wikipedia \"Irradiance\" for more info."""
    t = (i+1) * model.config["dt_years"]  # time in years
    eps = np.deg2rad(23.44) # obliquity
    theta = 2 * np.pi * t   # annual angle
    delta = np.arcsin(np.sin(eps) * np.sin(theta)) # Current declination δ = sin⁻¹(sin ε sin θ)

    # Hour angle at sunrise/sunset
    h0 = np.arccos(np.clip(- x / np.sqrt(1 - x**2 + 1e-10) * np.tan(delta), -1, 1))
    SIr = (S / np.pi) * (h0 * x * np.sin(delta) + np.sqrt(1 - x**2) * np.cos(delta) * np.sin(h0))
    return SIr

@Input
def albedo_from_T(T, x, k1):
    """Equation (12): effective albedo with ice fraction f_i = k1*(273-T) clipped to [0,1]."""
    alpha_a = 0.2 + 0.08 * x**2
    f_i = np.clip(k1 * (273.0 - T), 0.0, 1.0)
    alpha_s = 0.60 * f_i + (1.0 - f_i) * (0.1 + 0.15 * x**4)
    A_a = 0.32 * (1.0 - 0.85 * x**2)
    alpha = alpha_a + alpha_s - alpha_a * alpha_s - A_a * alpha_s
    return np.minimum(alpha, 0.7)

@Input
def diffusion_from_T(T, D0, k2, mean=True):
    T = T.mean() if mean else T
    return D0 * np.maximum(0.5, 1.0 + k2 * (T - T00))

@Input
def deltaT_of_Ts(Ts, k3):
    """Equation (13): δT(Ts) = DELTA_T0 + k3 (Ts - T00), with lower bound."""
    return np.maximum(DELTA_T0 + k3 * (Ts - T00), DELTA_T_MIN)

def heat_capacity_profile(x, T, k1):
    """
    Compute latitude-dependent effective heat capacities for a sin(lat)-spaced EBM grid.
    
    Parameters
    ----------
    nx : int
        Number of gridpoints (sin(lat) spaced from -1 to 1).
    T : array_like, shape (nx,)
        Surface temperature [K] at each gridpoint (used to estimate ice fraction).
    
    Returns
    -------
    C : ndarray, shape (nx,)
        Heat capacity per unit area [J m^-2 K^-1] at each gridpoint.
    
    Notes
    -----
    - Uses an asymmetric zonal ocean fraction f_ocean(lat) (Southern Hemisphere more ocean).
    - Uses a piecewise-interpolated ocean mixed-layer depth h_ocean(lat) (seasonal-scale).
    - Land is treated as a shallow water equivalent of h_land (default 8 m).
    - Ice fraction is a smooth function of T: full ice when T <= 271 K, none when T >= 275 K.
    
    Physical constants:
    rho = 1000 kg/m^3, cp = 4186 J/kg/K (water)
    
    Documentation / rationale:
    - Mixed-layer climatologies show shallow MLD in tropics (~tens m), shallow subtropical
        stratified minima, deeper seasonal mixing in mid-latitudes (~100-300 m), and large
        seasonal deepening at high latitudes (Monterey & Levitus 1997; de Boyer Montégut 2004).
        See references below. :contentReference[oaicite:0]{index=0}
    - Southern Hemisphere has greater ocean fraction than Northern Hemisphere; we reflect
        that asymmetry in f_ocean(lat). :contentReference[oaicite:1]{index=1}
    """    
    # physical constants
    rho = 1000.0        # kg/m^3
    cp = 4186.0         # J/kg/K
    h_land = 8.0        # m, land equivalent (water-equivalent), ~1/30 of deep-ocean reference
    
    # latitude in degrees from x = sin(lat)
    lat_deg = np.degrees(np.arcsin(x))
    
    # --------------------------
    # Ocean mixed-layer depth profile (seasonal-scale, more nuance)
    # knots (latitudes in degrees) and representative depths (m)
    # Explanation:
    #  - tropical/ITCZ: shallow seasonal MLD (20-70 m)
    #  - subtropical stratified belts: local shallow minimum (20-40 m)
    #  - mid-latitudes: more storm-driven deepening (100-250 m)
    #  - high-latitude (poleward of ~70): variable, seasonal deepening possible (50-200 m)
    # These numbers are chosen to represent the *seasonal* MLD climatology (not deep ocean).
    lat_knots = np.array([-90, -70, -50, -30, -15, 0, 15, 30, 50, 70, 90])
    # more oceanic south: allow slightly deeper high-latitude southern mixing (southern storms)
    # depths [m]
    h_knots = np.array([100.0,   # near South Pole: if ice-free, deep seasonal mixing possible; otherwise will be masked by ice
                        180.0,   # 70S - Southern high lat deep mixing in winter (Southern Ocean)
                        220.0,   # 50S - stormier, deeper seasonal ML
                        140.0,   # 30S - mid to subtropics
                        60.0,    # 15S - subtropical shoal
                        50.0,    # 0   - tropics (warm shallow seasonal ML)
                        60.0,    # 15N - slightly larger than equator
                        40.0,    # 30N - subtropical stratified minimum
                        140.0,   # 50N - northern mid-latitude seasonal deepening (but less than SH)
                        80.0,   # 70N - Arctic ocean (shallower than SH)
                        30.0])   # near North Pole (shallow seasonal ML do to fresh water)
    
    # interpolate mixed-layer depth onto grid
    interpolater = CubicSpline(lat_knots, h_knots, bc_type='clamped')
    h_ocean = interpolater(lat_deg)

    # # --------------------------
    # # Asymmetric zonal ocean fraction f_ocean(lat)
    # # Hard-coded (smooth) zonal ocean fraction knots. Values based on general land/ocean
    # # geography: Southern hemisphere has more ocean (esp. 30S-60S), Northern hemisphere has
    # # more land at mid-latitudes (Eurasia, North America). These are smooth, empirical values.
    # # Source: qualitative/quantitative zonal land fraction diagrams (e.g. land-fraction vs latitude).
    # lat_knots_f = np.array([-90, -70, -50, -30, -15, 0, 15, 30, 50, 70, 90])
    # # zonal ocean fraction at knots (0..1). Southern hemisphere has systematically larger ocean fraction.
    # f_ocean_knots = np.array([0.0,  # Continent/ice cap
    #                         0.75,   # 70S -> Mostly Antarctic land, but surrounding Southern Ocean begins
    #                         0.95,   # 50S -> Southern Ocean dominates
    #                         0.8,  # 30S -> Still ocean-dominated, only S. America, Africa, Australia
    #                         0.8,   # 15S -> Mostly ocean
    #                         0.75,  # 0   -> Continents cut across (Africa, S. America, Indonesia)
    #                         0.75,   # 15N -> Africa + Asia reduce ocean fraction
    #                         0.55,   # 30N -> Subtropics: Africa, Asia, N. America
    #                         0.4,  # 50N -> Eurasia + N. America dominate, but N. Atlantic/Pacific present
    #                         0.65,  # 70N -> Arctic Ocean exists, though partly enclosed
    #                         1.0])  # Central Arctic Ocean basin
    data = np.loadtxt("Datafiler/ocean_fraction_by_latitude_5deg.csv", delimiter=",", skiprows=1)
    lat_knots_f = data[:,0]
    f_ocean_knots = data[:,1]
    
    # interpolate ocean fraction onto grid
    interpolater = CubicSpline(lat_knots_f, f_ocean_knots, bc_type='clamped')
    f_ocean = np.clip(interpolater(lat_deg), 0.0, 1.0)

    # --------------------------
    # --- Ice fraction (same formula as in albedo) ---
    # ice_fraction in [0,1]: 1 => full ice cover (ocean behaves like land shallow)
    ice_fraction = np.clip(k1 * (273.15 - T), 0.0, 1.0)
    # effective ocean fraction after accounting for seasonal/persistent ice
    f_ocean_eff = f_ocean * (1.0 - ice_fraction)
    f_land_eff  = 1.0 - f_ocean_eff #Ice like land is assumed to act like h_land water depth
    
    # --------------------------
    # Effective depth: ocean fraction uses h_ocean, land fraction uses h_land
    h_eff = f_ocean_eff * h_ocean + f_land_eff * h_land
    
    # --------------------------
    # Heat capacity per unit area
    C = rho * cp * h_eff   # J m^-2 K^-1
    
    return C

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
    D = 0.5 * (D[:-1] + D[1:]) if isinstance(D, np.ndarray) else D
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
@Input
def meridional_transport_PW(T, x, D):
    """Calculate meridional heat transport HMTrans (PW = 10¹⁵ W) at boundaries from temperature profile T (K) at x = sin(lat) with diffusivity D (W m⁻² K⁻¹)."""
    # dTdx = np.gradient(T, x)
    dTdx = np.r_[0, (T[1:] - T[:-1]) / (x[1] - x[0]), 0]
    x_borders = np.r_[-1, (x[1:] + x[:-1]) / 2, 1]
    D = np.r_[D[0], (D[1:] + D[:-1]) / 2, D[-1]] if isinstance(D, np.ndarray) else D # Not entirely correct but ok approximation
    flux = - D * (1.0 - x_borders**2) * dTdx                    # W/m² (per-area heat flux)
    MHTrans = 2.0 * np.pi * R_EARTH**2 * flux                 # W (zonal integral around latitude circle)
    return np.arcsin(x_borders) / np.pi * 180, MHTrans / 1e15                            # PW

@Input
def poles_temperature(T):
    """Return temperature at poles (K) by extrapolation since T is at cell centers. Extrapolates by a quadratic fit with 0 gradient at poles.
    
    Parameters
        T: array of temperatures at cell centers
        
    Returns
        (T_south, T_north) temperatures at south and north poles (K)"""
    return (9*T[0] - T[1]) / 8.0, (9*T[-1] - T[-2]) / 8.0 # Formula can be easily found be Taylor expansion (error is O(dx^3) actually O(dx^4) since function is even)
