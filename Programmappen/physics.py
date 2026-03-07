import numpy as np
from scipy.interpolate import CubicSpline
import os

# ---------------- Physical / model constants ----------------
SIGMA = 5.67e-8                     # Stefan-Boltzmann (W/m²/K⁴)
RHO_WATER = 999                    # Density of water (kg/m³)
C_P_WATER = 4186                   # Specific heat capacity of water (J/kg/K)
C_M = C_P_WATER * RHO_WATER        # Heat capacity per meter depth of water (J m⁻² K⁻¹ / m)
SECONDS_PER_YEAR = 365 * 24 * 3600
R_EARTH = 6.371e6                   # m

# greenhouse offset parameters
DELTA_T0 = 33.1                  # δT at reference
T00 = 287.5                      # K (reference)
DELTA_T_MIN = 10.0               # K (lower bound)

# amplitude of initial temperature profile (eq. 17)
A_PROFILE = 45.0                 # K

# short description of the default parameters
PARAM_DESCS = {"k1": "ice temperature sensitivity", "k2": "diffusivity sensitivity", "k3": "longwave radiation sensitivity",
               "D0": "background diffusivity", "T0": "initial temperature", "SD": "mixed layer sea depth",
               "S0": "solar forcing initial", "S1": "solar forcing changed", "F": "additional forcing"}

# Wrapper to add model and i as optional input but ignore them (many functions must accept model and i even if unused)
def Input(func):
    """Decorator to add model and i as optional parameters to physics functions that do not need them."""
    return lambda *args, model=None, i=None, **kwargs: func(*args, **kwargs)

# ---------------- Tridiagonal solver ----------------
def thomas_solve(a, b, c, d): #If to slow, replace with scipy.linalg.solve_banded
    """
    Solve tridiagonal system Ax = d with A defined by diagonals a,b,c using Thomas algorithm.

    Parameters
    ----------
    a: ndarray
        lower diagonal (length n but a[0] unused)
    b: ndarray
        main diagonal (length n)
    c: ndarray
        upper diagonal (length n but c[-1] unused)
    d: ndarray
        right-hand side (length n)
    
    Returns
    -------
    x: ndarray
        solution vector (length n)
    
    Notes
    -----
    Implements the Thomas algorithm (specialized Gaussian elimination for tridiagonal matrices).
    """
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
    """
    Initial temperature profile (in K) as function of x = sin(lat).

    Parameters
    ----------
    x : array
        Grid points (sin(latitude)).
    T0 : float
        Reference temperature (K).
    
    Returns
    -------
    T : array
        Initial temperature profile (K) at each grid point.
    
    Notes
    -----
    - Uses a simple parabolic profile centered at equator.
    - Equation (17) in the documentation.
    - Global mean temperature is T0.
    """
    return T0 + A_PROFILE * (1/3 - x**2)

@Input
def Q_x(x, S):
    """
    Annual-mean insolation (TOA) as function of x = sin(lat).

    Parameters
    ----------
    x : array
        Grid points (sin(latitude)).
    S : float
        Solar constant / insolation parameter.
    
    Returns
    -------
    Q : array
        Annual-mean insolation (TOA) at each grid point.
    
    Notes
    -----
    - First equation on page 6 of the documentation.
    - Uses zonal averaging (integral over longitude cell) and annual averaging over time.
    """
    dx = x[1] - x[0]
    x_left = x - 0.5 * dx
    x_right = x + 0.5 * dx
    return 0.25 * S * (1.0 - 0.241 * (x_right**3 - x_left**3 - (x_right - x_left)) / dx)

def seasonal_Q(x, S, model, i):
    """
    Returns Solar Irradiance taking the earth inclination and season into account. See Wikipedia \"Solar irradiance\" for more info.

    Parameters
    ----------
    x : array
        Grid points (sin(latitude)).
    S : float
        Solar constant / insolation parameter.
    model : ClimateModel
        Model object containing configuration parameters.
    i : int
        Time step index.
    
    Returns
    -------
    SIr : array
        Solar irradiance at each grid point.

    Notes
    -----
    - Uses the current time step to calculate the declination of the earth and thus the insolation at each latitude.
    - Based on the formula: SIr = (S / π) [h0 sin(φ) sin(δ) + cos(φ) cos(δ) sin(h0)]
      where h0 is the hour angle at sunrise/sunset, δ is the declination, and φ is the latitude.
    - The declination δ varies throughout the year due to the tilt of the Earth's axis (obliquity). The formula used is:
      δ = sin⁻¹(sin(ε) sin(θ))
      where ε is the obliquity (approximately 23.44°) and θ is the annual angle (2πt, with t in years).
    - See https://en.wikipedia.org/wiki/Solar_irradiance for more details.
    """
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
    """
    Equation (12): effective albedo with ice fraction fraction_i = k1*(273-T) clipped to [0,1].
    
    Parameters
    ----------
    T : array
        Surface temperature (K) at each grid point.
    x : array
        Grid points (sin(latitude)).
    k1 : float
        Ice fraction coefficient.

    Returns
    -------
    alpha : array
        Effective albedo at each grid point.
    """
    alpha_a = 0.2 + 0.08 * x**2
    fraction_i = np.clip(k1 * (273.0 - T), 0.0, 1.0)
    alpha_s = 0.60 * fraction_i + (1.0 - fraction_i) * (0.1 + 0.15 * x**4)
    A_a = 0.32 * (1.0 - 0.85 * x**2)
    alpha = alpha_a + alpha_s - alpha_a * alpha_s - A_a * alpha_s
    return np.minimum(alpha, 0.7)

@Input
def diffusion_from_T(T, D0, k2, mean=True):
    """
    Equation (15): meridional diffusion coefficient D(T) = D0 * max[0.5, 1 + k2*(T - T00)].

    Parameters
    ----------
    T : array
        Surface temperature (K) at each grid point.
    D0 : float
        Base diffusion coefficient.
    k2 : float
        Temperature dependence coefficient.
    mean : bool, optional
        If True, use the mean temperature for diffusion calculation (default is True).

    Returns
    -------
    D : array or float
        Meridional diffusion coefficient at each grid point or a single value if mean is True.
    """
    T = T.mean() if mean else T
    return D0 * np.maximum(0.5, 1.0 + k2 * (T - T00))

@Input
def deltaT_of_Ts(Ts, k3):
    """
    Equation (13) for greenhouse effect: δT(Ts) = DELTA_T0 + k3 (Ts - T00), with lower bound.

    Parameters
    ----------
    Ts : array
        Surface temperature (K) at each grid point.
    k3 : float
        Greenhouse effect coefficient.

    Returns
    -------
    deltaT : array
        Temperature offset due to greenhouse effect at each grid point.
    """
    return np.maximum(DELTA_T0 + k3 * (Ts - T00), DELTA_T_MIN)

def heat_capacity_profile(x, T, k1):
    """
    Compute latitude-dependent effective heat capacities for a sin(lat)-spaced EBM grid.
    
    Parameters
    ----------
    x : array
        Grid points (sin(latitude)).
    T : array
        Surface temperature (K) at each grid point.
    k1 : float
        Ice fraction coefficient.
    
    Returns
    -------
    C : ndarray
        Heat capacity per unit area [J m^-2 K^-1] at each gridpoint.
    
    Notes
    -----
    - Uses an asymmetric zonal ocean fraction ocean_fraction(lat) (Southern Hemisphere more ocean).
    - Uses a piecewise-interpolated ocean mixed-layer depth h_ocean(lat) (seasonal-scale).
    - Land is treated as a shallow water equivalent of h_land (default 8 m).
    - Ice fraction is a smooth function of T: full ice when T <= -16.66 C°, none when T >= 0 C° when k1=0.06.
    
    Documentation / rationale:
    - Mixed-layer climatologies show shallow MLD in tropics (~ tens m), shallow subtropical
        stratified minima, deeper seasonal mixing in mid-latitudes (~ 100-300 m), and large
        seasonal deepening at high latitudes (Monterey & Levitus 1997; de Boyer Montégut 2004).
    - Southern Hemisphere has greater ocean fraction than Northern Hemisphere; we reflect
        that asymmetry in ocean_fraction(lat).
    """
    # physical constants
    rho = RHO_WATER     # kg/m^3
    cp = C_P_WATER      # J/kg/K
    h_land = 8.0        # m, land equivalent (water-equivalent), ~1/30 of deep-ocean reference
    
    # latitudes in degrees from x = sin(lat)
    lat_deg = np.degrees(np.arcsin(x))
    
    # --- Ocean mixed-layer depth profile (seasonal-scale, more nuance) ---
    # Explanation:
    #  - tropical/ITCZ: shallow seasonal MLD (20-70 m)
    #  - subtropical stratified belts: local shallow minimum (20-40 m)
    #  - mid-latitudes: more storm-driven deepening (100-250 m)
    #  - high-latitude (poleward of ~70): variable, seasonal deepening possible (50-200 m)
    # These numbers are chosen to represent the *seasonal* MLD climatology (not deep ocean).
    latitudes = np.array([-90, -70, -50, -30, -15, 0, 15, 30, 50, 70, 90])
    # more oceanic south: allow slightly deeper high-latitude southern mixing (southern storms)
    # depths [m]
    mixed_layer_depths = np.array([100.0,   # near South Pole: if ice-free, deep seasonal mixing possible; otherwise will be masked by ice
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
    
    # Interpolate mixed-layer depth onto grid
    interpolater = CubicSpline(latitudes, mixed_layer_depths, bc_type='clamped')
    h_ocean = interpolater(lat_deg)

    # --- Asymmetric zonal ocean fraction profile ---
    # Load ocean fraction data from CSV file (5-degree resolution)
    data = np.loadtxt(os.path.join(os.path.dirname(__file__), 'Datafiler/ocean_fraction_by_latitude_5deg.csv'), delimiter=",", skiprows=1)
    latitudes_f = data[:,0]
    ocean_fraction_knots = data[:,1]
    
    # Interpolate ocean fraction onto grid
    interpolater = CubicSpline(latitudes_f, ocean_fraction_knots, bc_type='clamped')
    ocean_fraction = np.clip(interpolater(lat_deg), 0.0, 1.0)

    # --- Ice fraction (same formula as in albedo) ---
    # Ice fraction in [0,1]: 1 => full ice cover (ice covered ocean behaves like land)
    ice_fraction = np.clip(k1 * (273.15 - T), 0.0, 1.0)
    # Effective ocean fraction after accounting for seasonal/persistent ice
    ocean_fraction_eff = ocean_fraction * (1.0 - ice_fraction)
    land_fraction_eff  = 1.0 - ocean_fraction_eff # Ice like land is assumed to act like h_land water depth
    
    # Effective depth: ocean fraction uses h_ocean, land fraction uses h_land
    h_eff = ocean_fraction_eff * h_ocean + land_fraction_eff * h_land
    
    # Heat capacity per unit area
    C = rho * cp * h_eff   # J m^-2 K^-1
    
    return C

def Forcing(model, i):
    """
    Apply constant external forcing F after control period.

    Parameters
    ----------
    model : ClimateModel
        Model object containing configuration parameters.
    i : int
        Current time step index.

    Returns
    -------
    F : float
        External radiative forcing at current time step. Is zero during control period.
    """
    return model.params['F'] * (i >= model.ctrl_nsteps)  # Step function forcing after control period

def VariableForcing(model, i):
    """
    Apply time-varying external forcing from pre-loaded forcing history.

    Parameters
    ----------
    model : ClimateModel
        Model object containing configuration parameters.
    i : int
        Current time step index.

    Returns
    -------
    F : float
        External radiative forcing at current time step.
    
    Notes
    -----
    - Uses model.F_History which should be set up by VariableForcing mode.
    """
    return model.F_History[i] # Use time-varying forcing from forcing history data

def build_diffusion_tridiag(x, D):
    """
    Build tridiagonal representation of diffusion operator L with diffusivity D (W m⁻² K⁻¹) on borders with nx cell points at x = sin(lat).

    Parameters
    ----------
    x : array
        Grid points (sin(latitude)).
    D : array or float
        Meridional diffusion coefficient at each grid point or a single value.

    Returns
    -------
    a : ndarray
        Lower diagonal of tridiagonal operator L.
    b : ndarray
        Main diagonal of tridiagonal operator L.
    c : ndarray
        Upper diagonal of tridiagonal operator L.
        
    Notes
    -----
    - The (linear) operator L approximates the term ∂x[D(1-x²) ∂x] using finite differences on a grid in x = sin(latitude).
      This operator, which happens to be tridiagonal, is used in the Crank-Nicolson scheme for meridional diffusion in the climate model.
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
    """
    Apply diffusion (tri-diagonal) operator L defined by diagonals a, b, c to temperature profile T.

    Parameters
    ----------
    a : ndarray
        Lower diagonal of tridiagonal operator L.
    b : ndarray
        Main diagonal of tridiagonal operator L.
    c : ndarray
        Upper diagonal of tridiagonal operator L.
    T : ndarray
        Temperature profile at cell centers.

    Returns
    -------
    out : ndarray
        Result of applying operator L to temperature profile T.
    
    Notes
    -----
    - Computes out = L(T) using the tridiagonal structure.
    - See build_diffusion_tridiag() for construction of L.
    """
    out = b * T
    out[1:]  += a[1:] * T[:-1]   # lower diagonal contribution
    out[:-1] += c[:-1] * T[1:]   # upper diagonal contribution
    return out

# ---------------- Diagnostics helpers ----------------
@Input
def meridional_transport_PW(T, x, D):
    """
    Calculate meridional heat transport HMTrans (PW = 10¹⁵ W) at boundaries from temperature profile T (K) at x = sin(lat) with diffusivity D (W m⁻² K⁻¹).
    
    Parameters
    ----------
    T : ndarray
        Temperature profile at cell centers.
    x : ndarray
        Grid points (sin(latitude)).
    D : ndarray or float
        Meridional diffusion coefficient at each grid point or a single value.

    Returns
    -------
    latitudes : ndarray
        Latitudes corresponding to the cell boundaries (degrees).
    MHTrans : ndarray
        Meridional heat transport at boundaries (PW).
    """
    # dTdx = np.gradient(T, x)
    dTdx = np.r_[0, (T[1:] - T[:-1]) / (x[1] - x[0]), 0]
    x_borders = np.r_[-1, (x[1:] + x[:-1]) / 2, 1]
    D = np.r_[D[0], (D[1:] + D[:-1]) / 2, D[-1]] if isinstance(D, np.ndarray) else D # Not entirely correct but ok approximation for diffusivity at borders
    flux = - D * (1.0 - x_borders**2) * dTdx                  # W/m² (per-area heat flux)
    MHTrans = 2.0 * np.pi * R_EARTH**2 * flux                 # W (zonal integral around latitude circle)
    return np.arcsin(x_borders) / np.pi * 180, MHTrans / 1e15 # PW

@Input
def poles_temperature(T):
    """
    Return temperature at poles (K) by extrapolation since T is at cell centers. Extrapolates by a quadratic fit with 0 gradient at poles.
    
    Parameters
    ----------
    T : ndarray
        Temperature profile at cell centers.
        
    Returns
    -------
    T_south : float
        Temperature at South Pole (K).
    T_north : float
        Temperature at North Pole (K).
    """
    return (9*T[0] - T[1]) / 8.0, (9*T[-1] - T[-2]) / 8.0 # Formula can be easily found be Taylor expansion (error is O(dx^3) actually O(dx^4) since function is even)
