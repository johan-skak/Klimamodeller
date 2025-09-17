#!/usr/bin/env python3

"""
Shell application version of the Kaas 1D Energy Balance Model (EBM).
Produces the SAME diagnostics as the Jupyter/Fortran versions:

Panel 1: initial, control and changed temperature profiles (°C)
Panel 2: OLR (W/m²)
Panel 3: Albedo
Panel 4: Meridional heat transport (PW)
Panel 5: Heat flux convergence (W/m²)
Panel 6: Change in zonal mean temperature (°C) and polar amplification printed
Panel 7: Time series of global mean temperature (°C)

Additionally prints and saves a summary to summary.txt.

Usage example:
  python ebm.py --input formoutput.txt --years_control 300 --years_forced 300 \
                    --nx 120 --dt 0.5 --outdir results
"""

import numpy as np
import matplotlib.pyplot as plt
import argparse, os, textwrap

# ---------------- Physical / model constants ----------------
sigma = 5.67e-8                  # Stefan-Boltzmann (W/m²/K⁴)
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
def thomas_solve(a, b, c, d):
    n = len(d)
    # Arrays to store modified coefficients
    super_diag_prime = np.zeros(n)
    rhs_prime = np.zeros(n)
    # Forward elimination: modify coefficients
    super_diag_prime[0] = c[0] / b[0]
    rhs_prime[0] = d[0] / b[0]
    for i in range(1, n):
        denom = b[i] - a[i] * super_diag_prime[i-1]
        # Avoid out-of-bounds for last super-diagonal element
        super_diag_prime[i] = c[i] / denom if i < n-1 else 0.0
        rhs_prime[i] = (d[i] - a[i] * rhs_prime[i-1]) / denom
    # Backward substitution: solve for solution vector
    sol = np.zeros(n)
    sol[-1] = rhs_prime[-1]
    for i in range(n-2, -1, -1):
        sol[i] = rhs_prime[i] - super_diag_prime[i] * sol[i+1]
    return sol

# ---------------- Physics building blocks (PDF exact forms) ----------------
def Q_x(x, S):
    """Annual-mean insolation (TOA) as function of x = sin(lat)."""
    dx = x[1] - x[0]
    x_left = x - 0.5 * dx
    x_right = x + 0.5 * dx
    return 0.25 * S * (1.0 - 0.241 * (x_right**3 - x_left**3 - (x_right - x_left)) / (dx))

def albedo_from_T(T, x, k1):
    """Equation (12): effective albedo with ice fraction fi = k1*(273-T) clipped to [0,1]."""
    alpha_a = 0.2 + 0.08 * x**2
    fi = np.clip(k1 * (273.0 - T), 0.0, 1.0)
    alpha_s = 0.60 * fi + (1.0 - fi) * (0.1 + 0.15 * x**4)
    Aa = 0.32 * (1.0 - 0.85 * x**2)
    alpha = alpha_a + alpha_s - alpha_a * alpha_s - Aa * alpha_s
    return np.minimum(alpha, 0.7)

def deltaT_of_Ts(Ts, k3):
    """Equation (13): δT(Ts) = DELTA_T0 + k3 (Ts - T00), with lower bound."""
    return np.maximum(DELTA_T0 + k3 * (Ts - T00), DELTA_T_MIN)

# ---------------- Diffusion operator L ≈ ∂x[D(1-x²) ∂x] ----------------
def build_diffusion_tridiag(nx, x, D):
    dx = x[1] - x[0]
    a = np.zeros(nx); b = np.zeros(nx); c = np.zeros(nx)
    x_half = 0.5 * (x[:-1] + x[1:]) #positions at cell faces except poles where flux=0
    w_half = D * (1.0 - x_half**2) # diffusivity at cell faces
    for i in range(nx):
        if i == 0:
            c[i] = w_half[i] / dx**2
            b[i] = -c[i]
        elif i == nx - 1:
            a[i] = w_half[i-1] / dx**2
            b[i] = -a[i]
        else:
            a[i] = w_half[i-1] / dx**2
            c[i] = w_half[i] / dx**2
            b[i] = -(a[i] + c[i])
    return a, b, c

def apply_L_to_T(a, b, c, T):
    n = len(T)
    out = np.zeros_like(T)
    for i in range(n):
        out[i] = b[i] * T[i]
        if i > 0:
            out[i] += a[i] * T[i-1]
        if i < n-1:
            out[i] += c[i] * T[i+1]
    return out

# ---------------- Diagnostics helpers ----------------
def meridional_transport_PW(T, x, D):
    dTdx = np.gradient(T, x)
    flux = - D * (1.0 - x**2) * dTdx         # W/m² (per-area heat flux)
    H = - 2.0 * np.pi * R_EARTH**2 * flux    # W (zonal integral around latitude circle)
    return H / 1e15                           # PW

def global_mean(T):
    return np.mean(T)

# ---------------- Single simulation (Crank–Nicolson for diffusion) -----------
def run_simulation(params, years, nx, dt_years, Tinit=None):
    dx = 2.0 / nx
    x = np.linspace(-1.0 + dx/2, 1.0 - dx/2, nx) # sin(lat) at cell centers

    if Tinit is None:
        T = params['T0'] + A_PROFILE * (1.0/3.0 - x**2)   # initial profile where center value approximates average cell value. Should be good enough only being initial values
    else:
        T = Tinit.copy()

    dt = dt_years * SECONDS_PER_YEAR
    nsteps = int(round(years / dt_years))

    Tg_series = []

    for _ in range(nsteps):
        # Explicit radiative terms
        Q = Q_x(x, params['S'])
        alpha = albedo_from_T(T, x, params['k1'])
        absorbed = Q * (1.0 - alpha)
        dTloc = deltaT_of_Ts(T, params['k3'])
        OLR = sigma * (T - dTloc)**4
        rad_term = absorbed - OLR + params['F']

        # Diffusivity depends on global mean temperature
        Tglob = global_mean(T)
        D = params['D0'] * max(0.5, 1.0 + params['k2'] * (Tglob - T00))

        # Build L and do Crank–Nicolson step
        aL, bL, cL = build_diffusion_tridiag(nx, x, D)
        LT = apply_L_to_T(aL, bL, cL, T)
        coef = dt / C
        rhs = T + 0.5 * coef * LT + (dt / C) * rad_term
        aA = -0.5 * coef * aL
        bA =  1.0 - 0.5 * coef * bL
        cA = -0.5 * coef * cL
        T = thomas_solve(aA, bA, cA, rhs)

        Tg_series.append(Tglob - 273.15)  # °C

    # End-state diagnostics
    alpha_end = albedo_from_T(T, x, params['k1'])
    dTloc_end = deltaT_of_Ts(T, params['k3'])
    OLR_end = sigma * (T - dTloc_end)**4
    D_end = params['D0'] * max(0.5, 1.0 + params['k2'] * (global_mean(T) - T00))
    aLe, bLe, cLe = build_diffusion_tridiag(nx, x, D_end)
    conv_end = apply_L_to_T(aLe, bLe, cLe, T)           # W/m² (convergence)
    H_end_PW = meridional_transport_PW(T, x, D_end)     # PW

    return dict(x=x, T=T, alpha=alpha_end, OLR=OLR_end, conv=conv_end,
                H_PW=H_end_PW, Tg=np.array(Tg_series), D_end=D_end)

# ---------------- Main program (control → forced) ----------------
def main():
    p = argparse.ArgumentParser(description='Kaas 1D EBM – shell diagnostics')
    p.add_argument('--input', type=str, default='formoutput.txt', help='parameter file (optional)')
    p.add_argument('--years_control', type=float, default=500)
    p.add_argument('--years_forced', type=float, default=500)
    p.add_argument('--nx', type=int, default=200)
    p.add_argument('--dt', type=float, default=1.0, help='timestep in years')
    p.add_argument('--outdir', type=str, default='results')
    args = p.parse_args()

    # Defaults (PDF/Table)
    base = dict(k1=0.06, k2=0.01, k3=0.5, D0=0.66, T0=288.0,
                S0=1365.0, S1=1365.0, F=0.0)

    # Try to read the earlier formoutput.txt structure if present
    if os.path.exists(args.input):
        with open(args.input) as f:
            lines = [l.strip() for l in f if l.strip()]
        if len(lines) >= 9:
            base['k1'] = float(lines[1]); base['k2'] = float(lines[2]); base['k3'] = float(lines[3])
            base['D0'] = float(lines[4]); base['T0'] = float(lines[5])
            base['S0'] = float(lines[6]); base['S1'] = float(lines[7]); base['F'] = float(lines[8])

    os.makedirs(args.outdir, exist_ok=True)

    # Print model parameters
    print(textwrap.dedent(f"""
    === EBM Model Parameters ===
    k1 (ice albedo sensitivity) \t: {base['k1']}
    k2 (diffusivity sensitivity)\t: {base['k2']}
    k3 (lapse rate sensitivity) \t: {base['k3']}
    D0 (background diffusivity) \t: {base['D0']}
    T0 (initial temperature)    \t: {base['T0']}
    S0 (initial solar forcing)  \t: {base['S0']}
    S1 (final solar forcing)    \t: {base['S1']}
    F  (additional forcing)     \t: {base['F']}
    ================================
    """))

    # ---- CONTROL RUN ----
    params_ctrl = dict(k1=base['k1'], k2=base['k2'], k3=base['k3'], D0=base['D0'], T0=base['T0'], S=base['S0'], F=0.0)
    ctrl = run_simulation(params_ctrl, years=args.years_control, nx=args.nx, dt_years=args.dt)

    # ---- FORCED RUN (continuation) ----
    params_forc = dict(k1=base['k1'], k2=base['k2'], k3=base['k3'], D0=base['D0'], T0=base['T0'], S=base['S1'], F=base['F'])
    forc = run_simulation(params_forc, years=args.years_forced, nx=args.nx, dt_years=args.dt, Tinit=ctrl['T'])

    # Common axes
    x = ctrl['x']; lat = np.degrees(np.arcsin(x))

    # Panel 6 quantities: Δ fields and polar amplification (EXACT as in Jupyter/Fortran here)
    dT_lat = (forc['T'] - ctrl['T'])             # K
    mean_ctrl = global_mean(ctrl['T'])
    mean_forc = global_mean(forc['T'])
    dT_global = (mean_forc - mean_ctrl)
    Ts_ctrl_pole = ctrl['T'][-1]
    Ts_forc_pole = forc['T'][-1]
    # Polar amplification per earlier implementation: (ΔT_pole - ΔT_global)/ΔT_global
    polar_ampl = np.nan
    if abs(dT_global) > 1e-12:
        polar_ampl = ((Ts_forc_pole - Ts_ctrl_pole) - dT_global) / dT_global

    # ---- Build multipanel figure ----
    fig, axs = plt.subplots(4, 2, figsize=(12, 14))
    axs = axs.flatten()

    # Panel 1: Temperature profiles (°C)
    T_init = params_ctrl['T0'] + A_PROFILE * (1.0/3.0 - x**2)
    axs[0].plot(lat, T_init - 273.15, label='Initial')
    axs[0].plot(lat, ctrl['T'] - 273.15, label='Control end')
    axs[0].plot(lat, forc['T'] - 273.15, label='Forced end')
    axs[0].set_title('Panel 1: Temperature profiles (°C)'); axs[0].set_xlabel('Latitude (°)'); axs[0].set_ylabel('°C'); axs[0].legend(); axs[0].grid(True)

    # Panel 2: OLR (W/m²)
    axs[1].plot(lat, ctrl['OLR'], label='Control')
    axs[1].plot(lat, forc['OLR'], label='Forced')
    axs[1].set_title('Panel 2: OLR (W/m²)'); axs[1].set_xlabel('Latitude (°)'); axs[1].set_ylabel('W/m²'); axs[1].legend(); axs[1].grid(True)

    # Panel 3: Albedo
    axs[2].plot(lat, ctrl['alpha'], label='Control')
    axs[2].plot(lat, forc['alpha'], label='Forced')
    axs[2].set_title('Panel 3: Albedo'); axs[2].set_xlabel('Latitude (°)'); axs[2].set_ylabel('albedo'); axs[2].legend(); axs[2].grid(True)

    # Panel 4: Meridional heat transport (PW)
    Hc = meridional_transport_PW(ctrl['T'], x, ctrl['D_end'])
    Hf = meridional_transport_PW(forc['T'], x, forc['D_end'])
    axs[3].plot(lat, Hc, label='Control')
    axs[3].plot(lat, Hf, label='Forced')
    axs[3].set_title('Panel 4: Meridional heat transport (PW)'); axs[3].set_xlabel('Latitude (°)'); axs[3].set_ylabel('PW'); axs[3].legend(); axs[3].grid(True)

    # Panel 5: Heat flux convergence (W/m²)
    axs[4].plot(lat, ctrl['conv'], label='Control')
    axs[4].plot(lat, forc['conv'], label='Forced')
    axs[4].set_title('Panel 5: Heat flux convergence (W/m²)'); axs[4].set_xlabel('Latitude (°)'); axs[4].set_ylabel('W/m²'); axs[4].legend(); axs[4].grid(True)

    # Panel 6: Change in zonal mean temperature (°C) + polar amplification
    axs[5].plot(lat, dT_lat, label='Forced - Control (°C)')
    axs[5].set_title(f'Panel 6: ΔT zonal (°C); polar amplification = {polar_ampl:.3f}')
    axs[5].set_xlabel('Latitude (°)'); axs[5].set_ylabel('°C'); axs[5].grid(True)

    # Panel 7: Global mean time series (°C)
    Tg_all = np.concatenate([ctrl['Tg'], forc['Tg']])
    time_years = np.arange(len(Tg_all)) * args.dt   # convert to years

    axs[6].plot(time_years, Tg_all, label='Global mean (°C)')
    axs[6].axvline(len(ctrl['Tg']) * args.dt, color='k', ls='--', label='forcing on')
    axs[6].set_title('Panel 7: Global mean time series (°C)')
    axs[6].set_xlabel('Time (years)'); axs[6].set_ylabel('°C'); axs[6].legend(); axs[6].grid(True)

    # Hide the 8th panel
    axs[7].axis('off')

    fig.tight_layout()
    multi_path = os.path.join(args.outdir, 'ebm_panels.png')
    fig.savefig(multi_path, dpi=150)

    # ---- Summary (console + file) ----
    summary = textwrap.dedent(f"""
    === EBM Summary ===
    Years (control, forced): ({args.years_control}, {args.years_forced})
    Grid points nx: {args.nx}, Δt (years): {args.dt}

    Control global mean T (°C): {mean_ctrl-273.15:.3f}
    Forced  global mean T (°C): {mean_forc-273.15:.3f}
    ΔT global (°C): {dT_global:.3f}

    North pole T control / forced (°C): {Ts_ctrl_pole-273.15:.3f} / {Ts_forc_pole-273.15:.3f}
    Polar amplification ( (ΔT_pole - ΔT_global)/ΔT_global ): {polar_ampl:.3f}

    D_end control / forced (W m⁻² K⁻¹): {ctrl['D_end']:.3f} / {forc['D_end']:.3f}

    Figure saved: {multi_path}
    """)
    print(summary)
    with open(os.path.join(args.outdir, 'summary.txt'), 'w', encoding='utf-8') as f:
        f.write(summary)

if __name__ == '__main__':
    main()