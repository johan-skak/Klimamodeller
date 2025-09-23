# outputs.py
import numpy as np
import matplotlib.pyplot as plt
import textwrap # For dedenting summary text
import os
from matplotlib.ticker import FixedLocator # For custom minor ticks
import physics as phys

def print_simulation_info(config, params):
    print("Running simulation with the following configuration and parameters\nNote: some modes may have changed the values specified in the config and parameter files\n")
    descs = phys.PARAM_DESCS # descriptions for parameter keys
    max_ckey_len = max((len(k) for k in config), default=0) # find max config key length
    max_pkey_len = max((len(k) for k in params), default=0) # find max parameters key length
    max_pdesc_len = max((len(descs.get(k) or "") for k in params.keys()), default=0) # find max length of descriptions associated with the keys in params (if they have an associated description)
    max_pval_len = max((len(str(v)) for v in params.values()), default=0) # find max length af values in params
    total_pwidth = max_pkey_len + max_pval_len + max_pdesc_len + 6

    header = "=== EBM Model Configuration "
    print(header + "=" * (total_pwidth - len(header))) # number of "=" set to right align with following prints
    # print config details
    for key, value in config.items():
        print(f"{key:<{max_ckey_len}} : {str(value)}") # - pad key to align the colons
    print("=" * total_pwidth + "\n")

    header = "=== EBM Model Parameters " # new header
    print(header + "=" * (total_pwidth - len(header)))
    # print parameter details
    for key, value in params.items():
        desc = descs.get(key) or "" # returns empty string if the key is not found
        desc = f"({desc})" # add parentheses to string
        print(f"{key:<{max_pkey_len}} {desc:<{max_pdesc_len+2}} : {value}")
    print("=" * total_pwidth + "\n")

    print("Starting simulation...")

def run_all_outputs(outputs, outdir):
    print(f"Finished simulation. Generating outputs in {outdir}")
    os.makedirs(outdir, exist_ok=True)

    axes_funcs = [ax_func for o in outputs for ax_func in o.axes_funcs]
    if axes_funcs:
        v_num = int(np.round(np.sqrt(2*len(axes_funcs)))) # Aim for 2:1 aspect ratio
        h_num = int(np.ceil(len(axes_funcs) / v_num))
        fig, axs = plt.subplots(v_num, h_num, figsize=(6*h_num, 3.5*v_num))
        axs = np.atleast_1d(axs).flatten()
        
        for axfunc, subplot_ax in zip(axes_funcs, axs):
            axfunc(subplot_ax)  # plotting function should accept "ax"
        for ax in axs[len(axes_funcs):]:
            ax.axis('off')  # Turn off unused subplots
        fig.tight_layout()
        fig.savefig(f"{outdir}/ebm_panels.png", dpi=150)

    summaries = [s for o in outputs for s in o.summaries]
    if summaries:#Also print the summary
        summary = "\n=== EBM Summary ==="
        for sfunc in summaries:
            summary += textwrap.dedent(sfunc) + "\n"
        summary += f"Figures and summary saved in {outdir}\n"
        print(summary)
        with open(f"{outdir}/summary.txt", "w", encoding="utf-8") as f:
            f.write(summary)

class OutPut:
    def __init__(self): pass

    summaries = [] # List of functions to plot on axes_funcs (returns nothing)
    axes_funcs = [] # List of functions to write summaries (returns strings)

    def initialize(self, model): pass
    def step(self, model, i): pass
    def finalize(self, model): pass
    
class DefaultOutput(OutPut):
    def __init__(self):
        self.Tg_series = []
        self.diags = {} # Diagnostics
    
    def initialize(self, model):
        self.diags["_init"] = self.simulation_diagnostics(model.funcs, model.x, model.T, model.params)
        self.x = model.x
        self.lat = np.degrees(np.arcsin(self.x))

    def step(self, model, i):
        if i+1 == model.nsteps//2:
            self.diags["_mid"] = self.simulation_diagnostics(model.funcs, model.x, model.T, model.params)
        # Tmean = model.T.mean() - 273.15
        # self.Tg_series.append(Tmean)

    def finalize(self, model):
        self.diags["_end"] = self.simulation_diagnostics(model.funcs, model.x, model.T, model.params)
        self.dt_global = self.diags["_end"]["T_mean"] - self.diags["_mid"]["T_mean"]
        self.polar_ampl = (self.diags["_end"]["T_poles"][1] - self.diags["_mid"]["T_poles"][1] - self.dt_global) / self.dt_global if self.dt_global != 0 else np.nan
        self.lat_ext = np.r_[-90, self.lat, 90]
        for case in ["_mid", "_end", "_init"]:
            self.diags[case]["T_ext"] = np.r_[self.diags[case]["T_poles"][0], self.diags[case]["T"], self.diags[case]["T_poles"][1]]
        # Finally set up axes_funcs and summaries
        self.axes_funcs = [self.panel1, self.panel2, self.panel3, self.panel4, self.panel5, self.panel6]
        self.summaries = [self.summarize(model, self.diags)]
    
    def summarize(self, model, diags):
        return textwrap.dedent(f"""
        Years (control, forced): ({model.config['ctrl_years']}, {model.config['years']-model.config['ctrl_years']})
        Grid points nx: {model.config['nx']}, Δt (years): {model.config['dt_years']}

        Control global mean T (°C): {diags['_mid']['T_mean']-273.15:.1f}
        Forced  global mean T (°C): {diags['_end']['T_mean']-273.15:.1f}
        ΔT global (°C): {diags['_end']['T_mean']-diags['_mid']['T_mean']:.2f}

        North pole T control / forced (°C): {diags['_mid']['T_poles'][1]-273.15:.1f} / {diags['_end']['T_poles'][1]-273.15:.1f}
        North polar amplification ( (ΔT_pole - ΔT_global)/ΔT_global ): {self.polar_ampl:.3f}

        Outgoing longwave radiation (OLR) control / forced (W m⁻²): {diags['_mid']['olr'].mean():.0f} / {diags['_end']['olr'].mean():.0f}
        Planetary albedo control / forced: {np.average(diags['_mid']['alpha'], weights=diags['_mid']['Q_x']):.3f} / {np.average(diags['_end']['alpha'], weights=diags['_end']['Q_x']):.3f}
        Diffusivity control / forced (W m⁻² K⁻¹): {diags['_mid']['D']:.3f} / {diags['_end']['D']:.3f}
        """)

    # Panel 1: Temperature profiles  (°C)
    def panel1(self, ax):
        """Plot initial, control and final temperature profiles."""
        for case, label in zip(["_mid", "_end", "_init"], ["Control", "Final", "Initial"]):
            ax.plot(self.lat_ext, self.diags[case]["T_ext"], label=label)
        ax.set_title("Temperature profile")
        ax.set_ylabel("°C")
        self.Stylize(ax)
    # Panel 2: OLR profiles (W/m²)
    def panel2(self, ax):
        """Plot control and final OLR profiles."""
        ax.plot(self.lat, self.diags["_mid"]['olr'], label='Control')
        ax.plot(self.lat, self.diags["_end"]['olr'], label='Forced')
        ax.set_title('Outgoing Longwave Radiation (OLR)'); ax.set_ylabel('W/m²')
        self.Stylize(ax)
    # Panel 3: Albedo profiles
    def panel3(self, ax):
        """Plot control and final albedo profiles."""
        ax.plot(self.lat, self.diags["_mid"]['alpha'], label='Control')
        ax.plot(self.lat, self.diags["_end"]['alpha'], label='Forced')
        ax.set_title('Planetary Albedo'); ax.set_ylabel('Albedo')
        self.Stylize(ax)
    # Panel 4: Meridional heat transport (PW)
    def panel4(self, ax):
        """Plot control and final meridional heat transport profiles."""
        ax.plot(self.diags["_mid"]['MHTrans_PW'][0], self.diags["_mid"]['MHTrans_PW'][1], label='Control')
        ax.plot(self.diags["_end"]['MHTrans_PW'][0], self.diags["_end"]['MHTrans_PW'][1], label='Forced')
        ax.set_title('Meridional Heat Transport'); ax.set_ylabel('PW (10¹⁵ W)')
        self.Stylize(ax)
    # Heat flux convergence (W/m²)
    def panel5(self, ax):
        """Plot control and final heat flux convergence profiles."""
        ax.plot(self.lat, self.diags["_mid"]['conv'], label='Control')
        ax.plot(self.lat, self.diags["_end"]['conv'], label='Forced')
        ax.set_title('Heat Flux Convergence'); ax.set_ylabel('W/m²')
        self.Stylize(ax)
    # Change in zonal mean temperature (°C) + polar amplification
    def panel6(self, ax):
        """Plot change in zonal mean temperature profile."""
        dT_ext = self.diags["_end"]['T_ext'] - self.diags["_mid"]['T_ext']
        ax.plot(self.lat_ext, dT_ext, label='Forced - Control')
        ax.set_title('Change in Zonal Mean Temperature'); ax.set_ylabel('ΔT (K)')
        self.Stylize(ax)

    def Stylize(self, ax):
        ax.set_xlim([-90, 90])
        ax.set_xlabel('Latitude')
        ax.legend()
        ax.grid(True)
        ax.set_xticks(np.linspace(-90, 90, 7))
        ax.set_xticklabels([f"{tick:.0f}°" for tick in np.linspace(-90, 90, 7)])
        # Minor grid with 40 ticks based on lat spacing
        minor_ticks = np.arcsin(np.linspace(-1, 1, 40)) * (180/np.pi)
        ax.xaxis.set_minor_locator(FixedLocator(minor_ticks))
        ax.grid(True, which='minor', linestyle=':', alpha=0.6)
    
    def simulation_diagnostics(self, funcs, x, T, params):
        """Calculate diagnostics from temperature profile T (K) at x = sin(lat) with model parameters params.

        Returns
            dict: diagnostic values
        """
        alpha = funcs['albedo_from_T'](T, x, k1=params['k1'])
        dTloc = funcs['deltaT_of_Ts'](T, k3=params['k3'])
        olr = phys.SIGMA * (T - dTloc)**4
        D = params['D0'] * max(0.5, 1.0 + params['k2'] * (funcs['global_mean'](T) - phys.T00))
        aL, bL, cL = funcs['build_diffusion_tridiag'](x, D)
        conv = funcs['apply_L_to_T'](aL, bL, cL, T)     # W/m² (convergence)
        MHTrans_PW = funcs['meridional_transport_PW'](T, x, D) # PW = 10^15 W
        T_mean = funcs['global_mean'](T)
        T_poles = funcs['poles_temperature'](T)
        Q_x = funcs['Q_x'](x, params['S'])
        return dict(T=T, alpha=alpha, olr=olr, conv=conv, MHTrans_PW=MHTrans_PW, D=D, T_mean=T_mean, T_poles=T_poles, Q_x=Q_x)

class TimeSeriesOutput(OutPut):
    def __init__(self, vline=False):
        self.Tg_series = []
        self.vline = vline

    def initialize(self, model):
        self.dt = model.config["dt_years"]
        self.Tg_series.append(model.T.mean() - 273.15)

    def step(self, model, i):
        self.Tg_series.append(model.T.mean() - 273.15)

    def finalize(self, model):
        self.axes_funcs = [self.panel]

    def panel(self, ax):
        """Plot global mean temperature time series."""
        ax.plot(np.arange(len(self.Tg_series)) * self.dt, self.Tg_series, label='Global Mean Temperature')
        ax.set_title("Global Mean Surface Temperature")
        ax.set_xlabel("Time (years)"); ax.set_xlim(0, len(self.Tg_series) * self.dt); ax.set_ylabel("°C"); ax.grid(True)
        if self.vline:
            ax.axvline(len(self.Tg_series)*self.dt/2, color='k', linestyle='--', label='Forcing On')
            ax.legend()

class SeasonalOutput(OutPut):
    def __init__(self):
        self.history = []

    def initialize(self, model):
        self.x = model.x
        self.lat = np.degrees(np.arcsin(self.x))

    def step(self, model, i):
        T = model.T - 273.15
        self.history.append((i*model.config["dt_years"], T.copy()))

    def finalize(self, model):
        self.axes_funcs = [self.panel1, self.panel2]

    def panel1(self, ax):
        # Example: plot last temperature profile
        t, T = self.history[-1]
        ax.plot(self.lat, T)
        ax.set_title(f"Seasonal profile at step {t}")
        ax.set_xlabel("Latitude"); ax.set_ylabel("°C")
    
    def panel2(self, ax):
        # Example: plot temperature near Denmark (lat ~ 56°N) over time
        lat_idx = np.argmin(np.abs(self.lat - 56))
        times = [t for t, T in self.history]
        temperatures = [T[lat_idx] for t, T in self.history]
        ax.plot(times, temperatures)
        ax.set_title("Temperature near Denmark (56°N) over time")
        ax.set_xlabel("Time (years)"); ax.set_ylabel("°C")