# outputs.py
import numpy as np
import matplotlib.pyplot as plt
import textwrap # For dedenting summary text
import os, datetime
from matplotlib.ticker import FixedLocator # For custom minor ticks
import physics as phys

def print_simulation_info(config, params):
    print("Running simulation with the following configuration and parameters\n\033[1mNote\033[0m: some modes may have changed the values specified in the config and parameter files\n")
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
        formatted = f"{value:.3g}" if isinstance(value, float) else str(value)
        print(f"{key:<{max_ckey_len}} : {formatted}") # - pad key to align the colons
    print("=" * total_pwidth + "\n")

    header = "=== EBM Model Parameters " # new header
    print(header + "=" * (total_pwidth - len(header)))
    # print parameter details
    for key, value in params.items():
        desc = descs.get(key) or "" # returns empty string if the key is not found
        desc = f"({desc})" # add parentheses to string
        print(f"{key:<{max_pkey_len}} {desc:<{max_pdesc_len+2}} : {value}")
    print("=" * total_pwidth + "\n")

    print("\033[1mStarting\033[0m simulation...")

def run_all_outputs(outputs, outdir):
    print(f"\033[1mFinished\033[0m simulation. Generating outputs and saving in the \033[4m{outdir}\033[0m folder")
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
    summary = ""
    if summaries:#Also print the summary
        summary = "\n=== EBM Summary ==="
        for sfunc in summaries:
            summary += textwrap.dedent(sfunc) + "\n"
        with open(f"{outdir}/summary.txt", "w", encoding="utf-8") as f:
            f.write(summary)
    summary += f"Figures and summary saved in \033[4m{outdir}\033[0m\n"
    print(summary)

class OutPut:
    def __init__(self): pass

    summaries = [] # List of functions to plot on axes_funcs (returns nothing)
    axes_funcs = [] # List of functions to write summaries (returns strings)

    def initialize(self, model): pass
    def step(self, model, i): pass
    def finalize(self, model): pass
    
class DefaultOutput(OutPut):
    diags = {} # Diagnostics
    
    def initialize(self, model):
        self.diags["init"] = self.simulation_diagnostics(model.funcs, model.x, model.T, model.params)
        self.x = model.x
        self.lat = np.degrees(np.arcsin(self.x))

    def step(self, model, i):
        if i+1 == model.nsteps//2:
            self.diags["mid"] = self.simulation_diagnostics(model.funcs, model.x, model.T, model.params)

    def finalize(self, model):
        self.diags["end"] = self.simulation_diagnostics(model.funcs, model.x, model.T, model.params)
        self.dt_global = self.diags["end"]["T_mean"] - self.diags["mid"]["T_mean"]
        self.polar_ampl = (self.diags["end"]["T_poles"][1] - self.diags["mid"]["T_poles"][1] - self.dt_global) / self.dt_global if self.dt_global != 0 else np.nan
        self.lat_ext = np.r_[-90, self.lat, 90]
        for case in ["mid", "end", "init"]:
            self.diags[case]["T_ext"] = np.r_[self.diags[case]["T_poles"][0], self.diags[case]["T"], self.diags[case]["T_poles"][1]]
        # Finally set up axes_funcs and summaries
        self.axes_funcs = [self.panel1, self.panel2, self.panel3, self.panel4, self.panel5, self.panel6]
        self.summaries = [self.summarize(model, self.diags)]
    
    def summarize(self, model, diags):
        T_ctrl_fmt = T_forc_fmt = end_fmt = "\033[0m"
        if diags['mid']['T_mean'] > 273.15 + 40:
            T_ctrl_fmt = "\033[31m"
        elif diags['mid']['T_mean'] < 273.15:
            T_ctrl_fmt = "\033[34m"
        if diags['end']['T_mean'] > 273.15 + 40:
            T_forc_fmt = "\033[31m"
        elif diags['end']['T_mean'] < 273.15:
            T_forc_fmt = "\033[34m"

        return textwrap.dedent(f"""
        Years (control, forced): ({model.config['ctrl_years']}, {model.config['years']-model.config['ctrl_years']})
        Grid points nx: {model.config['nx']}, Δt (years): {model.config['dt_years']}

        Control global mean T (°C): {T_ctrl_fmt}{diags['mid']['T_mean']-273.15:.1f}{end_fmt}
        Forced  global mean T (°C): {T_forc_fmt}{diags['end']['T_mean']-273.15:.1f}{end_fmt}
        ΔT global (°C): {diags['end']['T_mean']-diags['mid']['T_mean']:.2f}

        North pole T control / forced (°C): {diags['mid']['T_poles'][1]-273.15:.1f} / {diags['end']['T_poles'][1]-273.15:.1f}
        North polar amplification ( (ΔT_pole - ΔT_global)/ΔT_global ): {self.polar_ampl:.3f}

        Outgoing longwave radiation (OLR) control / forced (W m⁻²): {diags['mid']['olr'].mean():.0f} / {diags['end']['olr'].mean():.0f}
        Planetary albedo control / forced: {np.average(diags['mid']['alpha'], weights=diags['mid']['Q_x']):.3f} / {np.average(diags['end']['alpha'], weights=diags['end']['Q_x']):.3f}
        Diffusivity control / forced (W m⁻² K⁻¹): {diags['mid']['D']:.3f} / {diags['end']['D']:.3f}
        """)

    # Panel 1: Temperature profiles  (°C)
    def panel1(self, ax):
        """Plot initial, control and final temperature profiles."""
        for case, label in zip(["mid", "end", "init"], ["Control", "Final", "Initial"]):
            ax.plot(self.lat_ext, self.diags[case]["T_ext"], label=label)
        ax.set_title("Temperature profile")
        ax.set_ylabel("°C")
        self.Stylize(ax)
    # Panel 2: OLR profiles (W/m²)
    def panel2(self, ax):
        """Plot control and final OLR profiles."""
        ax.plot(self.lat, self.diags["mid"]['olr'], label='Control')
        ax.plot(self.lat, self.diags["end"]['olr'], label='Forced')
        ax.set_title('Outgoing Longwave Radiation (OLR)'); ax.set_ylabel('W/m²')
        self.Stylize(ax)
    # Panel 3: Albedo profiles
    def panel3(self, ax):
        """Plot control and final albedo profiles."""
        ax.plot(self.lat, self.diags["mid"]['alpha'], label='Control')
        ax.plot(self.lat, self.diags["end"]['alpha'], label='Forced')
        ax.set_title('Planetary Albedo'); ax.set_ylabel('Albedo')
        self.Stylize(ax)
    # Panel 4: Meridional heat transport (PW)
    def panel4(self, ax):
        """Plot control and final meridional heat transport profiles."""
        ax.plot(self.diags["mid"]['MHTrans_PW'][0], self.diags["mid"]['MHTrans_PW'][1], label='Control')
        ax.plot(self.diags["end"]['MHTrans_PW'][0], self.diags["end"]['MHTrans_PW'][1], label='Forced')
        ax.set_title('Meridional Heat Transport'); ax.set_ylabel('PW (10¹⁵ W)')
        self.Stylize(ax)
    # Heat flux convergence (W/m²)
    def panel5(self, ax):
        """Plot control and final heat flux convergence profiles."""
        ax.plot(self.lat, self.diags["mid"]['conv'], label='Control')
        ax.plot(self.lat, self.diags["end"]['conv'], label='Forced')
        ax.set_title('Heat Flux Convergence'); ax.set_ylabel('W/m²')
        self.Stylize(ax)
    # Change in zonal mean temperature (°C) + polar amplification
    def panel6(self, ax):
        """Plot change in zonal mean temperature profile."""
        dT_ext = self.diags["end"]['T_ext'] - self.diags["mid"]['T_ext']
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
    
    def simulation_diagnostics(self, funcs, x, T, params, model=None, i=0):
        """Calculate diagnostics from temperature profile T (K) at x = sin(lat) with model parameters params.

        Returns
            dict: diagnostic values
        """
        alpha = funcs['albedo_from_T'](T, x, k1=params['k1'], model=model, i=i)
        dTloc = funcs['deltaT_of_Ts'](T, k3=params['k3'], model=model, i=i)
        olr = phys.SIGMA * (T - dTloc)**4
        D = funcs["diffusion_from_T"](T.mean(), params['D0'], params['k2'], model=model, i=i)
        aL, bL, cL = funcs['build_diffusion_tridiag'](x, D)
        conv = funcs['apply_L_to_T'](aL, bL, cL, T)     # W/m² (convergence)
        MHTrans_PW = funcs['meridional_transport_PW'](T, x, D, model=model, i=i) # PW = 10^15 W
        T_mean = T.mean()
        T_poles = funcs['poles_temperature'](T, model=model, i=i)
        Q_x = funcs['Q_x'](x, params['S'], model=model, i=i)
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
        self.t = []
        self.series = {key: [] for key in ["T", "olr", "alpha", "conv", "MHTrans_PW", "D", "Q_x"]}

    def initialize(self, model):
        self.x = model.x
        self.lat = np.degrees(np.arcsin(self.x))
        self.dt = model.config["dt_years"]

    def step(self, model, i):
        self.t.append((i+1) * self.dt)
        diags = DefaultOutput().simulation_diagnostics(model.funcs, model.x, model.T, model.params, model=model, i=i)
        self.series["T"].append(model.T.copy())
        for key in ["olr", "alpha", "conv", "MHTrans_PW", "D", "Q_x"]:
            self.series[key].append(diags[key])

    def finalize(self, model):
        # Convert to numpy arrays
        for key in self.series:
            self.series[key] = np.array(self.series[key])

        # Extract last year
        steps_per_year = int(round(1 / self.dt)) #Already a whole number up to machine precision
        last_slice = slice(-steps_per_year-1, None)
        self.last = {key: arr[last_slice] for key, arr in self.series.items()}
        self.t_last = np.array(self.t[last_slice])

        # Seasonal phases (relative indices into last-year arrays)
        quarter = steps_per_year // 4
        self.phases = {
            "Spring eqx": 0,
            "Summer sol": quarter,
            "Autumn eqx": 2 * quarter,
            "Winter sol": 3 * quarter,
        }

        # Locations of interest (indices along latitude)
        self.locs = {
            "Global mean": None,
            "Equator": np.argmin(np.abs(self.lat - 0)),
            "Denmark (56°N)": np.argmin(np.abs(self.lat - 56)),
            "North pole": -1,
            "South pole": 0,
        }

        self.axes_funcs = [self.panel1, self.panel2, self.panel3, self.panel4,
                           self.panel5, self.panel6, self.panel7, self.panel8]
        self.summaries = [self.summarize(model)]

    # ---- Panels ----
    def plot_profiles(self, ax, field, ylabel, title):
        for label, idx in self.phases.items():
            data = self.last[field][idx]
            if field == "MHTrans_PW":   # (x,y) tuple
                ax.plot(data[0], data[1], label=label)
            else:
                values = data - 273.15 if field == "T" else data
                ax.plot(self.lat, values, label=label)
        ax.set_title(title); ax.set_ylabel(ylabel)
        DefaultOutput.Stylize(self, ax)

    def panel1(self, ax): self.plot_profiles(ax, "T", "°C", "Seasonal Temperature Profiles")
    def panel2(self, ax): self.plot_profiles(ax, "olr", "W/m²", "Seasonal OLR Profiles")
    def panel3(self, ax): self.plot_profiles(ax, "alpha", "Albedo", "Seasonal Albedo Profiles")
    def panel4(self, ax): self.plot_profiles(ax, "MHTrans_PW", "PW (10¹⁵ W)", "Seasonal Meridional Heat Transport")
    def panel5(self, ax): self.plot_profiles(ax, "conv", "W/m²", "Seasonal Heat Flux Convergence")
    def panel6(self, ax): self.plot_profiles(ax, "Q_x", "W/m²", "Seasonal Solar Irradiance")
        
    def plot_time_series(self, ax, field, ylabel, title, mean_is_zero=False):
        for name, idx in self.locs.items():
            mean = self.last[field].mean() if idx is None else self.last[field][:, idx].mean() if mean_is_zero else 0
            series = (self.last[field].mean(axis=1) if idx is None else self.last[field][:, idx])-mean
            ax.plot(self.t_last, series, label=name)
        ax.set_xticks(self.t_last[np.linspace(0,len(self.t_last)-1, 7, dtype=np.int16)])
        ax.set_xticklabels([self.date_from_fraction(t) for t in np.linspace(0, 1, 7)])
        ax.set_xlabel("Date (during the last simulated year)")
        ax.set_title(title); ax.set_ylabel(ylabel); ax.legend(); ax.grid(True)

    def panel7(self, ax): self.plot_time_series(ax, "Q_x", "W/m²", "Seasonal Solar Irradiance Time Series")
    def panel8(self, ax): self.plot_time_series(ax, "T", "°C", "Seasonal Temperature Variation Time Series", mean_is_zero=True)

    # ---- Summary ----
    def summarize(self, model):
        t = self.t_last
        
        def color_fmt(n, p=1):
            start_fmt = end_fmt = "\033[0m"
            if n > 40: start_fmt = "\033[31m"
            elif n < 0: start_fmt = "\033[34m"
            return f"{start_fmt}{n:>{p+4}.{p}f}{end_fmt}"
        
        def fmt(series):
            min_idx, max_idx = series.argmin(), series.argmax()
            min_time = t[min_idx] % 1   # fractional year since last equinox
            max_time = t[max_idx] % 1
            return (color_fmt(series.mean(), 2) + "°C " +
                    "(min " + color_fmt(series.min()) + f" on {self.date_from_fraction(min_time):>5} ({min_time:>4.2f}y), "
                    f"max " + color_fmt(series.max()) + f" on {self.date_from_fraction(max_time):>5} ({max_time:>4.2f}y))")

        global_T = self.last["T"].mean(axis=1) - 273.15
        equator_T = self.last["T"][:, self.locs["Equator"]] - 273.15
        denmark_T = self.last["T"][:, self.locs["Denmark (56°N)"]] - 273.15
        north_T = self.last["T"][:, self.locs["North pole"]] - 273.15
        south_T = self.last["T"][:, self.locs["South pole"]] - 273.15

        return textwrap.dedent(f"""
        === Seasonal Diagnostics (last year) ===
        Modes: {model.config.get("modes")}
        Years run: {model.config["years"]}, grid points: {model.config["nx"]}, Δt (years): 1 / {round(1 / self.dt)}
        
        Global mean temperature: {fmt(global_T)}
        Equator temperature:     {fmt(equator_T)}
        Denmark (56°N):          {fmt(denmark_T)}
        North pole:              {fmt(north_T)}
        South pole:              {fmt(south_T)}

        Last-year mean OLR:    {self.last['olr'].mean():.1f} W/m²
        Last-year mean albedo: {self.last['alpha'].mean():.3f}
        Last-year mean D:      {self.last['D'].mean():.3f} W m⁻² K⁻¹
        """)

    def date_from_fraction(self, frac):
        """Convert a fraction of a year since spring equinox to a date string."""
        # Anchor on March 21 of an arbitrary year (say year 2000, leap-safe)
        start = datetime.date(2000, 3, 21)
        days_in_year = 365
        offset_days = int(round(frac * days_in_year))
        date = start + datetime.timedelta(days=offset_days)
        return date.strftime("%b %d")

class SeaDepthOutput(OutPut):
    SeaDepths = [] # Series of sea depths

    def step(self, model, i):
        self.SeaDepths.append(model.C / phys.C_M)
    
    def finalize(self, model):
        quarter = int(round(1 / model.config['dt_years'])) // 4
        self.phases = {
                "Spring eqx": 0,
                "Summer sol": quarter,
                "Autumn eqx": 2 * quarter,
                "Winter sol": 3 * quarter,
            }
        self.lat = np.degrees(np.arcsin(model.x))
        
        self.axes_funcs = [self.panel]
    
    def panel(self, ax):
        for label, idx in self.phases.items():
            ax.plot(self.lat, self.SeaDepths[idx], label=label)
        ax.set_title(r"Heat capacities based on ML depth and % of landmass"); ax.set_ylabel("m (equivalent water depth)")
        DefaultOutput.Stylize(self, ax)