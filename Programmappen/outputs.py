# outputs.py
import numpy as np
import matplotlib.pyplot as plt
import textwrap # For dedenting summary text
import os, datetime, re
from matplotlib.ticker import FixedLocator # For custom minor ticks
#from asyncio import tools
import tools
import physics as phys
import AnimateOnEarth as Earth
import pandas as pd

def remove_ansi(text):
    ansi_escape = re.compile(r'\x1B\[[0-9;]*m') # Matches ANSI escape sequences like \033[1;33m
    return ansi_escape.sub('', text) # Remove ANSI sequences from text

def print_simulation_info(config, params, app_mode=False):
    if not app_mode: print("Running simulation with the following configuration and parameters\n\033[1mNote\033[0m: some modes may have changed the values specified in the config and parameter files\n")
    descs = phys.PARAM_DESCS # descriptions for parameter keys
    max_ckey_len = max((len(k) for k in config), default=0) # find max config key length
    max_pkey_len = max((len(k) for k in params), default=0) # find max parameters key length
    max_pdesc_len = max((len(descs.get(k) or "") for k in params.keys()), default=0) # find max length of descriptions associated with the keys in params (if they have an associated description)
    max_pval_len = max((len(str(v)) for v in params.values()), default=0) # find max length af values in params
    total_pwidth = max_pkey_len + max_pval_len + max_pdesc_len + 6

    header = "=== EBM Model Configuration "
    info_str = header + "=" * (total_pwidth - len(header)) + "\n" # The number of "=" set to right align with following prints
    # config details
    for key, value in config.items():
        formatted = f"{value:.3g}" if isinstance(value, float) else str(value) # Format floats to 3 significant digits
        info_str += f"{key:<{max_ckey_len}} : {formatted}\n" # - pad key to align the colons
    info_str += "=" * total_pwidth + "\n\n"

    header = "=== EBM Model Parameters " # new header
    info_str += header + "=" * (total_pwidth - len(header)) + "\n"
    # parameter details
    for key, value in params.items():
        desc = descs.get(key) or "" # returns empty string if the key is not found
        desc = f"({desc})" # add parentheses to string
        info_str += f"{key:<{max_pkey_len}} {desc:<{max_pdesc_len+2}} : {value}\n"
    info_str += "=" * total_pwidth + "\n\n"
    if not app_mode: print(info_str, end="") # no newline at end

    if not app_mode: print("\033[1mStarting\033[0m simulation...\n")
    return info_str

def aspect_ratio(n, goal):
    """Calculate number of rows and columns for n subplots to achieve a given aspect ratio goal (width/height)."""
    h_num_top = int(np.ceil(np.sqrt(goal*n))) # Aim for goal:1 aspect ratio
    h_num_bottom = int(np.sqrt(goal*n)) # Another option. v_num_top=v_num_bottom+1 unless goal*n is a perfect square
    h_num = h_num_top if (-n) % h_num_top <= (-n) % h_num_bottom else h_num_bottom # Choose the one that gives least empty plots # Prefers more columns at equality
    return int(np.ceil(n / h_num)), h_num

def generate_outputs_data(axes_funcs, summaries, outdir="", sim_info=""):
    """Generate all outputs (figures and summary text) and return them. If no outputs are defined, return None.
    Parameters:
        - axes_funcs: List of functions to draw axes for plots
        - summaries: List of summary strings for each output
        - outdir: Directory to save output files (string).
        - sim_info: String containing information about simulation parameters and configuration.

    Returns:
        - fig: Matplotlib figure object containing all plots, or None if no plots were generated.
        - summary: Summary string with ANSI formatting, or None if no summary was generated.
        - clean_summary: Summary string without ANSI formatting, or None if no summary was generated.
    """
    # Make plots
    if axes_funcs:
        v_num, h_num = aspect_ratio(len(axes_funcs), 1) # Aim for 16:9 aspect ratio of figure
        fig, axs = plt.subplots(v_num, h_num, figsize=(6*h_num, 27/8*v_num)) # Width:Height = 6*h:27/8*v = 16*h:9*v
        axs = np.atleast_1d(axs).flatten()
        
        for axfunc, subplot_ax in zip(axes_funcs, axs):
            axfunc(subplot_ax)  # plotting function should accept "ax"
        for ax in axs[len(axes_funcs):]:
            ax.axis('off')  # Turn off unused subplots
        fig.tight_layout()

    # Make summary
    summary = ""
    if summaries:#Also print the summary
        for sfunc in summaries:
            summary += textwrap.dedent(sfunc) + "\n"
        clean_summary = remove_ansi(summary)
        #Find max line length
        max_line_length = max(len(line) for line in clean_summary.split("\n"))
        header = "=== EBM Summary "
        pre_text = sim_info + f"Output generated on {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n" + header + "=" * (max_line_length - len(header)) + "\n"
        clean_summary = pre_text + clean_summary
    summary += f"Figures and summary saved in \033[4m{outdir}\033[0m\n"

    return fig, summary, clean_summary if summaries else None

def run_all_outputs(outputs, outdir, sim_info="", runtime=None, app=False):
    """Generate and save all outputs (figures and summary text) or return output generators if in app mode.
    Parameters:
        - outputs: List of output objects to generate data for.
        - outdir: Directory to save output files (string).
        - sim_info: String containing information about simulation parameters and configuration.
        - runtime: Optional float indicating the runtime of the simulation in seconds.
        - app: Boolean indicating if running in app mode (e.g., Streamlit). If True, returns outputs instead of saving to files.
    
    Returns (if app is True):
        - axes_funcs: List of functions to draw axes for plots.
        - summaries: List of summary strings for each output.
    """
    axes_funcs = [ax_func for o in outputs for ax_func in o.axes_funcs]
    summaries = [s for o in outputs for s in o.summaries]
    
    if not app: # Normal script mode
        fig, summary, clean_summary = generate_outputs_data(axes_funcs, summaries, outdir, sim_info)
        timedesc = f" in {runtime:.2f} seconds" if runtime is not None else ""
        print(f"\033[1mFinished\033[0m simulation{timedesc}. Generating outputs and saving in the \033[4m{outdir}\033[0m folder")
        os.makedirs(outdir, exist_ok=True)
        fig.savefig(f"{outdir}/ebm_panels.png", dpi=150)
        with open(f"{outdir}/summary.txt", "w", encoding="utf-8") as f:
            f.write(clean_summary)
        print(summary)
    else:
        return axes_funcs, summaries # For Streamlit app

class OutPut:
    def __init__(self):
        self.summaries = [] # List of functions to plot on axes_funcs (returns nothing)
        self.axes_funcs = [] # List of functions to write summaries (returns strings)

    def initialize(self, model): pass
    def step(self, model, i): pass
    def finalize(self, model): pass
    
class DefaultOutput(OutPut):
    def __init__(self):
        super().__init__()
        self.diags = {} # Diagnostics
    
    def initialize(self, model):
        self.diags["init"] = self.simulation_diagnostics(model.funcs, model.x, model.T, model.params)
        self.x = model.x
        self.lat = np.degrees(np.arcsin(self.x))
        self.Forcing_on = model.config["ctrl_years"] > 0 and (model.params.get('F') != 0 or model.params['S1'] != model.params['S0'])
            
    def step(self, model, i):
        if i == model.ctrl_nsteps:
            self.diags["mid"] = self.simulation_diagnostics(model.funcs, model.x, model.T, model.params)

    def finalize(self, model):
        self.diags["end"] = self.simulation_diagnostics(model.funcs, model.x, model.T, model.params)
        if "mid" not in self.diags: # No control run # Temporary solution
            self.diags["mid"] = self.diags["end"]
        self.dt_global = self.diags["end"]["T_mean"] - self.diags["mid"]["T_mean"]
        self.polar_ampl = (self.diags["end"]["T_poles"][1] - self.diags["mid"]["T_poles"][1] - self.dt_global) / self.dt_global if self.dt_global != 0 else np.nan
        self.lat_ext = np.r_[-90, self.lat, 90]
        # Set cases and labels depending on whether there was a (significant) control run
        self.cases = ["init", "mid", "end"] if self.Forcing_on else ["init", "end"]
        self.labels = ["Initial", "Control", "Forced"] if self.Forcing_on else ["Initial", "Final"]
        self.colors = ["C2", "C0", "C1"] if self.Forcing_on else ["C2", "C1"]
        for case in self.cases:
            self.diags[case]["T_ext"] = np.r_[self.diags[case]["T_poles"][0], self.diags[case]["T"], self.diags[case]["T_poles"][1]]
        # Finally set up axes_funcs and summaries
        self.axes_funcs = [self.panel1, self.panel2, self.panel3, self.panel4, self.panel5]
        if self.Forcing_on: self.axes_funcs.append(self.panel6) # Only add panel6 if there was a control run
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

        Control global mean T (°C): {temp_fmt(diags['mid']['T_mean']-273.15,1)}
        Forced global mean T (°C): {temp_fmt(diags['end']['T_mean']-273.15,1)}
        ΔT global (°C): {diags['end']['T_mean']-diags['mid']['T_mean']:.1f}

        North pole T control / forced (°C): {temp_fmt(diags['mid']['T_poles'][1]-273.15,1)} / {temp_fmt(diags['end']['T_poles'][1]-273.15,1)}
        North polar amplification ( (ΔT_pole - ΔT_global)/ΔT_global ): {self.polar_ampl:.3f}

        Outgoing longwave radiation (OLR) control / forced (W m⁻²): {diags['mid']['olr'].mean():.0f} / {diags['end']['olr'].mean():.0f}
        Planetary albedo control / forced: {np.average(diags['mid']['alpha'], weights=diags['mid']['Q_x']):.3f} / {np.average(diags['end']['alpha'], weights=diags['end']['Q_x']):.3f}
        Diffusivity control / forced (W m⁻² K⁻¹): {diags['mid']['D']:.3f} / {diags['end']['D']:.3f}
        """)

    # Panel 1: Temperature profiles  (°C)
    def panel1(self, ax):
        """Plot initial, control and final temperature profiles."""
        for case, label, color in zip(self.cases, self.labels, self.colors):
            if case == "init": continue
            ax.plot(self.lat_ext, self.diags[case]["T_ext"] - 273.15, label=label, color=color)
        ax.axhline(0, color="#00aeff", linestyle='--', alpha=0.7,label = "0 °C") # 0 °C line
        ax.set_title("Temperature Profile")
        ax.set_ylabel("°C")
        self.Stylize(ax)
    # Panel 2: OLR profiles (W/m²)
    def panel2(self, ax):
        """Plot control and final OLR profiles."""
        for case, label, color in zip(self.cases, self.labels, self.colors):
            if case == "init": continue
            ax.plot(self.lat, self.diags[case]['olr'], label=label, color=color)
        ax.set_title('Outgoing Longwave Radiation (OLR)'); ax.set_ylabel('W/m²')
        self.Stylize(ax)
    # Panel 3: Albedo profiles
    def panel3(self, ax):
        """Plot control and final albedo profiles."""
        for case, label, color in zip(self.cases, self.labels, self.colors):
            if case == "init": continue
            ax.plot(self.lat, self.diags[case]['alpha'], label=label, color=color)
        ax.set_title('Planetary Albedo'); ax.set_ylabel('Albedo')
        self.Stylize(ax)
    # Panel 4: Meridional heat transport (PW)
    def panel4(self, ax):
        """Plot control and final meridional heat transport profiles."""
        for case, label, color in zip(self.cases, self.labels, self.colors):
            if case == "init": continue
            ax.plot(self.diags[case]['MHTrans_PW'][0], self.diags[case]['MHTrans_PW'][1], label=label, color=color)
        ax.set_title('Meridional Heat Transport'); ax.set_ylabel('PW (10¹⁵ W)')
        self.Stylize(ax)
    # Heat flux convergence (W/m²)
    def panel5(self, ax):
        """Plot control and final heat flux convergence profiles."""
        for case, label, color in zip(self.cases, self.labels, self.colors):
            if case == "init": continue
            ax.plot(self.lat, self.diags[case]['conv'], label=label, color=color)
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
        ax.legend() if ax.get_legend_handles_labels()[1] else None # Only add legend if there are labels
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
        D = funcs["diffusion_from_T"](T, params['D0'], params['k2'], model=model, i=i)
        aL, bL, cL = funcs['build_diffusion_tridiag'](x, D)
        conv = funcs['apply_L_to_T'](aL, bL, cL, T)     # W/m² (convergence)
        MHTrans_PW = funcs['meridional_transport_PW'](T, x, D, model=model, i=i) # PW = 10^15 W
        T_mean = T.mean()
        T_poles = funcs['poles_temperature'](T, model=model, i=i)
        Q_x = funcs['Q_x'](x, params['S0'], model=model, i=i)
        return dict(T=T, alpha=alpha, olr=olr, conv=conv, MHTrans_PW=MHTrans_PW, D=D, T_mean=T_mean, T_poles=T_poles, Q_x=Q_x)
    
class ModifyOutput(DefaultOutput):
    def __init__(self):
        super().__init__()

    def initialize(self, model):
        super().initialize(model)

        print("Loading zonal temperature data from file:", model.config["zonal_temp_file"])
        T_x,CurrentZonalMeanTemperature  = tools.csv_reader(model.config["zonal_temp_file"],1)
        self.T_zonal = np.interp(model.x, T_x, CurrentZonalMeanTemperature)

        print("Loading zonal OLR data from file:", model.config["zonal_olr_file"])
        OLR_x,OLR = tools.csv_reader(model.config["zonal_olr_file"],1)
        self.OLR_zonal = np.interp(model.x, OLR_x, OLR)

        print("Loading zonal albedo data from file:", model.config["albedo_file"])
        Albedo_x, Albedo = tools.csv_reader(model.config["albedo_file"], 1)
        self.Albedo_zonal = np.interp(model.x, Albedo_x, Albedo)

        print("Loading zonal solar data from file:", model.config["solar_file"])
        Solar_x, Solar = tools.csv_reader(model.config["solar_file"], 1)
        self.Solar_zonal = np.interp(model.x, Solar_x, Solar)

        #print("Loading temperature history from file:", model.config["temperature_history"])
        #ds_giss = tools.netcdf_reader(model.config["temperature_history"])
        #self.giss_time = pd.to_datetime(ds_giss['time'].values)
        #self.giss_temp = np.squeeze(ds_giss['tempanomaly'].values)

    def panel1(self, ax):
        super().panel1(ax)
        ax.plot(self.lat, self.T_zonal - 273.15, label='Observed', linestyle='--', color="green")
        ax.legend()
    
    def panel2(self, ax):
        super().panel2(ax)
        ax.plot(self.lat, self.OLR_zonal, label='Observed', linestyle='--', color="green")
        ax.legend()

    def panel3(self, ax):
        super().panel3(ax)
        ax.plot(self.lat, self.Albedo_zonal/self.Solar_zonal, label='Observed', linestyle='--', color="green")
        ax.legend()

    #def panel7(self, ax):
        #ax.plot(self.giss_time, self.giss_temp, color='black', label='GISS Temperaturanomalier', linewidth=1.0)
        #ax.set_xlabel("Years")
        #ax.set_ylabel("Temperature anomalies [°C] ")
        #ax.set_title("Observed temperatures 1880-2024")
        #ax.legend()

    #def finalize(self, model):
        #super().finalize(model)
        #self.axes_funcs.append(self.panel7)


class TimeSeriesOutput(OutPut):
    def __init__(self):
        super().__init__()
        self.Tg_series = []

    def initialize(self, model):
        self.dt = model.config["dt_years"]
        self.Tg_series.append(model.T.mean() - 273.15)
         # Whether to draw vertical line at forcing time
        self.Forcing_on = model.config["ctrl_years"] > 0 and (model.params.get('F') != 0 or model.params['S1'] != model.params['S0'])
        self.ctrl_years = model.config["ctrl_years"]

    def step(self, model, i):
        self.Tg_series.append(model.T.mean() - 273.15)

    def finalize(self, model):
        self.axes_funcs = [self.panel]

    def panel(self, ax):
        """Plot global mean temperature time series."""
        ax.plot(np.arange(len(self.Tg_series)) * self.dt, self.Tg_series, label='Global Mean Temperature')
        ax.set_title("Global Mean Surface Temperature")
        ax.set_xlabel("Time [years]"); ax.set_xlim(0, len(self.Tg_series) * self.dt); ax.set_ylabel("Temperature [°C]"); ax.grid(True)
        if self.Forcing_on:
            ax.axvline(self.ctrl_years, color='k', linestyle='--', label='Forcing On')
            ax.legend()

class SeasonalOutput(OutPut):
    def __init__(self):
        super().__init__()
        self.t = []
        self.series = {key: [] for key in ["T", "T_ext", "olr", "alpha", "conv", "MHTrans_PW", "D", "Q_x", "Q_x_ext"]}

    def initialize(self, model):
        self.x = model.x
        self.x_ext = np.r_[-1, self.x, 1] # Extended grid including poles
        self.lat = np.degrees(np.arcsin(self.x))
        self.lat_ext = np.r_[-90, self.lat, 90]
        self.dt = model.config["dt_years"]

    def step(self, model, i):
        self.t.append((i+1) * self.dt)
        diags = DefaultOutput().simulation_diagnostics(model.funcs, model.x, model.T, model.params, model=model, i=i)
        self.series["T"].append(model.T.copy())
        for key in ["olr", "alpha", "conv", "MHTrans_PW", "D", "Q_x"]:
            self.series[key].append(diags[key])
        self.series["T_ext"].append(np.r_[diags["T_poles"][0], diags["T"], diags["T_poles"][1]]) # Include poles
        self.series["Q_x_ext"].append(phys.seasonal_Q(self.x_ext, model.params['S0'] if i < model.ctrl_nsteps else model.params['S1'], model, i)) # Seasonal Q including poles

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
            # if field == "Q_x_ext": print(data)
            lat = self.lat if len(data) == len(self.lat) else self.lat_ext
            if field == "MHTrans_PW":
                ax.plot(data[0], data[1], label=label) # MHTrans_PW is a tuple: (lat, value)
            else:
                values = data - 273.15 if field == "T_ext" else data
                ax.plot(lat, values, label=label)
            if field == "T_ext":
                ax.axhline(0, color="#00aeff", linestyle='--', alpha=0.7) # Add 0 °C line
        ax.set_title(title); ax.set_ylabel(ylabel)
        DefaultOutput.Stylize(self, ax)

    def panel1(self, ax): self.plot_profiles(ax, "T_ext", "°C", "Seasonal Temperature Profiles")
    def panel2(self, ax): self.plot_profiles(ax, "olr", "W/m²", "Seasonal OLR Profiles")
    def panel3(self, ax): self.plot_profiles(ax, "alpha", "Albedo", "Seasonal Albedo Profiles")
    def panel4(self, ax): self.plot_profiles(ax, "MHTrans_PW", "PW (10¹⁵ W)", "Seasonal Meridional Heat Transport")
    def panel5(self, ax): self.plot_profiles(ax, "conv", "W/m²", "Seasonal Heat Flux Convergence")
    def panel6(self, ax): self.plot_profiles(ax, "Q_x_ext", "W/m²", "Seasonal Solar Irradiance")
        
    def plot_time_series(self, ax, field, ylabel, title, mean_is_zero=False):
        for name, idx in self.locs.items():
            mean = self.last[field].mean() if idx is None else self.last[field][:, idx].mean() if mean_is_zero else 0
            series = (self.last[field].mean(axis=1) if idx is None else self.last[field][:, idx])-mean
            ax.plot(self.t_last, series, label=name)
        ax.set_xticks(self.t_last[np.linspace(0,len(self.t_last)-1, 7, dtype=np.int16)])
        ax.set_xticklabels([self.date_from_fraction(t) for t in np.linspace(0, 1, 7)])
        ax.set_xlabel("Date (during the last simulated year)")
        ax.set_title(title); ax.set_ylabel(ylabel); ax.legend(); ax.grid(True), ax.set_xlim(self.t_last[0], self.t_last[-1])

    def panel7(self, ax): self.plot_time_series(ax, "Q_x", "W/m²", "Seasonal Solar Irradiance Time Series")
    def panel8(self, ax): self.plot_time_series(ax, "T", "°C", "Seasonal Temperature Variation Time Series", mean_is_zero=True)

    # ---- Summary ----
    def summarize(self, model):
        t = self.t_last
        
        def fmt(series):
            min_idx, max_idx = series.argmin(), series.argmax()
            min_time = t[min_idx] % 1   # fractional year since last equinox
            max_time = t[max_idx] % 1
            return (temp_fmt(series.mean(), 2) + "°C " +
                    "(min " + temp_fmt(series.min()) + f" on {self.date_from_fraction(min_time):>5} ({min_time:>4.2f}y), "
                    f"max " + temp_fmt(series.max()) + f" on {self.date_from_fraction(max_time):>5} ({max_time:>4.2f}y))")

        global_T = self.last["T"].mean(axis=1) - 273.15
        equator_T = self.last["T"][:, self.locs["Equator"]] - 273.15
        denmark_T = self.last["T"][:, self.locs["Denmark (56°N)"]] - 273.15
        north_T = self.last["T"][:, self.locs["North pole"]] - 273.15
        south_T = self.last["T"][:, self.locs["South pole"]] - 273.15

        return textwrap.dedent(f"""
        === Seasonal Diagnostics (last year) ===
        Modes: {model.config.get("modes")}
        Years run: {model.config["years"]}, grid points: {model.config["nx"]}, Δt (years): 1 / {round(1 / self.dt)}
        
        {"Global mean temperature:":<24}{fmt(global_T)}
        {"Equator temperature:":<24}{fmt(equator_T)}
        {"Denmark (56°N):":<24}{fmt(denmark_T)}
        {"North pole:":<24}{fmt(north_T)}
        {"South pole:":<24}{fmt(south_T)}

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
    def __init__(self):
        super().__init__()
        self.T_ext = [] # Series of sea depths

    def initialize(self, model):
        self.k1 = model.params["k1"]

    def step(self, model, i):
        T = model.T.copy()
        T_poles = model.funcs['poles_temperature'](T, model=model, i=i)
        self.T_ext.append(np.r_[T_poles[0], T, T_poles[1]])
    
    def finalize(self, model):
        quarter = int(round(1 / model.config['dt_years'])) // 4
        self.phases = {
                "Spring eqx": 0,
                "Summer sol": quarter,
                "Autumn eqx": 2 * quarter,
                "Winter sol": 3 * quarter,
            } if "SeasonalVariation" in model.config["modes"] else {"": -1}
        self.x_ext = np.r_[-1, model.x, 1]
        self.lat_ext = np.degrees(np.arcsin(self.x_ext))

        self.axes_funcs = [self.panel]
    
    def panel(self, ax):
        for label, idx in self.phases.items():
            ax.plot(self.lat_ext, phys.heat_capacity_profile(self.x_ext, self.T_ext[idx], self.k1) / phys.C_M, label=label)
        ax.set_title(r"Heat capacities based on ML depth and % of landmass"); ax.set_ylabel("m (equivalent water depth)")
        DefaultOutput.Stylize(self, ax)

class TemperatureOnEarthOutput(OutPut):
    def __init__(self, last_year_only=False):
        super().__init__()
        self.T_ext_series = []
        self.last_year_only = last_year_only

    def initialize(self, model):
        self.dt = model.config["dt_years"]
        self.x = model.x
        self.lat = np.degrees(np.arcsin(self.x))
        self.x_ext = np.r_[-1, self.x, 1] # Extended grid including poles
        self.lat_ext = np.r_[-90, self.lat, 90]

    def step(self, model, i):
        T_poles = model.funcs['poles_temperature'](model.T)
        self.T_ext_series.append(np.r_[T_poles[0], model.T, T_poles[1]] - 273.15) # Store in °C

    def finalize(self, model):
        T_poles = model.funcs['poles_temperature'](model.T)
        self.T_ext_series.append(np.r_[T_poles[0], model.T, T_poles[1]] - 273.15) # Store in °C
        self.T_ext_series = np.array(self.T_ext_series)
        if self.last_year_only:
            # Extract last year
            steps_per_year = int(round(1 / self.dt)) #Already a whole number up to machine precision
            last_slice = slice(-steps_per_year-1, None)
            self.T_ext_series = self.T_ext_series[last_slice]
        self.axes_funcs = [self.panel]

    def panel(self, ax):
        """Plot temperature on Earth surface (latitude vs time)."""
        return Earth.animate_on_earth(self.lat_ext, self.T_ext_series, self.dt, ax=ax, title="Animation of surface temperature", cbar_label="°C")

class VariableForcingOutput(TimeSeriesOutput):
    def __init__(self):
        super().__init__()

    def initialize(self, model):
        super().initialize(model)
        self.x = model.x
        self.lat = np.degrees(np.arcsin(self.x))
        self.x_ext = np.r_[-1, self.x, 1] # Extended grid including poles
        self.lat_ext = np.r_[-90, self.lat, 90]
        self.F_history = np.array(model.F_History[int(self.ctrl_years/model.config["dt_years"]):]) if model.config["ctrl_years"] > 0 else model.F_History
        self.start_year = model.start_year
        self.ctrl_years = model.config["ctrl_years"]

    def panel(self, ax):
        ax2 = ax.twinx()
        T_series = np.array(self.Tg_series[int(self.ctrl_years/self.dt+1):]) if self.ctrl_years > 0 else np.array(self.Tg_series)
        ax2.axhline(0, color='black', linestyle='--', label='Zero Forcing')
        ax.plot(np.arange(len(T_series)) * self.dt + self.start_year, T_series, label='Global Mean Temperature')
        ax.set_xlim(self.start_year, self.start_year + len(T_series) * self.dt)
        ax.set_ylim(np.max(np.abs(T_series-T_series[0])) * -1.1 + T_series[0], np.max(np.abs(T_series-T_series[0]))* 1.1 + T_series[0])
        ax.set_xlabel("Time [years]"); ax.set_ylabel("Temperature [°C]"); ax.grid(True)
        ax2.set_ylabel(" Forcing [W/m²] "); ax2.set_ylim(np.max(np.abs(self.F_history)) * -1.1, np.max(np.abs(self.F_history)) * 1.1)
        ax2.plot(np.arange(len(self.F_history)) * self.dt + self.start_year, self.F_history, label='Total Radiative Forcing', color='orange',linestyle='--')
        handles,labels = ax.get_legend_handles_labels(); handles2, labels2 = ax2.get_legend_handles_labels()
        ax.set_title("Global Mean Surface Temperature and Total Radiative Forcing")
        ax.legend(handles + handles2, labels + labels2, loc='lower right')

class HistoricalOutput(TimeSeriesOutput):
    def __init__(self):
        super().__init__()

    def initialize(self, model):
        super().initialize(model)
        self.start_year = 1750
        print("Loading temperature history from file:", model.config["temperature_history"])
        ds_giss = tools.netcdf_reader(model.config["temperature_history"])
        self.giss_time = pd.to_datetime(ds_giss['time'].values)
        self.giss_temp = np.squeeze(ds_giss['tempanomaly'].values)
        self.temperature_anomaly = np.interp(np.linspace(1880, 2025, int((2025-1880)/self.dt)), np.linspace(1880, 2025, len(self.giss_time)), self.giss_temp)
    
    def panel(self, ax):
        T_series = np.array(self.Tg_series[int(self.ctrl_years/self.dt+1):]) if self.ctrl_years > 0 else np.array(self.Tg_series)
        ax.plot(np.arange(len(T_series)) * self.dt + self.start_year, T_series, label='Simulation Temperature')
        ax.set_xlabel("Time [years]"); ax.set_ylabel("Temperature [°C]"); ax.grid(True)
        ax.set_xlim(1880,2024)
        ax.set_ylim(14.5,17)
        ax.plot(np.linspace(1880, 2025, int((2025-1880)/self.dt)), self.temperature_anomaly + self.Tg_series[int(self.ctrl_years/self.dt)], color='black', label='GISS Observed Temperature', linewidth=1.0)
        ax.set_title("Global Mean Temperatures")
        ax.legend()

       

def temp_fmt(n, p=1):
    start_fmt = end_fmt = "\033[0m"
    if n > 40: start_fmt = "\033[31m"
    elif n < 0: start_fmt = "\033[34m"
    return f"{start_fmt}{n:>{p+4}.{p}f}{end_fmt}"