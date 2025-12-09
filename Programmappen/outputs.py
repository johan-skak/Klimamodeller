# outputs.py
import numpy as np
import matplotlib.pyplot as plt
import textwrap
import os, datetime, re
from matplotlib.ticker import FixedLocator
import physics as phys
import AnimateOnEarth as Earth

def remove_ansi(text):
    """
    Remove ANSI escape sequences from text.
    
    Parameters
    ----------
    text : str
        Input string potentially containing ANSI codes.
    
    Returns
    -------
    clean_text : str
        Input string with ANSI codes removed.
    """
    ansi_escape = re.compile(r'\x1B\[[0-9;]*m') # Regex pattern for ANSI codes like \033[1m
    return ansi_escape.sub('', text)

def print_simulation_info(config, params, app_mode=False):
    """
    Produce a human-readable summary of configuration and parameters and return it as a string.
    If not in app_mode, also prints to console.

    Parameters
    ----------
    config : dict
        Configuration dictionary.
    params : dict
        Parameters dictionary.
    app_mode : bool, optional
        If True, suppress printing to console and only return the info string.

    Returns
    -------
    info_str : str
        Summary of configuration and parameters.
    """
    descs = phys.PARAM_DESCS

    # Compute column widths for aligned textual output.
    config_key_width   = max((len(k) for k in config), default=0)
    param_key_width    = max((len(k) for k in params), default=0)
    param_desc_width   = max((len(descs.get(k) or "") for k in params), default=0)
    param_value_width  = max((len(str(v)) for v in params.values()), default=0)

    param_block_width = param_key_width + param_desc_width + param_value_width + 6 # 6 extra characters including spacing and colon

    if not app_mode:
        # Preface describing potential mode-modified values. Not included in app mode or in summmary file.
        print(
            "Running simulation with the following configuration and parameters\n"
            "\033[1mNote\033[0m: some modes may have changed the values specified in the config and parameter files\n"
        )

    lines = []

    # --- Configuration block ---
    header = "=== EBM Model Configuration "
    lines.append(header + "=" * (param_block_width - len(header))) # Set width based on parameter block
    for key, value in config.items():
        formatted = f"{value:.3g}" if isinstance(value, float) else str(value) # Format floats nicely
        lines.append(f"{key:<{config_key_width}} : {formatted}") # Left-align keys
    lines.append("=" * param_block_width + "\n")

    # --- Parameters block ---
    header = "=== EBM Model Parameters "
    lines.append(header + "=" * (param_block_width - len(header))) # Set width based on parameter block
    for key, value in params.items():
        desc = f"({descs.get(key, '')})" # Parameter description in parentheses
        lines.append(f"{key:<{param_key_width}} {desc:<{param_desc_width+2}} : {value}") # Left-align keys and descriptions
    lines.append("=" * param_block_width + "\n")

    info_str = "\n".join(lines)

    if not app_mode:
        print(info_str, end="")
        print("\033[1mStarting\033[0m simulation...\n")

    return info_str

def aspect_ratio(n, target_ratio):
    """
    Compute an integer grid (rows, cols) such that n subplots approximate
    a target width/height ratio. Chooses between ceil/floor variants and picks
    the arrangement with least empty cells.

    Parameters
    ----------
    n : int
        Number of subplots.
    target_ratio : float
        Desired width/height ratio of the subplot grid.
    
    Returns
    -------
    (rows, cols) : tuple of int
        Number of rows and columns for the subplot grid.
    """
    h_num_top    = int(np.ceil(np.sqrt(target_ratio * n))) # Ceiling variant of horizontal count
    h_num_bottom = int(np.sqrt(target_ratio * n))          # Floor variant of horizontal count
    h_num = h_num_top if (-n) % h_num_top <= (-n) % h_num_bottom else h_num_bottom # Pick variant with fewer empty cells
    return int(np.ceil(n / h_num)), h_num

def generate_outputs_data(axes_funcs, summaries, outdir="", sim_info=""):
    """
    Run all axis-drawing functions, assemble matplotlib figure, and produce the
    combined summary text. Used both for script-mode saving and app-mode display.

    Parameters
    ----------
    axes_funcs : list of callable
        Functions that draw on matplotlib axes.
    summaries : list of str
        Summary strings to be combined.
    outdir : str, optional
        Output directory for saving files.
    sim_info : str, optional
        Simulation information string to include in summary header.

    Returns
    -------
    fig : matplotlib.figure.Figure or None
        The assembled figure containing all subplots, or None if no axes functions were provided.
    summary : str
        Combined summary text with ANSI codes.
    clean_summary : str or None
        Summary text with ANSI codes removed, or None if no summaries were provided.
    """

    # --- Plot assembly ---
    if axes_funcs:
        v, h = aspect_ratio(len(axes_funcs), 1) # Target approx. square layout
        fig, axs = plt.subplots(v, h, figsize=(6*h, 27/8*v)) # But let panels have 16:9 aspect ratio
        axs = np.atleast_1d(axs).flatten()

        for func, ax in zip(axes_funcs, axs):
            func(ax) # Draw on the provided axis
        for ax in axs[len(axes_funcs):]:
            ax.axis('off') # Turn off unused axes

        fig.tight_layout()
    else:
        fig = None

    # --- Summary assembly ---
    summary = ""
    if summaries:
        for sum in summaries:
            summary += textwrap.dedent(sum) + "\n"

        clean_summary = remove_ansi(summary) # Obtain clean version without ANSI codes which are only suitable for terminal display

        # Compute header separator width.
        max_line_length = max(len(line) for line in clean_summary.split("\n"))
        header = "=== EBM Summary "
        prefix = (
            sim_info +
            f"Output generated on {datetime.datetime.now():%Y-%m-%d %H:%M:%S}\n\n" +
            header + "=" * (max_line_length - len(header)) + "\n" # Set width based on longest line
        )
        clean_summary = prefix + clean_summary # Add header to clean summary for file saving

    summary += f"Figures and summary saved in \033[4m{outdir}\033[0m\n"

    return fig, summary, clean_summary if summaries else ""

def run_all_outputs(outputs, outdir, sim_info="", runtime=None, app=False):
    """
    Driver: collects axis functions and summary strings from all outputs,
    either saves them (script mode) or returns them (app mode).

    Parameters
    ----------
    outputs : list
        Output handler instances (e.g. for plots, files, summaries).
    outdir : str
        Output directory for saving files.
    sim_info : str, optional
        Simulation information string to include in summary header.
    runtime : float or None, optional
        Runtime of the simulation in seconds.
    app : bool, optional
        If True, return axis functions and summaries instead of saving files.

    Returns
    -------
    None or tuple
        Returns None if app is False. If app is True, returns a tuple (axes_funcs, summaries).
    
    Notes
    -----
    - In script mode (app=False), saves a figure "ebm_panels.png" and a text file
      "summary.txt" in the specified output directory.
    - In app mode (app=True), returns the collected axis functions and summaries for
      further processing or display in the application.
    """

    axes_funcs = [f for o in outputs for f in o.axes_funcs]
    summaries  = [s for o in outputs for s in o.summaries]

    if not app:
        fig, summary, clean_summary = generate_outputs_data(axes_funcs, summaries, outdir, sim_info)

        timedesc = f" in {runtime:.2f} seconds" if runtime else "" # Optional runtime description
        print(f"\033[1mFinished\033[0m simulation{timedesc}. Generating outputs and saving in the \033[4m{outdir}\033[0m folder")

        # Save figure and summary text
        os.makedirs(outdir, exist_ok=True)
        fig.savefig(f"{outdir}/ebm_panels.png", dpi=150)
        with open(f"{outdir}/summary.txt", "w", encoding="utf-8") as f:
            f.write(clean_summary)

        print(summary)
    else:
        return axes_funcs, summaries

class OutPut:
    """
    Base class for all outputs. Output objects accumulate:
        - axes_funcs : list of callables(ax)
        - summaries  : list of strings or callables returning summary text
    Modes can add outputs by appending to the outputs list.
    """
    def __init__(self):
        self.summaries = []
        self.axes_funcs = []

    def initialize(self, model): pass
    def step(self, model, i): pass
    def finalize(self, model): pass

class DefaultOutput(OutPut):
    """
    Generates multi-panel diagnostics comparing initial, control, and final
    states, including temperature, OLR, albedo, heat transport, and convergence.
    """
    def __init__(self):
        super().__init__()
        self.diags = {} # Dictionary of dictionaries for initial, control, and final diagnostics

    def initialize(self, model):
        # Store initial diagnostics.
        self.diags["init"] = self.simulation_diagnostics(model.funcs, model.x, model.T, model.params)
        self.x   = model.x
        self.lat = np.degrees(np.arcsin(self.x))
        self.Forcing_on = ( # Whether to include end of control phase diagnostics
            model.config["ctrl_years"] > 0 and
            (model.params.get('F', 1) != 0 or model.params['S1'] != model.params['S0'])
        )

    def step(self, model, i):
        # Capture mid-state if a control phase exists.
        if i == model.ctrl_nsteps - 1:
            self.diags["mid"] = self.simulation_diagnostics(model.funcs, model.x, model.T, model.params)

    def finalize(self, model):
        # Final state diagnostics.
        self.diags["end"] = self.simulation_diagnostics(model.funcs, model.x, model.T, model.params)

        # Global warming metrics.
        self.dt_global = self.diags["end"]["T_mean"] - self.diags["mid"]["T_mean"]
        self.polar_ampl = (
            (self.diags["end"]["T_poles"][1] - self.diags["mid"]["T_poles"][1] - self.dt_global)
            / self.dt_global if self.dt_global != 0 else np.nan
        )

        self.lat_ext = np.r_[-90, self.lat, 90] # Extended latitude including poles

        # Case lists depend on whether forcing was applied.
        self.cases  = ["init", "mid", "end"] if self.Forcing_on else ["init", "end"]
        self.labels = ["Initial", "Control", "Final"] if self.Forcing_on else ["Initial", "Final"]
        self.colors = ["C2", "C0", "C1"] if self.Forcing_on else ["C2", "C1"]

        # Extend temperature arrays to include poles.
        for case in self.cases:
            self.diags[case]["T_ext"] = np.r_[
                self.diags[case]["T_poles"][0],
                self.diags[case]["T"],
                self.diags[case]["T_poles"][1]
            ]

        # Panels and summary
        self.axes_funcs = [self.panel1, self.panel2, self.panel3, self.panel4, self.panel5]
        if self.Forcing_on:
            self.axes_funcs.append(self.panel6) # Add delta T panel if two stages
        self.summaries = [self.summarize(model, self.diags)]
    
    def summarize(self, model, diags):
        """
        Produce a textual summary of the simulation results.
        """
        has_mid = "mid" in diags and diags["mid"] is not None
        end = diags["end"]

        if has_mid:
            mid = diags["mid"]

            return textwrap.dedent(f"""
                Years (control, forced): ({model.config['ctrl_years']}, {model.config['years']-model.config['ctrl_years']})
                Grid points nx: {model.config['nx']}, Δt (years): {model.config['dt_years']}

                Control global mean T (°C): {temp_fmt(mid['T_mean']-273.15,1)}
                Final global mean T (°C): {temp_fmt(end['T_mean']-273.15,1)}
                ΔT global (°C): {end['T_mean']-mid['T_mean']:.1f}

                North pole T control / forced (°C): {temp_fmt(mid['T_poles'][1]-273.15,1)} / {temp_fmt(end['T_poles'][1]-273.15,1)}
                North polar amplification ( (ΔT_pole - ΔT_global)/ΔT_global ): {self.polar_ampl:.3f}

                Outgoing longwave radiation (OLR) control / forced (W m⁻²): {mid['olr'].mean():.0f} / {end['olr'].mean():.0f}
                Planetary albedo control / forced: {np.average(mid['alpha'], weights=mid['Q_x']):.3f} / {np.average(end['alpha'], weights=end['Q_x']):.3f}
                Diffusivity control / forced (W m⁻² K⁻¹): {mid['D']:.3f} / {end['D']:.3f}
            """)

        # ---- Single stage simulation ----
        return textwrap.dedent(f"""
            Years simulated: {model.config['years']}
            Grid points nx: {model.config['nx']}, Δt (years): {model.config['dt_years']}

            Final global mean T (°C): {temp_fmt(end['T_mean']-273.15,1)}
            North pole temperature (°C): {temp_fmt(end['T_poles'][1]-273.15,1)}

            Outgoing longwave radiation (OLR) (W m⁻²): {end['olr'].mean():.0f}
            Planetary albedo: {np.average(end['alpha'], weights=end['Q_x']):.3f}
            Diffusivity (W m⁻² K⁻¹): {end['D']:.3f}
        """)

    # Panel 1: Temperature profiles  (°C)
    def panel1(self, ax):
        """Plot initial, control and final temperature profiles."""
        for case, label, color in zip(self.cases, self.labels, self.colors):
            ax.plot(self.lat_ext, self.diags[case]["T_ext"] - 273.15, label=label, color=color)
        ax.axhline(0, color="#00aeff", linestyle='--', alpha=0.7) # 0 °C line
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

    # Diagnostics helper used by DefaultOutput and SeasonalOutput
    def simulation_diagnostics(self, funcs, x, T, params, model=None, i=0):
        """
        Computes albedo, OLR, diffusivity, convergence, heat transport, etc.,
        for a given temperature field T at iteration i.

        Parameters
        ----------
        funcs : dict
            Dictionary of physics functions to use.
        x : np.ndarray
            Spatial grid points.
        T : np.ndarray
            Temperature profile.
        params : dict
            Model parameters.
        model : Model or None, optional
            Model instance for context, if needed.
        i : int, optional
            Iteration index.

        Returns
        -------
        dict
            Dictionary containing diagnostic variables:
                - T : Temperature profile
                - alpha : Albedo profile
                - olr : Outgoing longwave radiation profile
                - conv : Heat flux convergence profile
                - MHTrans_PW : Meridional heat transport (PW)
                - D : Diffusivity profile
                - T_mean : Mean temperature
                - T_poles : Pole temperatures
                - Q_x : Insolation profile
        """
        alpha = funcs['albedo_from_T'](T, x, k1=params['k1'], model=model, i=i)
        dTloc = funcs['deltaT_of_Ts'](T, k3=params['k3'], model=model, i=i)
        olr = phys.SIGMA * (T - dTloc)**4
        D = funcs["diffusion_from_T"](T, params['D0'], params['k2'], model=model, i=i)

        aL, bL, cL = funcs['build_diffusion_tridiag'](x, D)
        conv = funcs['apply_L_to_T'](aL, bL, cL, T)

        MHTrans_PW = funcs['meridional_transport_PW'](T, x, D, model=model, i=i)
        T_mean = T.mean()
        T_poles = funcs['poles_temperature'](T, model=model, i=i)
        Q_x = funcs['Q_x'](x, params['S0'], model=model, i=i)

        return dict(
            T=T, alpha=alpha, olr=olr, conv=conv,
            MHTrans_PW=MHTrans_PW, D=D, T_mean=T_mean,
            T_poles=T_poles, Q_x=Q_x
        )

class TimeSeriesOutput(OutPut):
    """
    Stores global mean temperature over time and produces a single time-series panel.
    """
    def __init__(self):
        super().__init__()
        self.Tg_series = []

    def initialize(self, model):
        self.dt = model.config["dt_years"]
        self.Tg_series.append(model.T.mean() - 273.15)
        self.Forcing_on = ( # Whether to include end of control phase diagnostics
            model.config["ctrl_years"] > 0 and
            (model.params.get('F', 1) != 0 or model.params['S1'] != model.params['S0'])
        )
        self.ctrl_years = model.config["ctrl_years"]

    def step(self, model, i):
        self.Tg_series.append(model.T.mean() - 273.15)

    def finalize(self, model):
        self.axes_funcs = [self.panel]

    def panel(self, ax):
        """Plot global mean temperature time series."""
        ax.plot(np.arange(len(self.Tg_series)) * self.dt, self.Tg_series, label='Global Mean Temperature')
        ax.set_title("Global Mean Surface Temperature")
        ax.set_xlabel("Time (years)"); ax.set_xlim(0, len(self.Tg_series) * self.dt); ax.set_ylabel("°C"); ax.grid(True)
        if self.Forcing_on:
            ax.axvline(self.ctrl_years, color='k', linestyle='--', label='Forcing On')
            ax.legend()

class SeasonalOutput(OutPut):
    """
    Produces seasonal-cycle diagnostics over the last simulated year.
    
    The class stores full time series of all relevant fields, then in finalize():
        - extracts the last year only (using dt)
        - computes four key seasonal phases
        - prepares both spatial profiles and time-series diagnostics
    """

    def __init__(self):
        super().__init__()
        # t : list of times (in model years)
        # series : dict[str -> list of arrays], accumulating one element to each array per step
        self.t = []
        self.series = {
            key: [] for key in [
                "T", "T_ext", "olr", "alpha", "conv", "MHTrans_PW", "D",
                "Q_x", "Q_x_ext"
            ]
        }

    def initialize(self, model):
        # Store grid and extended grid (latitudes including poles)
        self.x = model.x
        self.x_ext = np.r_[-1, self.x, 1]
        self.lat = np.degrees(np.arcsin(self.x))
        self.lat_ext = np.r_[-90, self.lat, 90]

        # Temporal spacing (years per step)
        self.dt = model.config["dt_years"]

    def step(self, model, i):
        """
        Store a full snapshot of all key diagnostics at each time step.

        This is intentionally heavy on memory, as it stores the full time series
        of all relevant fields for later seasonal analysis.
        But in this way the code still works even if another mode suddenly breaks the simulation loop.
        """
        # Physical time of this step
        self.t.append((i + 1) * self.dt)

        # Compute the same diagnostic fields used by DefaultOutput (but without
        # the full DefaultOutput object)
        diags = DefaultOutput().simulation_diagnostics(
            model.funcs, model.x, model.T, model.params, model=model, i=i
        )

        # Store main temperature field
        self.series["T"].append(model.T.copy())

        # Store per-diagnostic arrays
        for key in ["olr", "alpha", "conv", "MHTrans_PW", "D", "Q_x"]:
            self.series[key].append(diags[key])

        # Temperature including poles (3-point extension)
        self.series["T_ext"].append(
            np.r_[diags["T_poles"][0], diags["T"], diags["T_poles"][1]]
        )

        # Solar input including poles (seasonally varying, S0 or S1 after forcing)
        current_S = model.params['S0'] if i < model.ctrl_nsteps else model.params['S1']
        self.series["Q_x_ext"].append(
            phys.seasonal_Q(self.x_ext, current_S, model, i)
        )

    def finalize(self, model):
        """
        Convert lists to arrays, isolate the last model year, define seasonal
        phases and latitudes of interest, then prepare figure-generating functions.
        """
        # Convert series lists -> numpy arrays of shape (timesteps, ...)
        for key in self.series:
            self.series[key] = np.array(self.series[key])

        # Number of time steps per simulated year (should be an integer)
        steps_per_year = int(round(1 / self.dt))

        # Last-year interval
        last_slice = slice(-steps_per_year - 1, None) # Include one extra step for continuity in plots

        # Extract last-year fields into self.last dict
        self.last = {key: arr[last_slice] for key, arr in self.series.items()}
        self.t_last = np.array(self.t[last_slice])

        # Define indices corresponding to seasonal key phases
        quarter = steps_per_year // 4
        self.phases = {
            "Spring eqx": 0,
            "Summer sol": quarter,
            "Autumn eqx": 2 * quarter,
            "Winter sol": 3 * quarter,
        }

        # Pick specific locations for time-series plots. Maps name -> index in latitude grid.
        self.locs = {
            "Global mean":      None, # mean over latitudes
            "Equator":          np.argmin(np.abs(self.lat - 0)),
            "Denmark (56°N)":   np.argmin(np.abs(self.lat - 56)),
            "North pole":       -1,
            "South pole":       0,
        }

        # Prepare all panel functions
        self.axes_funcs = [
            self.panel1, self.panel2, self.panel3, self.panel4,
            self.panel5, self.panel6, self.panel7, self.panel8
        ]

        # Prepare summary
        self.summaries = [self.summarize(model)]

    def plot_profiles(self, ax, field, ylabel, title):
        """
        Plot the selected field across latitude for each seasonal phase.
        'field' corresponds to a key in self.last (e.g. 'T_ext').

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The axes on which to plot.
        field : str
            Key in self.last corresponding to the field to plot (e.g. 'T_ext').
        ylabel : str
            Label for the y-axis.
        title : str
            Title of the plot.

        Notes
        -----
        - For 'T_ext', values are converted to °C.
        - For 'MHTrans_PW', the data is stored as a tuple (latitudes, values) since the latitude grid is different.
        """
        # Iterate over seasonal phases
        for label, idx in self.phases.items():
            # Extract data for the given phase
            data = self.last[field][idx]

            # Choose correct latitude grid depending on array length
            lat = self.lat if len(data) == len(self.lat) else self.lat_ext

            if field == "MHTrans_PW":
                # MHTrans_PW is stored as a tuple (lat, values)
                ax.plot(data[0], data[1], label=label)
            else:
                values = data - 273.15 if field == "T_ext" else data
                ax.plot(lat, values, label=label)

            # Add 0°C line for temperature
            if field == "T_ext":
                ax.axhline(0, color="#00aeff", linestyle='--', alpha=0.7)

        ax.set_title(title)
        ax.set_ylabel(ylabel)
        DefaultOutput.Stylize(self, ax)

    # Panels 1–6: spatial seasonal profiles
    def panel1(self, ax): self.plot_profiles(ax, "T_ext", "°C", "Seasonal Temperature Profiles")
    def panel2(self, ax): self.plot_profiles(ax, "olr", "W/m²", "Seasonal OLR Profiles")
    def panel3(self, ax): self.plot_profiles(ax, "alpha", "Albedo", "Seasonal Albedo Profiles")
    def panel4(self, ax): self.plot_profiles(ax, "MHTrans_PW", "PW (10¹⁵ W)", "Seasonal Meridional Heat Transport")
    def panel5(self, ax): self.plot_profiles(ax, "conv", "W/m²", "Seasonal Heat Flux Convergence")
    def panel6(self, ax): self.plot_profiles(ax, "Q_x_ext", "W/m²", "Seasonal Solar Irradiance")

    # ---- Time-series plotting helper ----

    def plot_time_series(self, ax, field, ylabel, title, mean_is_zero=False):
        """
        For each named latitude group, plot the time series over the final year.
        If mean_is_zero=True, subtract the time-mean for each series to show
        only seasonal anomalies.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The axes on which to plot.
        field : str
            Key in self.last corresponding to the field to plot (e.g. 'T').
        ylabel : str
            Label for the y-axis.
        title : str
            Title of the plot.
        mean_is_zero : bool, optional
            If True, subtract the time-mean for each series to show only seasonal anomalies (default is False).
        """
        # Iterate over locations
        for name, idx in self.locs.items():
            # Global mean -> mean over latitude
            if idx is None:
                series_raw = self.last[field].mean(axis=1)
            else:
                series_raw = self.last[field][:, idx]

            # Remove mean if showing anomalies
            mean_offset = series_raw.mean() if mean_is_zero else 0
            series = series_raw - mean_offset

            ax.plot(self.t_last, series, label=name)

        # Set x tick labels as dates through the year
        xticks = self.t_last[np.linspace(0, len(self.t_last) - 1, 7, dtype=np.int16)]
        ax.set_xticks(xticks)
        ax.set_xticklabels([self.date_from_fraction(t) for t in np.linspace(0, 1, 7)])

        ax.set_xlabel("Date (during the last simulated year)")
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(True)
        ax.set_xlim(self.t_last[0], self.t_last[-1])

    # Panels 7–8: seasonal time-series plots
    def panel7(self, ax): self.plot_time_series(ax, "Q_x", "W/m²", "Seasonal Solar Irradiance Time Series")
    def panel8(self, ax): self.plot_time_series(ax, "T", "°C", "Seasonal Temperature Variation Time Series", mean_is_zero=True)

    def summarize(self, model):
        """
        Produce a multi-line summary reporting seasonal extrema and timing
        for several latitudes and for global mean fields.
        """
        t = self.t_last

        def fmt(series):
            # Identify timing of extrema within the last year
            min_idx, max_idx = series.argmin(), series.argmax()
            min_time = t[min_idx] % 1
            max_time = t[max_idx] % 1

            return (temp_fmt(series.mean(), 2) + "°C "
                    "(min " + temp_fmt(series.min()) +
                    f" on {self.date_from_fraction(min_time):>5} ({min_time:>4.2f}y), "
                    f"max " + temp_fmt(series.max()) +
                    f" on {self.date_from_fraction(max_time):>5} ({max_time:>4.2f}y))")

        # Convert arrays to °C for readability
        global_T  = self.last["T"].mean(axis=1) - 273.15
        equator_T = self.last["T"][:, self.locs["Equator"]] - 273.15
        den_T     = self.last["T"][:, self.locs["Denmark (56°N)"]] - 273.15
        north_T   = self.last["T"][:, self.locs["North pole"]] - 273.15
        south_T   = self.last["T"][:, self.locs["South pole"]] - 273.15

        return textwrap.dedent(f"""
        === Seasonal Diagnostics (last year) ===
        Modes: {model.config.get("modes")}
        Years run: {model.config["years"]}, grid points: {model.config["nx"]}, Δt (years): 1 / {round(1 / self.dt)}

        {"Global mean temperature:":<24}{fmt(global_T)}
        {"Equator temperature:":<24}{fmt(equator_T)}
        {"Denmark (56°N):":<24}{fmt(den_T)}
        {"North pole:":<24}{fmt(north_T)}
        {"South pole:":<24}{fmt(south_T)}

        Last-year mean OLR:    {self.last['olr'].mean():.1f} W/m²
        Last-year mean albedo: {self.last['alpha'].mean():.3f}
        Last-year mean D:      {self.last['D'].mean():.3f} W m⁻² K⁻¹
        """)

    def date_from_fraction(self, frac):
        """
        Convert fractional year since spring equinox into a calendar date.
        Only used for axis labelling and summaries.
        """
        start = datetime.date(2000, 3, 21)   # fixed reference
        days_in_year = 365
        offset = int(round(frac * days_in_year))
        date = start + datetime.timedelta(days=offset)
        return date.strftime("%b %d")

class SeaDepthOutput(OutPut):
    """
    Computes and plots the effective mixed-layer depth (converted to an equivalent
    water depth) as a function of latitude. This is derived from the heat-capacity
    profile, which depends on local temperature and land fraction.

    The object stores T_ext (temperature including poles) at *each* model step.
    At finalize(), it chooses representative seasonal phases (if SeasonalVariation
    mode was active) and generates a single panel.
    """

    def __init__(self):
        super().__init__()
        # List of arrays of T including poles, one per timestep
        self.T_ext = []

    def initialize(self, model):
        # k1 is used in heat capacity / mixed-layer depth conversion
        self.k1 = model.params["k1"]

    def step(self, model, i):
        """
        Store full temperature including poles each timestep.
        Using extended grid ensures land fraction and polar behaviour are handled.
        """
        T = model.T.copy()
        T_poles = model.funcs['poles_temperature'](T, model=model, i=i)
        self.T_ext.append(np.r_[T_poles[0], T, T_poles[1]])

    def finalize(self, model):
        """
        After the run, choose which time indices to plot:
        - If SeasonalVariation is active: plot 4 seasonal phases.
        - Otherwise: plot a single curve representing the end state.
        """
        # For seasonal mode, define four phase indices (same pattern as SeasonalOutput)
        if "SeasonalVariation" in model.config["modes"]:
            quarter = int(round(1 / model.config['dt_years'])) // 4
            self.phases = {
                "Spring eqx": 0,
                "Summer sol": quarter,
                "Autumn eqx": 2 * quarter,
                "Winter sol": 3 * quarter,
            }
            steps_per_year = int(round(1 / model.config['dt_years']))
            last_slice = slice(-steps_per_year - 1, None) # Include one extra step for continuity
            self.T_ext = self.T_ext[last_slice] # Keep only last simulated year
        else:
            # Non-seasonal run: only final state without label
            self.phases = {"": -1}
        

        # Precompute extended lat grid
        self.x_ext = np.r_[-1, model.x, 1]
        self.lat_ext = np.degrees(np.arcsin(self.x_ext))

        # One plotting function only
        self.axes_funcs = [self.panel]

    def panel(self, ax):
        """
        Compute and plot the “equivalent water depth” from the heat capacity profile.
        phys.heat_capacity_profile returns (J/m²/K); dividing by C_M converts to meters.
        """
        for label, idx in self.phases.items():
            # idx = -1 for non-seasonal runs → last timestep
            depth = phys.heat_capacity_profile(
                self.x_ext, self.T_ext[idx], self.k1
            ) / phys.C_M
            ax.plot(self.lat_ext, depth, label=label)

        ax.set_title(r"Heat capacities based on ML depth and % of landmass")
        ax.set_ylabel("m (equivalent water depth)")
        DefaultOutput.Stylize(self, ax)

class TemperatureOnEarthOutput(OutPut):
    """
    Produces an animation panel showing the surface temperature on Earth, built on the
    `Earth.animate_on_earth` helper. This class is only applicable in app mode.
    Stores full T_ext (°C) series in order to create the animation.
    """
    def __init__(self, last_year_only=False):
        super().__init__()
        self.T_ext_series = []
        self.last_year_only = last_year_only

    def initialize(self, model):
        self.dt = model.config["dt_years"]
        self.x = model.x
        self.lat = np.degrees(np.arcsin(self.x))
        self.x_ext  = np.r_[-1, self.x, 1]
        self.lat_ext = np.r_[-90, self.lat, 90]

    def step(self, model, i):
        # Store temperature series including poles, already in °C
        T_poles = model.funcs['poles_temperature'](model.T)
        self.T_ext_series.append(
            np.r_[T_poles[0], model.T, T_poles[1]] - 273.15
        )

    def finalize(self, model):
        # Append final state too
        T_poles = model.funcs['poles_temperature'](model.T)
        self.T_ext_series.append(
            np.r_[T_poles[0], model.T, T_poles[1]] - 273.15
        )
        self.T_ext_series = np.array(self.T_ext_series)

        # Optionally reduce to last simulated year only
        if self.last_year_only:
            steps_per_year = int(round(1 / self.dt))
            last_slice = slice(-steps_per_year - 1, None) # Include one extra step for continuity
            self.T_ext_series = self.T_ext_series[last_slice] # Only keep last year

        self.axes_funcs = [self.panel_wrapper]

    def panel_wrapper(self, ax):
        """
        Draw the Earth-surface temperature animation panel using the external Earth
        utility. Returns a wrapper for the artist (an html-element) for Streamlit-mode playback.
        (Thus this function is a wrapper for a wrapper.)
        """
        return Earth.animate_on_earth(
            self.lat_ext, self.T_ext_series, self.dt,
            ax=ax, title="Animation of surface temperature", cbar_label="°C"
        )

class SeasonalTempOnEarthOutput(TemperatureOnEarthOutput):
    """
    Same as TemperatureOnEarthOutput, but always last-year-only.

    TemperatureOnEarthOutput produces an animation panel showing the surface temperature on Earth, built on the
    `Earth.animate_on_earth` helper. This class is only applicable in app mode.
    Stores full T_ext (°C) series in order to create the animation.
    """
    def __init__(self):
        super().__init__(last_year_only=True)

output_registry = { # Maps output classes to (category, priority)
    DefaultOutput:              ("Default", 0),
    TimeSeriesOutput:           ("Time Series", 0),
    SeasonalOutput:             ("Default", 1), # SeasonalOutput has higher priority than DefaultOutput
    TemperatureOnEarthOutput:   ("Temperature on Earth", 0),
    SeasonalTempOnEarthOutput:  ("Temperature on Earth", 1),
    SeaDepthOutput:             ("Sea Depth", 0)
}

def collect_outputs(modes_list, app_mode):
    """
    Collect outputs from all modes + required defaults, group them by category,
    and keep only the highest-priority output per category.

    Parameters
    ----------
    modes_list : list
        A list of modes (instances of Mode)
    app_mode : bool
        A boolean whether the program is in app mode
    
    Returns
    -------
    A list of outputs (instances of OutPut)

    Mechanism
    ---------

    1. Collect output objects from all modes:\n
       Each mode instance (subclasses of Mode) has an attribute 'outputs'
       holding instances of OutPut subclasses. These are appended verbatim.

    2. Add required baseline outputs:\n
       Always include a fresh DefaultOutput and TimeSeriesOutput.
       When in app_mode, also add TemperatureOnEarthOutput.
       (These are appended before deduplication.)

    3. Perform category-based selection:\n
       output_registry maps each output class → (category, priority).
       Categories define which outputs compete with each other.
       Priorities determine which one wins.

         Example:
             DefaultOutput → ("Default", 0)
             SeasonalOutput → ("Default", 1)

         Both are in category "Default".
         Priority 1 > 0, so SeasonalOutput overrides DefaultOutput.

    4. best_outputs dictionary:\n
         key   = category name (string)
         value = surviving output object (instance)

        On each object:
    - If category is new: insert it.
    - If category exists:
        Compare priorities via the registry.
        - If same priority but different classes → raise ValueError
        (to enforce strict resolution rules)
        - Else keep the higher-priority object.

    5. Return:\n
         The *instances* stored in best_outputs.values().
         (Important: no re-instantiation; the objects retain their internal state.)

    Notes
    -----
    - The ordering of categories in the output list is determined by
       dictionary insertion order, which is deterministic in Python ≥3.7.
    - The mechanism is stable: earlier objects lose only if a strictly higher
       priority exists later in the list.
    """
    # Gather all outputs attached to modes.
    output_list = [o for m in modes_list for o in m.outputs]

    # Insert mandatory baseline outputs.
    output_list = [DefaultOutput()] + output_list
    output_list.append(TimeSeriesOutput())
    if app_mode:
        output_list.append(TemperatureOnEarthOutput())

    best_outputs = {}

    for obj in output_list:
        cls = type(obj)
        try:
            category, priority = output_registry[cls]
        except KeyError:
            raise ValueError(f"Unknown output class: {cls.__name__}")

        if category not in best_outputs:
            best_outputs[category] = obj
        else:
            current = best_outputs[category]
            current_priority = output_registry[type(current)][1]

            if priority == current_priority and type(obj) != type(current):
                raise ValueError(
                    f"Duplicate output in category {category} with equal priority: "
                    f"{cls.__name__} and {type(current).__name__}"
                )

            if priority > current_priority:
                best_outputs[category] = obj

    return list(best_outputs.values())

def temp_fmt(n, p=1):
    """
    Apply conditional ANSI colouring to temperature values.
    """
    start_fmt = end_fmt = "\033[0m" # Reset
    if n > 40: start_fmt = "\033[31m" # Red for hot
    elif n < 0: start_fmt = "\033[34m" # Blue for cold
    return f"{start_fmt}{n:>{p+4}.{p}f}{end_fmt}"
