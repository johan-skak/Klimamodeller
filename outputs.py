# outputs.py
import numpy as np
import matplotlib.pyplot as plt
import textwrap # For dedenting summary text
import os
from matplotlib.ticker import FixedLocator # For custom minor ticks
import physics as phys

def print_simulation_info(config, params):
    print("=== EBM Model Configuration =======")
    for key, value in config.items():
        print(f"{key} \t: {value}")

    print("=== EBM Model Parameters ========")
    for key, value in params.items():
        print(f"{key} \t: {value}")

    print("===================================")
    print("Starting simulation...")

def run_all_outputs(outputs, outdir):
    print(f"Finished simulation. Generating outputs in {outdir}")
    os.makedirs(outdir, exist_ok=True)

    axes = [ax for o in outputs for ax in o.axes] if outputs else []
    if axes:
        v_num = int(np.round(np.sqrt(2*len(axes)))) # Aim for 2:1 aspect ratio
        h_num = int(np.ceil(len(axes) / v_num))
        fig, axs = plt.subplots(h_num, v_num, figsize=(6*h_num, 4*v_num))
        axs = np.atleast_1d(axs).flatten()
        
        for axfunc, subplot_ax in zip(axes, axs):
            axfunc(subplot_ax)  # plotting function should accept "ax"
        for ax in axs[len(axes):]:
            ax.axis('off')  # Turn off unused subplots
        fig.tight_layout()
        fig.savefig(f"{outdir}/ebm_panels.png", dpi=150)

    summaries = [s for o in outputs for s in o.summaries] if outputs else []
    if summaries:#Also print the summary
        summary = "=== EBM Summary ===\n\n"
        for sfunc in summaries:
            summary += textwrap.dedent(sfunc()) + "\n"
        summary += f"Figures and summary saved in {outdir}/ebm_panels.png\n"
        print(summary)
        with open(f"{outdir}/summary.txt", "w", encoding="utf-8") as f:
            f.write(summary)

class OutPut:
    def __init__(self, mods=[]): #mods can be a single modifier function (Parameters: output object, model object) or a list of them
        self.axes = []      # List of functions to plot on axes
        self.summaries = [] # List of functions to write summaries
        self.mods = [mods] if callable(mods) else mods # List of modifier functions

    def initialize(self, model): pass
    def step(self, model, i): pass
    def finalize(self, model):
        for m in self.mods: # Call modifier functions if any
            m(self, model)
    
class DefaultOutput(OutPut):
    def __init__(self, mods=[]):
        super().__init__(mods)
        self.Tg_series = []
        self.diags = {}
    
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
        self.axes = [self.panel1, self.panel2]
        self.summaries = [lambda: self.summarize(model, self.diags)]
        super().finalize(model)
    
    def summarize(self, model, diags):
        return textwrap.dedent(f"""
        === EBM Summary ===
        Years (control, forced): ({model.config['years']//2}, {(model.config['years']+1)//2})
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

    def panel1(self, ax):
        """Plot initial, control and final temperature profiles."""
        lat_ext = np.r_[-90, self.lat, 90]
        for case, label in zip(["_mid", "_end", "_init"], ["Control", "Final", "Initial"]):
            T_ext = np.r_[self.diags[case]["T_poles"][0], self.diags[case]["T"], self.diags[case]["T_poles"][1]] - 273.15
            ax.plot(lat_ext, T_ext, label=label)
        ax.set_title("Temperature profile")
        ax.set_ylabel("°C")
        self.Stylize(ax)

    def panel2(self, ax):
        """Plot control and final OLR profiles."""
        ax.plot(self.lat, self.diags["_mid"]['olr'], label='Control')
        ax.plot(self.lat, self.diags["_end"]['olr'], label='Forced')
        ax.set_title('Outgoing Longwave Radiation (OLR)'); ax.set_ylabel('W/m²')
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
        Q_x = funcs['Q_x'](x, params['S0'])
        return dict(T=T, alpha=alpha, olr=olr, conv=conv, MHTrans_PW=MHTrans_PW, D=D, T_mean=T_mean, T_poles=T_poles, Q_x=Q_x)

class TimeSeriesOutput(OutPut):
    def __init__(self, mods=[]):
        super().__init__(mods)
        self.Tg_series = []

    def initialize(self, model):
        self.dt = model.config["dt_years"]
        self.Tg_series.append(model.T.mean() - 273.15)

    def step(self, model, i):
        self.Tg_series.append(model.T.mean() - 273.15)

    def finalize(self, model):
        self.axes = [self.panel]
        super().finalize(model)

    def panel(self, ax):
        """Plot global mean temperature time series."""
        ax.plot(np.arange(len(self.Tg_series)) * self.dt, self.Tg_series)
        ax.set_title("Global Mean Surface Temperature")
        ax.set_xlabel("Time (years)"); ax.set_xlim(0, len(self.Tg_series) * self.dt); ax.set_ylabel("°C"); ax.grid(True)

class SeasonalOutput(OutPut):
    def __init__(self):
        super().__init__()
        self.history = []

    def step(self, model, t):
        lat = np.degrees(np.arcsin(model.x))
        T = model.T - 273.15
        self.history.append((t, T.copy()))

    def finalize(self, model):
        # Example: plot last temperature profile
        t, T = self.history[-1]
        lat = np.degrees(np.arcsin(model.x))
        plt.plot(lat, T)
        plt.title(f"Seasonal profile at step {t}")
        plt.xlabel("Latitude"); plt.ylabel("°C")
        plt.show()

#Default function needed for default model with forcing
def vline(output_obj, model):
    """Modifier: wrap first panel to add a vertical line at half the years."""
    old_panel = output_obj.axes[0]   # save the original function
    
    def new_panel(ax):
        old_panel(ax)  # call original panel
        ax.axvline(model.config['years'] // 2, color='k', ls='--')
    
    output_obj.axes[0] = new_panel   # replace it with the wrapped version