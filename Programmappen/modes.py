# modes.py
import csv
import numpy as np
import outputs
import physics as phys
import os

class Mode:
    """
    Base class for all modes.

    A mode can:
      - Modify config/params before the simulation starts (initialize)
      - Modify model state on each time step (step)
      - Perform cleanup or final calculations (finalize)
      - Contribute output handlers (self.outputs)
      - Reject incompatible combinations (check_compatibility)

    Subclasses override the relevant methods.
    """
    def __init__(self, app_mode=False):
        self.outputs = []         # List of output instances provided by the mode
        self.app_mode = app_mode  # Used for Streamlit-specific outputs

    def __str__(self):
        return self.__class__.__name__

    def initialize(self, model): pass # Hook before simulation starts
    def post_initialize(self, model): pass # Optional hook after all modes initialized and model setup done
    def step(self, model, i): pass # Hook called each time step
    def finalize(self, model): pass # Hook after simulation ends
    def check_compatibility(self, modes): pass # Check for incompatible modes

class SeasonalVariation(Mode):
    """
    Mode enabling seasonal insolation variation.

    Behaviour:
    ----------
    - Adds seasonal output classes.
    - Adjusts dt so that the year is divided into a multiple of 4 steps.
    - Limits run length and dt if defaults would not resolve the seasonal cycle.
    - Appends "_SeVa" to output directory name.
    - Replaces the insolation function Q_x with physics.seasonal_Q.
    """
    def __init__(self, app_mode=False):
        super().__init__(app_mode=app_mode)
        self.outputs.append(outputs.SeasonalOutput())
        if app_mode:
            self.outputs.append(outputs.SeasonalTempOnEarthOutput())

    def initialize(self, model):
        years = model.config["years"]
        dt_years = model.config["dt_years"]

        # Ensure temporal resolution is sufficient to resolve seasons
        if dt_years > 1/12:
            warn(f"Time step {dt_years} > 1/12 years is too large for seasonal cycles. "
                 f"Set to half-month resolution.")
            dt_years = 1/24

        if years > 1000:
            warn(f"Simulation time {years} > 1000 years is too long for seasonal runs. "
                 f"Set to 50 years.")
            years = 50

        # Apply corrected values
        model.config["years"] = max(1, int(years))  # Always at least one whole year
        model.config["dt_years"] = 1 / round(1 / dt_years / 4) / 4
        model.config["output_dir"] += "_SeVa"

        # Replace insolation kernel with seasonal version
        model.funcs['Q_x'] = phys.seasonal_Q

class VariableSeaDepth(Mode):
    """
    Mode that replaces the constant heat capacity with a spatially varying one.

    Behaviour:
    ----------
    - Appends "_SeaDep" to output directory name.
    - Removes SD from params (it is no longer used).
    - On each step, updates model.C using heat_capacity_profile(T, x, k1).
    """
    def __init__(self, app_mode=False):
        super().__init__(app_mode=app_mode)
        self.outputs.append(outputs.SeaDepthOutput())

    def initialize(self, model):
        model.config["output_dir"] += "_SeaDep"
        del model.params["SD"]

    def step(self, model, i):
        model.C = phys.heat_capacity_profile(model.x, model.T, model.params["k1"])

class VariableForcing(Mode):
    """
    Mode enabling time-varying external forcing (e.g., historical CO₂ forcing).

    Behaviour:
    ----------
    - Rejects coexistence with SeasonalVariation.
    - Replaces the physics.Forcing() function.
    - Tags output directory with "_VarForc".
    - Removes key F from params (indicates forcing is variable).
    - Loads forcing history from:
          - config["forcing_data"] (array-like), OR
          - CSV file (forcing_file)
    - Then in post_initialize, after model time steps are known:
        Interpolates forcing to match the number of model steps after the control period.
    - Stores full time series in model.F_History.
    """
    def check_compatibility(self, modes):
        if any(isinstance(m, SeasonalVariation) for m in modes):
            raise ValueError("VariableForcing mode is not compatible with SeasonalVariation mode.")

    def initialize(self, model):
        model.funcs["Forcing"] = phys.VariableForcing
        model.config["output_dir"] += "_VarForc"
        del model.params["F"]  # Marks that forcing is dynamic

        # --- Read forcing time series ---
        if model.config.get("forcing_data") is not None:
            # Provided directly in config
            self.ForcingHistory = np.array(model.config["forcing_data"])
        else:
            # Load from CSV file
            if "forcing_file" not in model.config:
                model.config["forcing_file"] = 'ForcingHistory.csv'

            print("Loading forcing data from file:", model.config["forcing_file"])
            path = os.path.join(os.path.dirname(__file__),
                                'Datafiler',
                                model.config["forcing_file"])
            with open(path) as f:
                reader = csv.reader(f)
                header = next(reader)  # optional header
                self.ForcingHistory = np.array([row for row in reader])

        # Extract years (assumed in first column)
        year = self.ForcingHistory[:, 0].astype(float)

        # Set model runtime to cover control period plus forcing history
        model.config["years"] = model.config["ctrl_years"] + len(year)

    def post_initialize(self, model):
        # Extract forcing values (assumed in last column)
        forcing = self.ForcingHistory[:, -1].astype(float)

        # Interpolate forcing to match simulation resolution after control period
        forcing_interp = np.interp(
            np.linspace(0, 1, model.nsteps - model.ctrl_nsteps),
            np.linspace(0, 1, len(forcing)),
            forcing
        )

        # Forcing is zero during control, then follows the history
        model.F_History = np.concatenate(
            (np.zeros(model.ctrl_nsteps), forcing_interp)
        )

def warn(msg):
    """
    Print a formatted warning message in bold yellow with a warning symbol.
    """
    YELLOW_BOLD = "\033[1;33m"
    RESET = "\033[0m"
    print(f"{YELLOW_BOLD}⚠️  Warning:{RESET} {msg}")
