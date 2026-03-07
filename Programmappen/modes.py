# modes.py
import csv
import numpy as np
import outputs
import physics as phys
import os
import tools

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
    - Adjusts dt so that the years is divided into a multiple of 4 steps.
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
            tools.warn(f"Time step {dt_years} > 1/12 years is too large for seasonal cycles. "
                       f"Set to half-month resolution.")
            dt_years = 1/24 # half monthly steps

        if years > 1000: # too long computation time # not relevant with default settings
            tools.warn(f"Simulation time {years} > 1000 years is too long for seasonal runs. "
                       f"Set to 50 years.")
            years = 50

        # Apply corrected values
        model.config["years"] = max(1, int(years)) #Run a whole number of years; at least 1
        model.config["dt_years"] = 1 / round(1 / dt_years / 4) / 4 #Should be 1 / number_of_steps where number_of_steps is divisible by 4
        model.config["output_dir"] += "_SeVa" #Modify output directory name

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
        model.config["output_dir"] += "_SeaDep" #Modify output directory name
        if "SD" in model.params:
            del model.params["SD"] #Remove unused key from output
    
    def post_initialize(self, model):
        model.C = phys.heat_capacity_profile(model.x, model.T, model.params["k1"]) #Initialize heat capacity profile

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
    def __init__(self, app_mode=False):
        super().__init__(app_mode=app_mode)
        self.outputs.extend([outputs.VariableForcingOutput()]) #plot the variable forcing data on the time series plot of temperature

    # The VariableForcing mode is not compatible with SeasonalVariation mode. Because:
    # The object of interest in SeasonalVariation is seasonal variation relative to an annual mean in equilibrium
    # While in VariableForcing the annual mean temperature is not in equilibrium by definition  
    # Also the values of the heat capacity parameter needed to get good results from the model are too different in the two modes.
    def check_compatibility(self, modes):
        if any(isinstance(m, SeasonalVariation) for m in modes): #check if SeasonalVariation mode is active
            raise ValueError("VariableForcing mode is not compatible with SeasonalVariation mode.")

    def initialize(self, model):
        model.funcs["Forcing"] = phys.VariableForcing #enable the model to vary the forcing for each timestep
        model.config["output_dir"] += "_VarForc"  #name the output directory accordingly
        if "F" in model.params:
            del model.params["F"] #Remove unused key from output. This also makes the outputs aware that forcing is on

        # --- Read forcing time series ---
        if model.config.get("forcing_data") is not None:
            # Provided directly in config
            self.years, self.ForcingHistory = np.array(model.config["forcing_data"]).astype(float).T
        else:
            # Load from CSV file
            if "forcing_file" not in model.config:
                model.config["forcing_file"] = 'ForcingHistory.csv'

            print("Loading forcing data from file:", model.config["forcing_file"])
            self.years, self.ForcingHistory = tools.csv_reader(model.config["forcing_file"]) #load data from file

        # Set model runtime to cover control period plus forcing history
        model.config["years"] = model.config["ctrl_years"] + len(self.years)

    def post_initialize(self, model):
        # Interpolate forcing to match simulation resolution after control period
        forcing_interp = np.interp(
            np.linspace(0, 1, model.nsteps - model.ctrl_nsteps),
            np.linspace(0, 1, len(self.ForcingHistory)),
            self.ForcingHistory
        )

        # Forcing is zero during control, then follows the history
        model.F_History = np.concatenate(
            (np.zeros(model.ctrl_nsteps), forcing_interp)
        )
        model.start_year = self.years[0] #define forcing period start year parameter to be used in outputs

class HistoricalData(Mode):
    """
    Mode to run the model with historical observational data as a reference.
    """
    def __init__(self, app_mode=False):
        super().__init__(app_mode=app_mode)
        self.outputs.extend([outputs.HistoricalOutput()]) #plot historical temperature data on default plots

    # The HistoricalData mode is not compatible with SeasonalVariation mode. Because:
    # The datasets used in HistoricalData do not account for seasonal variation, so comparing seasonal model outputs to annual mean observational data would be misleading.
    def check_compatibility(self, modes):
        if any(isinstance(m, SeasonalVariation) for m in modes):
            raise ValueError("HistoricalData mode is not compatible with SeasonalVariation mode.")

    def initialize(self, model):
        model.config["output_dir"] += "_HistData" # name output directory according to mode
        model.zonal_temp_file = "current_zonal_mean_temperature.csv" # change default zonal temperature file to historical data file
        model.temperature_history_file = "temperature_history.nc" # change default temperature history file to historical data file
        model.zonal_olr_file = "zonal_mean_longwave_out.csv" # change default zonal olr file to historical data file
        model.albedo_file = "zonal_mean_shortwave_out.csv" # change default albedo file to historical data file
        model.solar_file = "zonal_mean_solar_in.csv" # change default solar