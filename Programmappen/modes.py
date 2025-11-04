# modes.py
import csv
import numpy as np
import outputs
import physics as phys
import os

def Ignore_modes(func):
    return lambda self, modes=None, app_mode=False: func(self, app_mode=app_mode)

class Mode:
    @Ignore_modes
    def __init__(self, app_mode=False):
        self.outputs = [] # List of output class instances
        self.app_mode = app_mode
    def __str__(self):
        return self.__class__.__name__
    def initialize(self, model): pass
    def step(self, model, i): pass
    def finalize(self, model): pass
    def check_compatibility(self, modes): pass

class SeasonalVariation(Mode):
    def __init__(self, modes, app_mode=False):
        super().__init__(app_mode=app_mode)
        self.outputs.extend([outputs.TimeSeriesOutput(), outputs.SeasonalOutput()])
        if self.app_mode: self.outputs.append(outputs.TemperatureOnEarthOutput(last_year_only=True))

    def initialize(self, model):
        years = model.config["years"]
        dt_years = model.config["dt_years"]
        # Override config if necessary
        if dt_years > 1/12: # is a problem with default settings
            warn(f"Time step - \033[4m{dt_years} > 1/12 years\033[0m - is to large to capture seasonal variation. Has been set to half a month.")
            dt_years = 1/24 # half monthly steps
        if years > 1000: # too long computation time # not relevant with default settings
            warn(f"Simulation time - \033[4m{years} > 1000 years\033[0m - is to large for reasonable run time. Has been set to fifty years.")
            years = 50
        model.config["years"] = int(years) if years >= 1 else 1 #Run a whole number of years; at least 1
        model.config["dt_years"] = 1 / round(1 / dt_years / 4) / 4 #Should be 1 / num where num is divisible by 4
        model.config["output_dir"] += "_SeVa" #Modify output directory name

        # Replace insolation kernel
        model.funcs['Q_x'] = phys.seasonal_Q # Replaces Q_x with seasonal_Q in ClimateModel object

class VariableSeaDepth(Mode):
    def __init__(self, modes, app_mode=False):
        super().__init__(app_mode=app_mode)
        if len(modes) == 1:
            self.outputs.extend([outputs.TimeSeriesOutput(), outputs.DefaultOutput()])
            if self.app_mode: self.outputs.append(outputs.TemperatureOnEarthOutput())
        self.outputs.append(outputs.SeaDepthOutput())

    def initialize(self, model):
        model.config["output_dir"] += "_SeaDep" #Modify output directory name
        del model.params["SD"] #Remove unused key from output

    def step(self, model, i):
        model.C = phys.heat_capacity_profile(model.x, model.T, model.params["k1"])

class VariableForcing(Mode):
    def __init__(self, modes, app_mode=False):
        super().__init__(app_mode=app_mode)
        #self.outputs.extend([outputs.TimeSeriesOutput(), outputs.DefaultOutput()])
        if self.app_mode: self.outputs.append(outputs.TemperatureOnEarthOutput())
        self.outputs.extend([outputs.VariableForcingOutput(), outputs.DefaultOutput()])
        
    def check_compatibility(self, modes):
        if any(isinstance(m, SeasonalVariation) for m in modes):
            raise ValueError("VariableForcing mode is not compatible with SeasonalVariation mode.")

    def initialize(self, model):
        model.funcs["Forcing"] = phys.VariableForcing
        model.config["output_dir"] += "_VarForc"
        del model.params["F"] #Remove unused key from output. This also (paradoxically) makes the outputs aware that forcing is on

       #Lav forceringshistorik her #open() returnerer nok en fejl hvis stien ikke findes og det er godt
        if model.config.get("forcing_data") is not None:
           ForcingHistory = np.array(model.config["forcing_data"])
        else:
            print("Loading forcing data from file:", model.config["forcing_file"])
            with open(os.path.join(os.path.dirname(__file__), 'Datafiler', model.config["forcing_file"])) as f:
                reader = csv.reader(f)
                header = next(reader)  # Skip header row if present
                ForcingHistory = np.array([row for row in reader]) # Reads CSV data
        year = ForcingHistory[:,0].astype(float)
        forcing = ForcingHistory[:,-1].astype(float)

        model.config["years"] = len(year) + model.config["ctrl_years"]
        model.nsteps = int(np.ceil(model.config["years"] / model.config["dt_years"])) # Run for at least config["years"]
        model.ctrl_nsteps = int(round(model.config["ctrl_years"] / model.config["dt_years"]))

        forcing = np.interp(np.linspace(0, 1, model.nsteps - model.ctrl_nsteps), np.linspace(0, 1, len(forcing)), forcing) # Interpolation
        model.F_History = np.concatenate( (np.zeros(model.ctrl_nsteps), forcing) ) #Start with 0's under the control period
        model.start_year = year[0]


def warn(msg):
    """
    Print a formatted warning message in bold yellow with a ⚠️ symbol.
    Works in most Unix shells and modern PowerShell.
    """
    YELLOW_BOLD = "\033[1;33m"
    RESET = "\033[0m"
    print(f"{YELLOW_BOLD}⚠️  Warning:{RESET} {msg}")