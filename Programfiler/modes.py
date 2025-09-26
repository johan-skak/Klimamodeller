# modes.py
import numpy as np
import outputs
import physics as phys

class Mode:
    def __str__(self):
        return self.__class__.__name__
    def initialize(self, model): pass
    def step(self, model, i): pass
    def finalize(self, model): pass
    def check_compatibility(self, modes): pass
    outputs = [] # Is a list of output class instances

class SeasonalVariation(Mode):
    def initialize(self, model):
        years = model.config["years"]
        dt_years = model.config["dt_years"]
        # Override config if necessary
        if dt_years > 1/12: # is a problem with default settings
            warn(f"Time step - \033[4m{dt_years} > 1/12 years\033[0m - is to large to capture seasonal variation. Has been set to half a month.")
            model.config["dt_years"] = 1/24 # half monthly steps
        if years > 1000: # too long computation time # not relevant with default settings
            warn(f"Simulation time - \033[4m{years} > 1000 years\033[0m - is to large for reasonable run time. Has been set to fifty years.")
            model.config["years"] = 50
        model.config["years"] = int(years) if years >= 1 else 1 #Run a whole number of years; at least 1
        model.config["dt_years"] = 1 / round(1 / dt_years / 4) / 4 #Should be 1 / num where num is divisible by 4
        model.config["output_dir"] += "_SeVa" #Modify output directory name

        # Replace insolation kernel
        model.funcs['Q_x'] = phys.seasonal_Q # Replaces Q_x with seasonal_Q in ClimateModel object
    
    outputs = [outputs.SeasonalOutput(), outputs.TimeSeriesOutput()]

class VariableSeaDepth(Mode):
    def initialize(self, model):
        model.config["output_dir"] += "_SeaDep" #Modify output directory name
        del model.params["SD"] #Remove unused key from output

    def step(self, model, i):
        model.C = phys.heat_capacity_profile(model.config["nx"], model.T, model.params["k1"])

def warn(msg):
    """
    Print a formatted warning message in bold yellow with a ⚠️ symbol.
    Works in most Unix shells and modern PowerShell.
    """
    YELLOW_BOLD = "\033[1;33m"
    RESET = "\033[0m"
    print(f"{YELLOW_BOLD}⚠️  Warning:{RESET} {msg}")
