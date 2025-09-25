# modes.py
import numpy as np
import outputs

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
        model.config["dt_years"] = 1 / round(1 / dt_years)
        model.config["output_dir"] += "_SeVa" #Modify output directory name

        # Replace insolation kernel
        # See Wikipedia Solar Irradiance
        def seasonal_Q(x, S, model, i):
            t = (i+1) * model.config["dt_years"]  # time in years
            eps = np.deg2rad(23.44) # obliquity
            theta = 2 * np.pi * t   # annual angle
            delta = np.arcsin(np.sin(eps) * np.sin(theta)) # Current declination δ = sin⁻¹(sin ε sin θ)

            # Hour angle at sunrise/sunset
            h0 = np.arccos(np.clip(- x / np.sqrt(1 - x**2 + 1e-10) * np.tan(delta), -1, 1))
            SIr = (S / np.pi) * (h0 * x * np.sin(delta) + np.sqrt(1 - x**2) * np.cos(delta) * np.sin(h0))
            return SIr

        model.funcs['Q_x'] = seasonal_Q # Replaces Q_x with seasonal_Q in ClimateModel object
    
    outputs = [outputs.SeasonalOutput(), outputs.TimeSeriesOutput()]

def warn(msg):
    """
    Print a formatted warning message in bold yellow with a ⚠️ symbol.
    Works in most Unix shells and modern PowerShell.
    """
    YELLOW_BOLD = "\033[1;33m"
    RESET = "\033[0m"
    print(f"{YELLOW_BOLD}⚠️  Warning:{RESET} {msg}")
