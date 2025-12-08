# modes.py
import csv
import numpy as np
import outputs
import physics as phys
import os
import tools

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
        self.outputs.append(outputs.SeasonalOutput())
        if app_mode:
            self.outputs.append(outputs.SeasonalTempOnEarthOutput())

    def initialize(self, model):
        years = model.config["years"]
        dt_years = model.config["dt_years"]
        # Override config if necessary
        if dt_years > 1/12: # is a problem with default settings
            tools.warn(f"Time step - \033[4m{dt_years} > 1/12 years\033[0m - is to large to capture seasonal variation. Has been set to half a month.")
            dt_years = 1/24 # half monthly steps
        if years > 1000: # too long computation time # not relevant with default settings
            tools.warn(f"Simulation time - \033[4m{years} > 1000 years\033[0m - is to large for reasonable run time. Has been set to fifty years.")
            years = 50
        model.config["years"] = int(years) if years >= 1 else 1 #Run a whole number of years; at least 1
        model.config["dt_years"] = 1 / round(1 / dt_years / 4) / 4 #Should be 1 / num where num is divisible by 4
        model.config["output_dir"] += "_SeVa" #Modify output directory name

        # Replace insolation kernel
        model.funcs['Q_x'] = phys.seasonal_Q # Replaces Q_x with seasonal_Q in ClimateModel object

class VariableSeaDepth(Mode): 
    def __init__(self, modes, app_mode=False):
        super().__init__(app_mode=app_mode)
        self.outputs.append(outputs.SeaDepthOutput())

    def initialize(self, model):
        model.config["output_dir"] += "_SeaDep" #Modify output directory name
        del model.params["SD"] #Remove unused key from output

    def step(self, model, i):
        model.C = phys.heat_capacity_profile(model.x, model.T, model.params["k1"])

"VariableForcing mode enables time-varying radiative forcing based on external data."
class VariableForcing(Mode):
    def __init__(self, modes, app_mode=False):
        super().__init__(app_mode=app_mode)
        self.outputs.extend([outputs.VariableForcingOutput()]) #plot the variable forcing data on the time series plot of temperature

    #The VariableForcing mode is not compatible with SeasonalVariation mode. Because:
    # The object of interest in SeasonalVariation is seasonal variation relative to an annual mean in equilibrium
    # While in VariableForcing the annual mean temperature is not in equilibrium by definition  
    # Also the values of the heat capacity parameter needed to get good results from the model are too different in the two modes.
    def check_compatibility(self, modes):
        if any(isinstance(m, SeasonalVariation) for m in modes): #check if SeasonalVariation mode is active
            raise ValueError("VariableForcing mode is not compatible with SeasonalVariation mode.")

    def initialize(self, model):
        model.funcs["Forcing"] = phys.VariableForcing #enable the model to vary the forcing for each timestep
        model.config["output_dir"] += "_VarForc"  #name the output directory accordingly
        del model.params["F"] #Remove unused key from output. This also makes the outputs aware that forcing is on

        # de næste 2 linjer bruges kun i app mode 
        if model.config.get("forcing_data") is not None:
           ForcingHistory = np.array(model.config["forcing_data"])
        #følgende 4 linjer køres altid hvis ikke i app mode
        else:
            if "forcing_file" not in model.config: model.config["forcing_file"] = 'ForcingHistory.csv' #set expected data location to a default filename
            print("Loading forcing data from file:", model.config["forcing_file"]) #print that file is being loaded
            year,forcing = tools.csv_reader(model.config["forcing_file"]) #load data from file

        model.config["years"] = len(year) + model.config["ctrl_years"] #let forcing file overwrite non-control simulation years
        model.nsteps = int(np.ceil(model.config["years"] / model.config["dt_years"])) # define new number of time steps
        model.ctrl_nsteps = int(round(model.config["ctrl_years"] / model.config["dt_years"])) # define new number of control time steps

        forcing = np.interp(np.linspace(0, 1, model.nsteps - model.ctrl_nsteps), np.linspace(0, 1, len(forcing)), forcing) # Interpolation from data time points to model time points
        model.F_History = np.concatenate( (np.zeros(model.ctrl_nsteps), forcing) ) #set Forcing to 0's under the control period
        model.start_year = year[0] #define forcing period start year parameter to be used in outputs

"HistoricalData mode adds outputs comparing model results to historical observational data."
class HistoricalData(Mode):
    def __init__(self, modes, app_mode=False):
        super().__init__(modes, app_mode=app_mode)
        self.outputs.extend([outputs.ObservedOutput(),outputs.HistoricalOutput()]) #plot observed temperature data on a time series and observed data on default plots

# The HistoricalData mode is not compatible with SeasonalVariation mode. Because:
# The datasets used in HistoricalData do not account for seasonal variation, so comparing seasonal model outputs to annual mean observational data would be misleading.
    def check_compatibility(self, modes):
        if any(isinstance(m, SeasonalVariation) for m in modes):
            raise ValueError("HistoricalData mode is not compatible with SeasonalVariation mode.")

    def initialize(self, model):
        model.config["output_dir"] += "_HistData" # name output directory according to mode