# main.py
import os, yaml, time
import model, modes, outputs

PARAMETERS_FILE = 'parameters.yaml'
CONFIG_FILE = 'config.yaml'

def main(config, params, app_mode=False):
    # Adjust config and params if necessary
    if config["ctrl_years"] is None or config["ctrl_years"] < 0: # Default to half simulation without forcing
        config["ctrl_years"] = config["years"]//2
    config["modes"].sort() # Sort modes alphabetically to have consistent naming of output directories
    if params["S1"] is None: # Default to no change in solar forcing
        params["S1"] = params["S0"]

    # Create mode instances
    modes_list = [] # Is a list of mode class instances
    for mode_name in config["modes"]:
        if hasattr(modes, mode_name):
            modes_list.append(getattr(modes, mode_name)(config["modes"], app_mode)) # Some modes needs to know what other modes there are to choose correct outputs
        else:
            raise ValueError(f"Unknown mode: {mode_name}")

    # Gather outputs from modes
    outputs_list = [o for m in modes_list for o in m.outputs]
    if not outputs_list:
        outputs_list = [outputs.TimeSeriesOutput(), outputs.DefaultOutput()] # Default outputs with forcing line
        if app_mode: outputs_list.append(outputs.TemperatureOnEarthOutput()) # Add Earth surface output in app mode
    
    # Create and run model
    climate_model = model.ClimateModel(config, params, modes_list, outputs_list, app_mode)
    start_time = time.perf_counter()
    climate_model.run()
    end_time = time.perf_counter()

    # Make outputs
    out = outputs.run_all_outputs(outputs_list, climate_model.config["output_dir"], climate_model.sim_info, end_time - start_time, app_mode) # Climate_model.config may be different from input config due to modes
    if app_mode: return out # Only relevant for Streamlit app

def configure_program():
    # Default config
    config = {"years": 1000, "ctrl_years": -1, "dt_years": 1, "nx": 200, "modes": [],
              "output_dir": "Results", "forcing_file": 'ForcingHistory.csv', "zonal_temp_file": "current_zonal_mean_temperature.csv","temperature_history": "temperature_history.nc"}

    # Default parameters
    params = dict(k1=0.06, k2=0.01, k3=0.5, D0=0.66, T0=288.0, SD=250,
                S0=1365.0, S1=None, F=0.0)
    
    # Read config from file if it exists and update defaults
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE) as f:
            f_dict = yaml.safe_load(f)

        for key, value in f_dict.items():
            if key not in config:
                modes.warn(f"Unknown config key: {key}. This key will be ignored.")
            if value is not None:
                config[key] = value
    else:
        modes.warn(f"No config file found ({CONFIG_FILE}) in current directory. Using default configs.")

    # Read parameters from file if it exists and update defaults
    if os.path.exists(PARAMETERS_FILE):
        with open(PARAMETERS_FILE) as f:
            f_dict = yaml.safe_load(f)
        
        for key, value in f_dict.items():
            if key not in params:
                modes.warn(f"Unknown parameter: {key}. This parameter will be ignored.")
            if value is not None:
                params[key] = value
    else:
        modes.warn(f"No parameters file found ({PARAMETERS_FILE}) in current directory. Using default parameters.")

    return config, params

if __name__== "__main__":
    config, params = configure_program() # Read config and params from files or use defaults
    main(config, params) # Run model with config and params from files