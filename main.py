# main.py
import os, json
import model, modes, outputs

PARAMETERS_FILE = 'parameters.json'
CONFIG_FILE = 'config.json'

if __name__ == "__main__":
    # Default config
    config = {"years": 1000, "ctrl_years": -1, "dt_years": 1, "nx": 200, "modes": [], "output_dir": "Results"}
    # Read config from file if it exists and update defaults
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE) as f:
            config |= json.load(f)

    # Default parameters
    params = dict(k1=0.06, k2=0.01, k3=0.5, D0=0.66, T0=288.0,
                S=1365.0, F=0.0)
    # Read parameters from file if it exists and update defaults
    if os.path.exists(PARAMETERS_FILE):
        with open(PARAMETERS_FILE) as f:
            params |= json.load(f)

    modes_list = [] # Is a list of mode class instances
    for mode_name in config["modes"]:
        if hasattr(modes, mode_name):
            modes_list.append(getattr(modes, mode_name)())
        else:
            raise ValueError(f"Unknown mode: {mode_name}")

    # Gather outputs from modes
    outputs_list = [o for m in modes_list for o in m.outputs]
    if not outputs_list:
        outputs_list = [outputs.DefaultOutput(), outputs.TimeSeriesOutput(True)] # Default outputs with forcing line
    
    # Create and run model
    climate_model = model.ClimateModel(config, params, modes_list, outputs_list)
    climate_model.run()

    # Make outputs
    outputs.run_all_outputs(outputs_list, climate_model.config["output_dir"]) # Climate_model.config may be different from input config due to modes