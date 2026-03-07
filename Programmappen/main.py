#main.py
import os, yaml, time
import model, modes, outputs, tools

PARAMETERS_FILE = 'parameters.yaml'
CONFIG_FILE = 'config.yaml'

def main(config, params, app_mode=False):
    """
    Run the climate model with a given configuration and parameter set.

    Parameters
    ----------
    config : dict
        Simulation settings. Expected keys:
        {
            "years": int           # Total simulation length in years
            "ctrl_years": int      # Length of control run; if <0, defaults to half of total
            "dt_years": float      # Time step in years
            "nx": int              # Number of spatial grid points (latitude)
            "modes": list[str]     # Names of active modes (strings matching classes in modes.py)
            "output_dir": str      # Directory where results are saved
        }

    params : dict
        Physical model parameters:
        {
            "k1": float   # Snow/ice-albedo sensitivity
            "k2": float   # Meridional transport sensitivity
            "k3": float   # Longwave feedback sensitivity
            "D0": float   # Base meridional diffusion coefficient
            "T0": float   # Initial global mean temperature (K)
            "SD": float   # Water column heat capacity depth (m)
            "S0": float   # Present-day solar constant
            "S1": float   # Modified solar constant (None = use S0)
            "F": float    # External radiative forcing
        }

    app_mode : bool, optional
        Used by the Streamlit interface. If True, output objects are returned
        instead of writing files.

    Notes
    -----
    - For each name in config["modes"], a corresponding class is retrieved
      from the `modes` module using getattr(). This is simply a way of
      constructing mode objects based on strings in the config file.
    - If config["ctrl_years"] is missing or negative, it is replaced by
      half of the total simulation time.
    """
    # Sort mode names for stable output folder naming
    config["modes"].sort()

    # Default control run length
    if config["ctrl_years"] is None or config["ctrl_years"] < 0:
        config["ctrl_years"] = config["years"] // 2

    # Default to no solar-forcing change
    if params["S1"] is None:
        params["S1"] = params["S0"]

    # Instantiate modes
    modes_list = []
    for mode_name in config["modes"]:
        if hasattr(modes, mode_name):
            # Each mode receives the full mode list (needed for mode-dependent outputs)
            modes_list.append(getattr(modes, mode_name)(app_mode))
        else:
            raise ValueError(f"Unknown mode: {mode_name}")

    # Collect all output objects defined by the active modes
    outputs_list = outputs.collect_outputs(modes_list, app_mode)

    # Create and run the model
    climate_model = model.ClimateModel(config, params, modes_list, outputs_list, app_mode)

    start_time = time.perf_counter()
    climate_model.run()
    end_time = time.perf_counter()

    # Process outputs (plots, files, tables, etc.)
    out = outputs.run_all_outputs(
        outputs_list,
        climate_model.config["output_dir"],   # May have been adjusted by modes
        climate_model.sim_info,
        end_time - start_time,
        app_mode
    )

    if app_mode:
        return out

def configure_program():
    """
    Build configuration and parameter dictionaries, applying overrides
    from config.yaml and parameters.yaml when present.

    Returns
    -------
    (config, params) : tuple of dict

    config dict has the structure:
    {
        "years": 1000,
        "ctrl_years": -1,
        "dt_years": 1,
        "nx": 200,
        "modes": [],
        "output_dir": "Results"
    }

    params dict has the default structure:
    {
        "k1": 0.06,
        "k2": 0.01,
        "k3": 0.5,
        "D0": 0.66,
        "T0": 288.0,
        "SD": 250,
        "S0": 1365.0,
        "S1": None,
        "F": 0.0
    }

    Notes
    -----
    - YAML files may contain null values; these leave defaults unchanged.
    - Unknown keys cause warnings but are otherwise ignored.
    """
    # Default model configuration
    config = {
        "years": 1000,
        "ctrl_years": -1,
        "dt_years": 1,
        "nx": 200,
        "modes": [],
        "output_dir": "Results",
    }

    # Default physical parameters
    params = dict(
        k1=0.06, k2=0.01, k3=0.5,
        D0=0.66, T0=288.0, SD=250,
        S0=1365.0, S1=None,
        F=0.0
    )

    # Load config overrides from YAML
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE) as f:
            file_cfg = yaml.safe_load(f) or {}

        for key, value in file_cfg.items():
            if key not in config:
                tools.warn(f"Unknown config key: {key}. This key will be ignored.")
            if value is not None:
                config[key] = value
    else:
        tools.warn(f"No config file found ({CONFIG_FILE}) in current directory. Using default configs.")

    # Load parameter overrides
    if os.path.exists(PARAMETERS_FILE):
        with open(PARAMETERS_FILE) as f:
            file_params = yaml.safe_load(f) or {}

        for key, value in file_params.items():
            if key not in params:
                tools.warn(f"Unknown parameter: {key}. This parameter will be ignored.")
            if value is not None:
                params[key] = value
    else:
        tools.warn(f"No parameters file found ({PARAMETERS_FILE}) in current directory. Using default parameters.")

    return config, params

if __name__ == "__main__": # Run first when executed as a script
    config, params = configure_program()
    main(config, params)
