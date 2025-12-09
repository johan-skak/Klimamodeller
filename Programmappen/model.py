# model.py
import numpy as np
import physics as phys
import outputs

class ClimateModel:
    """
    Time-dependent 1D meridional energy balance climate model.

    This class holds all model state (temperature field, grid, time step,
    parameters, active modes, and output handlers) and contains the
    central time-stepping loop.

    Parameters
    ----------
    config : dict
        Simulation settings:
        - "years": total simulated years
        - "ctrl_years": length of control run
        - "dt_years": time step in years
        - "nx": number of grid points in x = sin(latitude)
        - "output_dir": destination for files
        - "modes": list of mode names (strings)

    params : dict
        Physical parameters:
        - k1, k2, k3 : feedback strengths
        - D0        : base diffusion coefficient
        - T0        : initial global-mean surface temperature
        - SD        : mixed-layer depth (affects heat capacity)
        - S0, S1    : solar constants (before/after control run)
        - F         : external radiative forcing

    modes : list
        List of mode instances. Each mode may modify configuration,
        parameters, physics functions or add diagnostics.

    outputs : list
        Output handler instances (e.g. for plots, files, summaries).

    app_mode : bool
        Used by the Streamlit interface. If True, some printing and
        output behaviour is adapted.
    """
    def __init__(self, config, params, modes, outputs, app_mode=False):
        self.config = config
        self.params = params
        self.modes = modes
        self.outputs = outputs
        self.app_mode = app_mode

        # Collect all callables (functions) from the physics module.
        # This allows modes to override individual physics functions.
        self.funcs = {
            name: func for name, func in vars(phys).items() if callable(func)
        }

        # Effective heat capacity (J/K) depending on chosen water depth
        self.C = phys.C_M * params['SD']

        # Modes may declare that they cannot run together
        for m in self.modes:
            m.check_compatibility(self.modes)

    def run(self):
        """
        Execute the full simulation.

        Workflow:
            1. Modes perform setup via `initialize()`
            2. Grid and initial temperature are created
            3. Outputs initialize
            4. Main time loop
                - Temperature update
                - Mode hooks
                - Output hooks
            5. Modes and outputs finalize
        """
        # Let modes do pre-processing (modify config/params/functions/etc.)
        for m in self.modes:
            m.initialize(self)

        # Print info summary (also used by Streamlit but with no printing)
        self.sim_info = outputs.print_simulation_info(
            self.config, self.params, self.app_mode
        )

        # --- Grid and initial conditions ---
        self.dx = 2.0 / self.config["nx"]
        self.x = np.linspace(
            -1.0 + self.dx / 2, 1.0 - self.dx / 2, self.config["nx"]
        )

        # Initial zonal-mean temperature profile
        self.T = phys.T_init(self.x, self.params["T0"])

        # Convert time step to seconds
        self.dt = self.config["dt_years"] * phys.SECONDS_PER_YEAR

        # Number of total steps and control-run steps
        self.nsteps = int(np.ceil(self.config["years"] / self.config["dt_years"]))
        self.ctrl_nsteps = int(
            round(self.config["ctrl_years"] / self.config["dt_years"])
        )

        # Let modes do post-initialization (after model setup)
        for m in self.modes:
            m.post_initialize(self)

        # Allow outputs to record initial state
        for o in self.outputs:
            o.initialize(self)

        # --- Main time loop ---
        for i in range(self.nsteps):
            # Evolve temperature by one time step
            self.T = self.update_temperature(
                self.T, self.x, self.params, self.funcs, i
            )

            # Mode hooks (e.g., forcing switches, heat capacity changes)
            for m in self.modes:
                m.step(self, i)

            # Output hooks (e.g., saving snapshots)
            for o in self.outputs:
                o.step(self, i)

        # --- Finalization ---
        for m in self.modes:
            m.finalize(self)
        for o in self.outputs:
            o.finalize(self)

    def update_temperature(self, T, x, params, funcs, i):
        """
        Advance temperature by one time step using:

            - Radiative balance
            - Meridional diffusion
            - Crank-Nicolson implicit scheme for diffusion

        Parameters
        ----------
        T : array
            Current temperature field.
        x : array
            Grid points (sin(latitude)).
        params : dict
            Physical parameters.
        funcs : dict
            Physics functions (may be modified by modes).
        i : int
            Current time-step index.

        Returns
        -------
        T_new : array
            Updated temperature field.
        """

        # Choose solar constant before/after control run
        S = params["S0"] if i < self.ctrl_nsteps else params["S1"]

        # Shortwave: Q(x), albedo(T), absorbed flux
        Q_x = funcs["Q_x"](x, S, model=self, i=i)
        alpha = funcs["albedo_from_T"](T, x, params["k1"], model=self, i=i)
        absorbed = Q_x * (1.0 - alpha)

        # Longwave: emission temperature offset, outgoing longwave radiation
        dTloc = funcs["deltaT_of_Ts"](T, params["k3"], model=self, i=i)
        olr = phys.SIGMA * (T - dTloc)**4

        # Net radiative term, including external forcing F
        rad_term = absorbed - olr + funcs["Forcing"](model=self, i=i)

        # Meridional diffusion coefficient depends on mean temperature
        D = funcs["diffusion_from_T"](T, params["D0"], params["k2"],
                                      model=self, i=i)

        # Build diffusion operator L (tridiagonal representation)
        aL, bL, cL = funcs["build_diffusion_tridiag"](x, D)
        LT = funcs["apply_L_to_T"](aL, bL, cL, T)

        # Crank-Nicolson time discretization
        coef = self.dt / self.C
        rhs = T + 0.5 * coef * LT + coef * rad_term

        # Left-hand-side tridiagonal matrix for implicit diffusion
        aA = -0.5 * coef * aL
        bA = 1.0 - 0.5 * coef * bL
        cA = -0.5 * coef * cL

        # Solve the tridiagonal system
        return funcs["thomas_solve"](aA, bA, cA, rhs)
