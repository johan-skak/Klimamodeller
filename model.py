# model.py
import numpy as np
import physics as phys
import outputs

class ClimateModel:
    def __init__(self, config, params, modes, outputs):
        self.config = config
        self.params = params
        self.modes = modes
        self.outputs = outputs
        self.funcs = {name: func for name, func in vars(phys).items() if callable(func)} # Physics functions
        for m in self.modes: m.check_compatibility(self.modes)

    def run(self):
        # Let modes modify config/params/T/funcs as needed
        for m in self.modes: m.initialize(self)
        outputs.print_simulation_info(self.config, self.params)

        # Define grid and initial state
        self.dx = 2.0 / self.config["nx"]
        self.x = np.linspace(-1.0 + self.dx/2, 1.0 - self.dx/2, self.config["nx"])
        self.T = phys.T_init(self.x, self.params["T0"])  # Initial temperature profile (K)

        self.dt = self.config["dt_years"] * phys.SECONDS_PER_YEAR # time step in seconds
        self.nsteps = int(np.ceil(self.config["years"] / self.config["dt_years"])) # Run for at least config["years"]
        self.ctrl_nsteps = int(round(self.config["ctrl_years"] / self.config["dt_years"]))

        # Let outputs collect initial data
        for o in self.outputs: o.initialize(self)
        
        for i in range(self.nsteps):
            # Evolve model one step
            self.T = self.update_temperature(self.T, self.x, self.dt, self.params, self.funcs, i)

            # Modes hook
            for m in self.modes: m.step(self, i)

            # Outputs hook
            for o in self.outputs: o.step(self, i)

        # Finalize
        for m in self.modes: m.finalize(self)
        for o in self.outputs: o.finalize(self)

    def update_temperature(self, T, x, dt, params, funcs, i):
        # Explicit radiative terms
        Q_x = funcs['Q_x'](x, params['S'], model=self, i=i) # The model and i arguments are ignored in default mode but necessary for other modes
        alpha = funcs['albedo_from_T'](T, x, params['k1'], model=self, i=i)
        absorbed = Q_x * (1.0 - alpha)
        dTloc = funcs['deltaT_of_Ts'](T, params['k3'], model=self, i=i)
        olr = phys.SIGMA * (T - dTloc)**4
        rad_term = absorbed - olr + params['F'] * (i >= self.ctrl_nsteps) # Only apply forcing after control period

        # Diffusivity depends on global mean temperature
        D = params['D0'] * max(0.5, 1.0 + params['k2'] * (T.mean() - phys.T00))

        # Build L and do Crank–Nicolson step
        aL, bL, cL = funcs['build_diffusion_tridiag'](x, D)
        LT = funcs['apply_L_to_T'](aL, bL, cL, T)
        coef = dt / phys.C
        rhs = T + 0.5 * coef * LT + coef * rad_term
        aA = -0.5 * coef * aL
        bA =  1.0 - 0.5 * coef * bL
        cA = -0.5 * coef * cL
        return funcs['thomas_solve'](aA, bA, cA, rhs)