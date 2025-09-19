# model.py
import numpy as np
from .physics import Q_x, albedo_from_T, deltaT_of_Ts, build_diffusion_tridiag, apply_L_to_T, global_mean, thomas_solve
from .constants import SIGMA, C, SECONDS_PER_YEAR, T00

class ClimateModel:
    def __init__(self, config, modes, outputs):
        self.config = config
        self.modes = modes
        self.outputs = outputs
        self.state = {}

        # Default kernel functions
        self.Q_x = Q_x
        self.update_temperature = self._base_update_temperature

    def _base_update_temperature(self, T, params, dt):
        """Default update: radiative balance + diffusion (annual mean)."""
        # (This is adapted from your run_simulation in ebm.py)
        # Modes may overwrite this function
        raise NotImplementedError("Default kernel not yet implemented")

    def run(self):
        # Let modes modify config/state
        for m in self.modes: m.initialize(self)

        nsteps = int(round(self.config["years"] / self.config["dt_years"]))
        for t in range(nsteps):
            # Evolve model one step
            self.state = self.update_temperature(self.state, self.config, self.config["dt_years"] * SECONDS_PER_YEAR)

            # Modes hook
            for m in self.modes: m.step(self, t)

            # Outputs hook
            for o in self.outputs: o.after_step(self, t)

        # Finalize
        for m in self.modes: m.finalize(self)
        for o in self.outputs: o.finalize(self)
