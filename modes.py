# modes.py
import numpy as np
import math

class Mode:
    def initialize(self, model): pass
    def step(self, model, t): pass
    def finalize(self, model): pass
    def check_compatibility(self, other): return True

class DefaultMode(Mode):
    """Always-on physics (annual mean)."""
    def initialize(self, model):
        # Setup grid, initial temperature, etc.
        nx = model.config["nx"]
        dx = 2.0 / nx
        x = np.linspace(-1.0 + dx/2, 1.0 - dx/2, nx) # sin(lat)
        T0 = model.config["T0"]
        A_PROFILE = 45.0
        T = T0 + A_PROFILE * (1/3 - x**2)

        model.state.update({"x": x, "T": T})

class SeasonalVariation(Mode):
    def initialize(self, model):
        # Override config
        model.config["dt_years"] = 1/24    # monthly steps
        model.config["years"] = 50

        # Replace insolation kernel
        def seasonal_Q(x, S, t):
            # δ = sin⁻¹(sin ε sin θ)
            eps = math.radians(23.44)
            theta = 2*math.pi * (t/12)   # annual angle
            delta = math.asin(math.sin(eps) * math.sin(theta))

            # hour angle at sunset
            h0 = np.arccos(-np.tan(np.arcsin(x)) * np.tan(delta))
            h0 = np.clip(h0, -math.pi, math.pi)

            # Wikipedia formula
            return (S / math.pi) * (h0 * np.sin(np.arcsin(x)) * np.sin(delta) +
                                    np.cos(np.arcsin(x)) * np.cos(delta) * np.sin(h0))

        model.Q_x = seasonal_Q
