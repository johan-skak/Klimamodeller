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
        # Override config
        model.config["dt_years"] = 1/24    # monthly steps
        model.config["years"] = 50

        # Replace insolation kernel
        # See Wikipedia Solar Irradiance
        def seasonal_Q(x, S, model, i):
            t = i * model.config["dt_years"]  # time in years
            eps = np.deg2rad(23.44) # obliquity
            theta = 2 * np.pi * t   # annual angle
            delta = np.arcsin(np.sin(eps) * np.sin(theta)) # Current declination δ = sin⁻¹(sin ε sin θ)

            # Hour angle at sunrise/sunset
            h0 = np.arccos(np.clip(- x / np.sqrt(1 - x**2 + 1e-10) * np.tan(delta), -1, 1))
            SIr = (S / np.pi) * (h0 * x * np.sin(delta) + np.sqrt(1 - x**2) * np.cos(delta) * np.sin(h0))
            return SIr

        model.funcs['Q_x'] = seasonal_Q
    
    outputs = [outputs.SeasonalOutput(), outputs.TimeSeriesOutput()]
