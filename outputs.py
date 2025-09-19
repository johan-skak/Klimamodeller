# outputs.py
import numpy as np
import matplotlib.pyplot as plt

class DefaultOutput:
    def __init__(self):
        self.history = []

    def after_step(self, model, t):
        Tmean = model.state["T"].mean() - 273.15
        self.history.append(Tmean)

    def finalize(self, model):
        plt.plot(self.history)
        plt.title("Global mean temperature (°C)")
        plt.show()

class SeasonalOutput:
    def __init__(self):
        self.history = []

    def after_step(self, model, t):
        lat = np.degrees(np.arcsin(model.state["x"]))
        T = model.state["T"] - 273.15
        self.history.append((t, T.copy()))

    def finalize(self, model):
        # Example: plot last temperature profile
        t, T = self.history[-1]
        lat = np.degrees(np.arcsin(model.state["x"]))
        plt.plot(lat, T)
        plt.title(f"Seasonal profile at step {t}")
        plt.xlabel("Latitude"); plt.ylabel("°C")
        plt.show()
