# main.py
from model import ClimateModel
from modes import DefaultMode, SeasonalVariation
from outputs import DefaultOutput, SeasonalOutput

if __name__ == "__main__":
    config = {"nx": 120, "dt_years": 1.0, "years": 100, "T0": 288.0}

    modes = [DefaultMode(), SeasonalVariation()]
    outputs = [SeasonalOutput()]

    model = ClimateModel(config, modes, outputs)
    model.run()
