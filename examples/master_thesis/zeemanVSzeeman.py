from time import time, sleep

import numpy as np

from src.DQDSystemFactory import DQDSystemFactory, DQDAttributes
from src.PlotOptionsManager import PlotOptions
from src.base.auxiliaryMethods import formatComputationTime

# Define the iteration parameters for X axis and Y axis. The X axis one will be the same for the two dots
# The range will be the same for all parameters (can be different in different axes)

zeemanValues = np.linspace(-2.5, 2.5, 10)

# Changes in specific parameters

DQDSystemFactory.changeParameter(DQDAttributes.FACTOR_OME.value, 0.5)

# Title

DQDSystemFactory.addToTitle(DQDAttributes.DETUNING.value)

# Optional: configure plot options globally
DQDSystemFactory.addToPlotOptions(PlotOptions.GRID.value, True)
DQDSystemFactory.addToPlotOptions(PlotOptions.APPLY_TO_ALL.value, False)
DQDSystemFactory.addToPlotOptions(PlotOptions.COLOR_BAR_MIN.value, 0.0)
DQDSystemFactory.addToPlotOptions(PlotOptions.COLOR_BAR_MAX.value, 0.39)
DQDSystemFactory.addToPlotOptions(PlotOptions.PLOT_ONLY.value, None)
DQDSystemFactory.addToPlotOptions(PlotOptions.LOG_COLOR_BAR.value, False)
DQDSystemFactory.addToPlotOptions(PlotOptions.GAUSSIAN_FILTER.value, True)

# Start the timer
timeStart = time()

detuningsValues = [0.0, 0.15, 0.3, 0.5, 0.7, 0.85, 1.0, 1.15, 1.30]

for idx, detuning in enumerate(detuningsValues):
    DQDSystemFactory.changeParameter(DQDAttributes.DETUNING.value, detuning)

    # Create and simulate both systems (Here we select the kind of simulation we want)
    dqdSystem = DQDSystemFactory.zeemanYvsZeemanZ(zeemanValues, zeemanValues, loadData=False)

    DQDSystemFactory.addToPlotOptions(PlotOptions.PLOT_ONLY.value, 0)
    # Plot comparison
    dqdSystem.plotSimulation(
        title=DQDSystemFactory.getTitleForSystem(),
        options=DQDSystemFactory.getPlotOptionsForSystem(),
        saveData=True,
        saveFigure=True
    )

    sleep(1)
    # Create and simulate both systems (Here we select the kind of simulation we want)
    dqdSystem = DQDSystemFactory.zeemanYvsZeemanZ(zeemanValues, zeemanValues, loadData=True)

    DQDSystemFactory.addToPlotOptions(PlotOptions.PLOT_ONLY.value, 1)
    # Plot comparison
    dqdSystem.plotSimulation(
        title=DQDSystemFactory.getTitleForSystem(),
        options=DQDSystemFactory.getPlotOptionsForSystem(),
        saveData=True,
        saveFigure=True
    )

# Show elapsed time
print("Total time:", formatComputationTime(time() - timeStart))
