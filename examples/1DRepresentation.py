from time import time

import numpy as np

from src.DQDSystemFactory import DQDSystemFactory, DQDAttributes
from src.PlotOptionsManager import PlotOptions
from src.base.auxiliaryMethods import formatComputationTime

"""
Script to plot the variation of the current and the polarity through the variation of one parameter
"""

# Define the iteration parameter

valuesToPlot = np.linspace(-3, 3, 300)

# Changes in specific fixed parameters

DQDSystemFactory.changeParameter(DQDAttributes.ZEEMAN.value + "Left",
                                 [0.0, 0.0,
                                  1.2])  # Zeeman vector of both dots set to [1.2, 0.0,0.0] times the ac frequency
DQDSystemFactory.changeParameter(DQDAttributes.ZEEMAN.value + "Right",
                                 [0.0, 0.0,
                                  1.2])

# Quantities to set in the title

DQDSystemFactory.addToTitle(DQDAttributes.ZEEMAN.value + "ZLeft")

# Optional: configure plot options globally
DQDSystemFactory.addToPlotOptions(PlotOptions.GRID.value, True)
DQDSystemFactory.addToPlotOptions(PlotOptions.APPLY_TO_ALL.value, False)
DQDSystemFactory.addToPlotOptions(PlotOptions.COLOR_BAR_MIN.value, 0.02)
DQDSystemFactory.addToPlotOptions(PlotOptions.COLOR_BAR_MAX.value, 0.35)
DQDSystemFactory.addToPlotOptions(PlotOptions.PLOT_ONLY.value, None)
DQDSystemFactory.addToPlotOptions(PlotOptions.LOG_COLOR_BAR.value, False)
DQDSystemFactory.addToPlotOptions(PlotOptions.GAUSSIAN_FILTER.value, False)

# Start the timer
timeStart = time()

# Create and run the system (Here we select the kind of simulation we want)
dqdSystem = DQDSystemFactory.detuning(valuesToPlot)
dqdSystem.runSimulation()

# Plot using factory-managed options and title
dqdSystem.plotSimulation(
    title=DQDSystemFactory.getTitleForSystem(),
    options=DQDSystemFactory.getPlotOptionsForSystem(),
    saveData=True,
    saveFigure=True
)

# Show elapsed time
print("Total time:", formatComputationTime(time() - timeStart))
