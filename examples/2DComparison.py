from time import time

import numpy as np

from src.DQDSystemFactory import DQDSystemFactory, DQDAttributes
from src.PlotOptionsManager import PlotOptions
from src.base.auxiliaryMethods import formatComputationTime

"""
Script to plot the difference in current and polarity of two similiar parameters with respect to one fixed
"""

# Define the iteration parameters for X axis and Y axis. The X axis one will be the same for the two dots
# The range will be the same for all parameters (can be different in different axes)

valuesToPlot = np.linspace(-2.5, 2.5, 300)

# Changes in specific parameters

DQDSystemFactory.changeParameter(DQDAttributes.DETUNING.value, 0.8)  # Detuning setted to 0.8 times the ac frequency

# Title

DQDSystemFactory.addToTitle("Comparison between different magnetic field axes")  # We can add text instead of parameters
DQDSystemFactory.addToTitle(DQDAttributes.DETUNING.value)

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

# Create and simulate both systems (Here we select the kind of simulation we want)
dqdSystem = DQDSystemFactory.magneticFieldXvsMagneticFieldY(valuesToPlot, valuesToPlot)
dqdSystem.runSimulation()

otherSystem = DQDSystemFactory.magneticFieldXvsMagneticFieldZ(valuesToPlot, valuesToPlot)
otherSystem.runSimulation()

# Plot comparison
dqdSystem.compareSimulationsAndPlot(
    otherSystemDict=otherSystem,
    title=DQDSystemFactory.getTitleForSystem(),
    options=DQDSystemFactory.getPlotOptionsForSystem(),
    saveData=True,
    saveFigure=True
)

# Show elapsed time
print("Total time:", formatComputationTime(time() - timeStart))
