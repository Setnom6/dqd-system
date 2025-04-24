from time import time

import numpy as np

from src.DQDSystemFactory import DQDSystemFactory, DQDAttributes
from src.base.auxiliaryMethods import formatComputationTime

# Start the timer
timeStart = time()

# Set fixed simulation parameters
DQDSystemFactory.changeParameter(DQDAttributes.DETUNING.value, 0.8)

# Define the array to sweep ZeemanZ
arrayValues = np.linspace(-3, 3, 300)

# Optional: configure plot options globally
DQDSystemFactory.addToPlotOptions("grid", True)
DQDSystemFactory.addToPlotOptions("applyToAll", False)
DQDSystemFactory.addToPlotOptions("colorBarMin", 0.02)
DQDSystemFactory.addToPlotOptions("colorBarMax", 0.35)
DQDSystemFactory.addToPlotOptions("plotOnly", None)
DQDSystemFactory.addToPlotOptions("logColorBar", False)
DQDSystemFactory.addToPlotOptions("gaussianFilter", False)

# Optional: configure dynamic title fields
DQDSystemFactory.addToTitle(DQDAttributes.DETUNING.value)

# Create and run the system
dqdSystem = DQDSystemFactory.zeemanZ(arrayValues)
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
