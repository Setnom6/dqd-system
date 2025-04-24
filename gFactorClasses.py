from time import time

import numpy as np

from src.DQDSystemFactory import DQDSystemFactory, DQDAttributes
from src.base.auxiliaryMethods import formatComputationTime

# Start timer
timeStart = time()

# Define iteration array
xArray = np.linspace(-2.5, 2.5, 10)

# Set fixed parameters
DQDSystemFactory.changeParameter(DQDAttributes.DETUNING.value, 0.0)
DQDSystemFactory.changeParameter(DQDAttributes.FACTOR_OME.value, 0.0)
DQDSystemFactory.changeParameter(DQDAttributes.MAGNETIC_FIELD.value, [1.0, 1.0, 0.0])
DQDSystemFactory.changeParameter(DQDAttributes.G_FACTOR.value + "Left", [[6, 0, 0], [0, 4, 0], [0, 0, 5]])
DQDSystemFactory.changeParameter(DQDAttributes.G_FACTOR.value + "Right", [[1.02723, 0, 1.8791],
                                                                          [0, 5, 0],
                                                                          [-2.81865, 0, 0.684822]])

# Optional: modify plot options
DQDSystemFactory.addToPlotOptions("grid", True)
DQDSystemFactory.addToPlotOptions("plotOnly", 0)

# Optional: set title fields
DQDSystemFactory.addToTitle(DQDAttributes.DETUNING.value)
DQDSystemFactory.addToTitle(DQDAttributes.FACTOR_OME.value)

# Create and simulate system
dqdSystem = DQDSystemFactory.magneticFieldXvsMagneticFieldY(xArray, xArray)
dqdSystem.runSimulation()
dqdSystem.plotSimulation(
    title=DQDSystemFactory.getTitleForSystem(),
    options=DQDSystemFactory.getPlotOptionsForSystem(),
    saveData=True,
    saveFigure=True
)

# Print elapsed time
print("Total time:", formatComputationTime(time() - timeStart))
