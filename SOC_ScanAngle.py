from time import time

import numpy as np

from src.DQDSystemFactory import DQDSystemFactory
from src.base.DoubleQuantumDot import DQDAttributes
from src.base.auxiliaryMethods import formatComputationTime

# Start timer
timeStart = time()

# Define iteration arrays
xArray = np.linspace(0, 1, 50)  # scan_angle (in units of π)
yArray = np.linspace(1.7, 1.9, 5)  # magnetic field

# Set fixed simulation parameters
DQDSystemFactory.changeParameter(DQDAttributes.DETUNING.value, 0.8)
DQDSystemFactory.changeParameter(DQDAttributes.FACTOR_OME.value, 0.5)
DQDSystemFactory.changeParameter(DQDAttributes.MAGNETIC_FIELD.value, [1.0, 1.0, 0.0])

# Configure plot options
DQDSystemFactory.addToPlotOptions("grid", True)
DQDSystemFactory.addToPlotOptions("applyToAll", False)
DQDSystemFactory.addToPlotOptions("colorBarMin", None)
DQDSystemFactory.addToPlotOptions("colorBarMax", None)
DQDSystemFactory.addToPlotOptions("plotOnly", 0)
DQDSystemFactory.addToPlotOptions("logColorBar", False)
DQDSystemFactory.addToPlotOptions("gaussianFilter", False)
DQDSystemFactory.addToPlotOptions("Draw1DLines", 1)

# Set custom title fields
DQDSystemFactory.addToTitle(DQDAttributes.DETUNING.value)
DQDSystemFactory.addToTitle(DQDAttributes.FACTOR_OME.value)
DQDSystemFactory.addToTitle(DQDAttributes.SOC_PHI_ANGLE.value)
DQDSystemFactory.addToTitle(DQDAttributes.SOC_THETA_ANGLE.value)

# Create and simulate the system
dqdSystem = DQDSystemFactory.scanAngleVsMagneticFieldModule(xArray, yArray)
dqdSystem.runSimulation()

# Plot results using factory-managed options and title
dqdSystem.plotSimulation(
    title=DQDSystemFactory.getTitleForSystem(),
    options=DQDSystemFactory.getPlotOptionsForSystem(),
    saveData=True,
    saveFigure=True
)

# Print total execution time
print("Total time:", formatComputationTime(time() - timeStart))
