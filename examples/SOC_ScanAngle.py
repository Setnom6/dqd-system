from time import time

import numpy as np

from src.DQDSystemFactory import DQDSystemFactory
from src.PlotOptionsManager import PlotOptions
from src.base.DoubleQuantumDot import DQDAttributes
from src.base.auxiliaryMethods import formatComputationTime

"""
Script for observe the current resonances when varying over the modulus of the magnetic field and the angle in the XY plane
"""

# Start timer
timeStart = time()

# Define iteration arrays
scanAngleValues = np.linspace(0, 1, 30)  # scan_angle (in units of π)
magneticFieldModulus = np.linspace(0.0, 2.5, 30)  # magnetic field

# Set fixed simulation parameters
DQDSystemFactory.changeParameter(DQDAttributes.DETUNING.value, 0.8)
DQDSystemFactory.changeParameter(DQDAttributes.FACTOR_OME.value, 0.5)
DQDSystemFactory.changeParameter(DQDAttributes.SOC_THETA_ANGLE.value, 90 * np.pi / 180)
DQDSystemFactory.changeParameter(DQDAttributes.SOC_PHI_ANGLE.value, 0.0 * np.pi / 180)

# Configure plot options
DQDSystemFactory.addToPlotOptions(PlotOptions.GRID.value, True)
DQDSystemFactory.addToPlotOptions(PlotOptions.APPLY_TO_ALL.value, False)
DQDSystemFactory.addToPlotOptions(PlotOptions.COLOR_BAR_MIN.value, 0.02)
DQDSystemFactory.addToPlotOptions(PlotOptions.COLOR_BAR_MAX.value, 0.35)
DQDSystemFactory.addToPlotOptions(PlotOptions.PLOT_ONLY.value, None)
DQDSystemFactory.addToPlotOptions(PlotOptions.LOG_COLOR_BAR.value, False)
DQDSystemFactory.addToPlotOptions(PlotOptions.GAUSSIAN_FILTER.value, False)
DQDSystemFactory.addToPlotOptions(PlotOptions.ANNOTATE.value, True)

# Set custom title fields
DQDSystemFactory.addToTitle(DQDAttributes.DETUNING.value)
DQDSystemFactory.addToTitle(DQDAttributes.FACTOR_OME.value)
DQDSystemFactory.addToTitle(DQDAttributes.SOC_PHI_ANGLE.value)
DQDSystemFactory.addToTitle(DQDAttributes.SOC_THETA_ANGLE.value)

# Create and simulate the system
dqdSystem = DQDSystemFactory.scanAngleVsMagneticFieldModule(scanAngleValues, magneticFieldModulus)

# Plot results using factory-managed options and title
dqdSystem.plotSimulation(
    title=DQDSystemFactory.getTitleForSystem(),
    options=DQDSystemFactory.getPlotOptionsForSystem(),
    saveData=True,
    saveFigure=True
)

# Print total execution time
print("Total time:", formatComputationTime(time() - timeStart))
