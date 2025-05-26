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

# Set fixed simulation parameters
bidimensional = False
polarityPlot = False
gaussianFilter = False
n = 1
sign = -1
detuning = 0.8
kOME = 0.5
convertToPi = np.pi / 180
alphaTheta = 90 * convertToPi
alphaPhi = 20 * convertToPi
loadData = False

# G Factors

gL = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
gR = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

# Define parameters
scanAngleValues = np.linspace(0, 1, 300)  # scan_angle (in units of π)
magneticFieldModulus = np.linspace(0.0, 2.5, 300)  # magnetic field

DQDSystemFactory.changeParameter(DQDAttributes.DETUNING.value, detuning)
DQDSystemFactory.changeParameter(DQDAttributes.FACTOR_OME.value, kOME)
DQDSystemFactory.changeParameter(DQDAttributes.SOC_THETA_ANGLE.value, alphaTheta)
DQDSystemFactory.changeParameter(DQDAttributes.SOC_PHI_ANGLE.value, alphaPhi)
DQDSystemFactory.changeParameter(DQDAttributes.G_FACTOR.value + "Left", gL)
DQDSystemFactory.changeParameter(DQDAttributes.G_FACTOR.value + "Right", gR)

# Configure plot options
plotOnly = None
if not polarityPlot:
    plotOnly = 0
DQDSystemFactory.addToPlotOptions(PlotOptions.GRID.value, True)
DQDSystemFactory.addToPlotOptions(PlotOptions.APPLY_TO_ALL.value, False)
DQDSystemFactory.addToPlotOptions(PlotOptions.COLOR_BAR_MIN.value, None)
DQDSystemFactory.addToPlotOptions(PlotOptions.COLOR_BAR_MAX.value, None)
DQDSystemFactory.addToPlotOptions(PlotOptions.PLOT_ONLY.value, plotOnly)
DQDSystemFactory.addToPlotOptions(PlotOptions.LOG_COLOR_BAR.value, False)
DQDSystemFactory.addToPlotOptions(PlotOptions.GAUSSIAN_FILTER.value, gaussianFilter)

if bidimensional:
    DQDSystemFactory.addToPlotOptions(PlotOptions.DRAW_1D_LINES.value, None)
    DQDSystemFactory.addToPlotOptions(PlotOptions.ANNOTATE.value, True)
else:
    line = (n + sign * detuning)
    DQDSystemFactory.addToPlotOptions(PlotOptions.DRAW_1D_LINES.value, 1)
    DQDSystemFactory.addToPlotOptions(PlotOptions.ANNOTATE.value, False)
    magneticFieldModulus = np.linspace(line - 0.05 * line, line + 0.05 * line, 5)  # magnetic field
    if sign == +1:
        sign_str = "+"
    elif sign == -1:
        sign_str = "-"
    DQDSystemFactory.addToTitle(f"n = {n}, n {sign_str} " r"$\delta / \omega = $" + f"{line:.2f}")

# Set custom title fields common
DQDSystemFactory.addToTitle(DQDAttributes.DETUNING.value)
DQDSystemFactory.addToTitle(DQDAttributes.FACTOR_OME.value)
DQDSystemFactory.addToTitle(DQDAttributes.SOC_PHI_ANGLE.value)
DQDSystemFactory.addToTitle(DQDAttributes.SOC_THETA_ANGLE.value)

# Create and simulate the system
dqdSystem = DQDSystemFactory.scanAngleVsMagneticFieldModule(scanAngleValues, magneticFieldModulus, loadData=loadData)

# Plot results using factory-managed options and title
dqdSystem.plotSimulation(
    title=DQDSystemFactory.getTitleForSystem(),
    options=DQDSystemFactory.getPlotOptionsForSystem(),
    saveData=True,
    saveFigure=True
)

# Print total execution time
print("Total time:", formatComputationTime(time() - timeStart))
