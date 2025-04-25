from time import time

import numpy as np

from examples.PredefinedQuantities import PredefinedQuantities
from src.DQDSystemFactory import DQDSystemFactory, DQDAttributes
from src.PlotOptionsManager import PlotOptions
from src.base.auxiliaryMethods import formatComputationTime

"""
Script for generating the resonances varying over the three directions of the magnetic field with an special election of g tensor
"""

# Start timer
timeStart = time()

# Define iteration array for the three plots
magneticFieldArray = np.linspace(-2.5, 2.5, 300)

# Set fixed parameters
classToPlot = "A"

DQDSystemFactory.changeParameter(DQDAttributes.DETUNING.value, 0.0)
DQDSystemFactory.changeParameter(DQDAttributes.FACTOR_OME.value, 0.0)
DQDSystemFactory.changeParameter(DQDAttributes.MAGNETIC_FIELD.value, [1.0, 1.0, 0.0])
DQDSystemFactory.changeParameter(DQDAttributes.G_FACTOR.value,
                                 PredefinedQuantities.getSelectedClassCompleteGTensor(classToPlot))

# Optional: modify plot options
DQDSystemFactory.addToPlotOptions(PlotOptions.GRID.value, True)
DQDSystemFactory.addToPlotOptions(PlotOptions.PLOT_ONLY.value, 0)

# Optional: set title fields
DQDSystemFactory.addToTitle(DQDAttributes.DETUNING.value)
DQDSystemFactory.addToTitle(DQDAttributes.FACTOR_OME.value)
DQDSystemFactory.addToTitle(f"Geometric class: {classToPlot}")

# Create and simulate system for X-Y
dqdSystem = DQDSystemFactory.magneticFieldXvsMagneticFieldY(magneticFieldArray, magneticFieldArray)
dqdSystem.runSimulation()
dqdSystem.plotSimulation(
    title=DQDSystemFactory.getTitleForSystem(),
    options=DQDSystemFactory.getPlotOptionsForSystem(),
    saveData=True,
    saveFigure=True
)

# Create and simulate system for X-Z
dqdSystem = DQDSystemFactory.magneticFieldXvsMagneticFieldZ(magneticFieldArray, magneticFieldArray)
dqdSystem.runSimulation()
dqdSystem.plotSimulation(
    title=DQDSystemFactory.getTitleForSystem(),
    options=DQDSystemFactory.getPlotOptionsForSystem(),
    saveData=True,
    saveFigure=True
)

# Create and simulate system for Y-Z
dqdSystem = DQDSystemFactory.magneticFieldYvsMagneticFieldZ(magneticFieldArray, magneticFieldArray)
dqdSystem.runSimulation()
dqdSystem.plotSimulation(
    title=DQDSystemFactory.getTitleForSystem(),
    options=DQDSystemFactory.getPlotOptionsForSystem(),
    saveData=True,
    saveFigure=True
)

# Print elapsed time
print("Total time:", formatComputationTime(time() - timeStart))
