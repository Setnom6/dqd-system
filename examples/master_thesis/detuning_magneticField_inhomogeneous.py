from time import time, sleep

import numpy as np

from src.DQDSystemFactory import DQDSystemFactory, DQDAttributes
from src.PlotOptionsManager import PlotOptions
from src.base.auxiliaryMethods import formatComputationTime

# Define the iteration parameters for X axis and Y axis. The X axis one will be the same for the two dots
# The range will be the same for all parameters (can be different in different axes)

detuningValues = np.linspace(-1.0, 1.0, 200)
zeemanValues = np.linspace(0.5, 1.5, 200)

# Changes in specific parameters

# Title

# Optional: configure plot options globally
DQDSystemFactory.addToPlotOptions(PlotOptions.GRID.value, True)
DQDSystemFactory.addToPlotOptions(PlotOptions.APPLY_TO_ALL.value, False)
DQDSystemFactory.addToPlotOptions(PlotOptions.COLOR_BAR_MIN.value, None)
DQDSystemFactory.addToPlotOptions(PlotOptions.COLOR_BAR_MAX.value, None)
DQDSystemFactory.addToPlotOptions(PlotOptions.PLOT_ONLY.value, None)
DQDSystemFactory.addToPlotOptions(PlotOptions.LOG_COLOR_BAR.value, False)
DQDSystemFactory.addToPlotOptions(PlotOptions.GAUSSIAN_FILTER.value, False)

# Start the timer
timeStart = time()

proportionalityFactors = [1.0, 1.25, 1.5, 1.75, 2.0, 2.5]
proportionalityFactors = [1.0, 1.1, 1.25, 1.5, 2.0]
axes = [2]
OMEFactors = [0.0]
polarityPlot = True
for OME in OMEFactors:
    for factor in proportionalityFactors:
        for i in axes:
            DQDSystemFactory.changeParameter(DQDAttributes.FACTOR_OME.value, OME)
            DQDSystemFactory.changeParameter(DQDAttributes.G_FACTOR.value + "Right",
                                             [[factor, 0.0, 0.0], [0.0, factor, 0.0], [0.0, 0.0, factor]])
            DQDSystemFactory.addToTitle(r"$g_{{R}} =$" + f"{factor}" + r"$g_{{L}}$")
            # Create and simulate both systems (Here we select the kind of simulation we want)
            dqdSystem = DQDSystemFactory.detuningvsOneMagneticField(detuningValues, zeemanValues, axis=i,
                                                                    loadData=False)

            DQDSystemFactory.addToPlotOptions(PlotOptions.PLOT_ONLY.value, 0)
            # Plot comparison
            dqdSystem.plotSimulation(
                title=DQDSystemFactory.getTitleForSystem(),
                options=DQDSystemFactory.getPlotOptionsForSystem(),
                saveData=True,
                saveFigure=True
            )

            if polarityPlot:
                sleep(1)

                # Create and simulate both systems (Here we select the kind of simulation we want)
                dqdSystem = DQDSystemFactory.detuningvsOneMagneticField(detuningValues, zeemanValues, axis=i,
                                                                        loadData=True)

                DQDSystemFactory.addToPlotOptions(PlotOptions.PLOT_ONLY.value, 1)
                # Plot comparison
                dqdSystem.plotSimulation(
                    title=DQDSystemFactory.getTitleForSystem(),
                    options=DQDSystemFactory.getPlotOptionsForSystem(),
                    saveData=True,
                    saveFigure=True
                )

            DQDSystemFactory.resetTitle()

# Show elapsed time
print("Total time:", formatComputationTime(time() - timeStart))
