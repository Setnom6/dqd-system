from time import time, sleep

from examples.master_thesis.spinConservingFrame import *
from src.DQDSystemFactory import DQDSystemFactory, DQDAttributes
from src.PlotOptionsManager import PlotOptions
from src.base.auxiliaryMethods import formatComputationTime

# iteration arrays

detuningArray = np.linspace(-2.0, 2.0, 150)
magneticFieldArray = np.linspace(-2.0, 2.0, 150)
detuningPlot = False
barrido = True
detuningForBarrido = 0.8
applyRotatedFrame = True
polarityPlot = True

# Original parameters

tau = 0.1
chi = 0.1
socVectorTheta = np.pi / 2
socVectorPhi = 0.0
kOmeFactor = 0.0
axisMagneticField = 2

gL = np.array([[3.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 2.0]])
gROriginal = np.array([[3.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 2.0]])

# G-tensors

axisToRotate = "Y"
convertToRad = np.pi / 180
anglesToRotate = np.array(
    [0.0, 30 * convertToRad, 45 * convertToRad, 60 * convertToRad, 90 * convertToRad, 180 * convertToRad])

# Start timer
timeStart = time()

# Options


# Optional: configure plot options globally
DQDSystemFactory.addToPlotOptions(PlotOptions.GRID.value, True)
DQDSystemFactory.addToPlotOptions(PlotOptions.APPLY_TO_ALL.value, False)
DQDSystemFactory.addToPlotOptions(PlotOptions.COLOR_BAR_MIN.value, None)
DQDSystemFactory.addToPlotOptions(PlotOptions.COLOR_BAR_MAX.value, None)
DQDSystemFactory.addToPlotOptions(PlotOptions.PLOT_ONLY.value, None)
DQDSystemFactory.addToPlotOptions(PlotOptions.LOG_COLOR_BAR.value, False)
DQDSystemFactory.addToPlotOptions(PlotOptions.GAUSSIAN_FILTER.value, False)

for angleToRotate in anglesToRotate:
    gR = rotateMatrix(gROriginal, angleToRotate, axisToRotate)
    chiEff = chi
    tauEff = tau
    title_str = "Angular misalignment: " + f"{angleToRotate * 180 / np.pi:.2f}º around axis {axisToRotate}"
    if applyRotatedFrame:
        gR, tauEff = applySpinConservingRotation(gR, tau, chi, socVectorTheta, socVectorPhi)
        chiEff = 0.0
        title_str = title_str + f", class {computeGeometricalClass(gL, gR)}"

    DQDSystemFactory.changeParameter(DQDAttributes.CHI.value, chiEff)
    DQDSystemFactory.changeParameter(DQDAttributes.TAU.value, tauEff)
    DQDSystemFactory.changeParameter(DQDAttributes.FACTOR_OME.value, kOmeFactor)
    DQDSystemFactory.changeParameter(DQDAttributes.SOC_THETA_ANGLE.value, socVectorTheta)
    DQDSystemFactory.changeParameter(DQDAttributes.SOC_PHI_ANGLE.value, socVectorPhi)
    DQDSystemFactory.changeParameter(DQDAttributes.G_FACTOR.value + "Left", gL)
    DQDSystemFactory.changeParameter(DQDAttributes.G_FACTOR.value + "Right", gR)

    DQDSystemFactory.addToTitle(title_str)

    if detuningPlot:

        dqdSystem = DQDSystemFactory.detuningvsOneMagneticField(detuningArray, magneticFieldArray,
                                                                axis=axisMagneticField,
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
            dqdSystem = DQDSystemFactory.detuningvsOneMagneticField(detuningArray, magneticFieldArray,
                                                                    axis=axisMagneticField,
                                                                    loadData=True)

            DQDSystemFactory.addToPlotOptions(PlotOptions.PLOT_ONLY.value, 1)
            # Plot comparison
            dqdSystem.plotSimulation(
                title=DQDSystemFactory.getTitleForSystem(),
                options=DQDSystemFactory.getPlotOptionsForSystem(),
                saveData=True,
                saveFigure=True
            )

    if barrido:

        DQDSystemFactory.changeParameter(DQDAttributes.DETUNING.value, detuningForBarrido)
        dqdSystem = DQDSystemFactory.magneticFieldXvsMagneticFieldZ(magneticFieldArray, magneticFieldArray,
                                                                    loadData=False)

        DQDSystemFactory.addToPlotOptions(PlotOptions.PLOT_ONLY.value, 0)
        DQDSystemFactory.addToTitle(DQDAttributes.DETUNING.value)

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
            dqdSystem = DQDSystemFactory.magneticFieldXvsMagneticFieldZ(magneticFieldArray, magneticFieldArray,
                                                                        loadData=True)

            DQDSystemFactory.addToPlotOptions(PlotOptions.PLOT_ONLY.value, 1)
            # Plot comparison
            dqdSystem.plotSimulation(
                title=DQDSystemFactory.getTitleForSystem(),
                options=DQDSystemFactory.getPlotOptionsForSystem(),
                saveData=True,
                saveFigure=True
            )

            DQDSystemFactory.changeParameter(DQDAttributes.DETUNING.value, 0.0)

    DQDSystemFactory.resetTitle()

# Print elapsed time
print("Total time:", formatComputationTime(time() - timeStart))
