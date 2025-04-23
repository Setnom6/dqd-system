from time import time

import numpy as np

from src.DQDSystemFactory import DQDSystemFactory
from src.UnifiedParameters import UnifiedParameters
from src.base.auxiliaryMethods import formatComputationTime

# Inicia el temporizador
timeStart = time()

# Define los arrays de iteración
xArray = np.linspace(0, 1, 50)  # para scan_angle en unidades de PI
yArray = np.linspace(0, 2.5, 50)  # para magnetic field

DQDSystemFactory.changeParameter(UnifiedParameters.DETUNING.value, 0.8)
DQDSystemFactory.changeParameter(UnifiedParameters.FACTOR_OME.value, 0.0)
DQDSystemFactory.changeParameter(UnifiedParameters.MAGNETIC_FIELD.value, [1.0, 0.0, 0.0])

dqdSystem = DQDSystemFactory.ScanAngleVsMagneticField(xArray, yArray)

# Define las opciones de ploteo
plotOptions = {
    "grid": True,
    "applyToAll": False,
    "colorBarMin": None,
    "colorBarMax": None,
    "plotOnly": 0,
    "logColorBar": False,
    "gaussianFilter": False
}
# Opcional: título personalizado como lista de strings para concatenar
titleOptions = [UnifiedParameters.DETUNING.value, UnifiedParameters.FACTOR_OME.value,
                UnifiedParameters.ALPHA_PHI_ANGLE.value,
                UnifiedParameters.ALPHA_THETA_ANGLE.value]

dqdSystem.runSimulation()
dqdSystem.plotSimulation(title=titleOptions, options=plotOptions, saveData=True, saveFigure=True)

# Calcula y muestra el tiempo total de ejecución
timeEnd = time() - timeStart
print("Total time: {}".format(formatComputationTime(timeEnd)))
