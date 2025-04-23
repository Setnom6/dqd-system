from time import time

import numpy as np

from src.DQDSystem import DQDSystem
from src.UnifiedParameters import UnifiedParameters
from src.base.auxiliaryMethods import formatComputationTime

# Inicia el temporizador
timeStart = time()

# Define los arrays de iteración
xArray = np.linspace(0, 1, 50)  # para scan_angle en unidades de PI
yArray = np.linspace(0, 2.5, 50)  # para magnetic field

# Define los parámetros de iteración
iterationParameters = [
    {"array": xArray, "features": UnifiedParameters.SCAN_ANGLE.value},
    {"array": yArray, "features": UnifiedParameters.MAGNETIC_FIELD.value + "M"},
]

# Define los parámetros fijos
factorOME = 0.0
fixedParameters = {UnifiedParameters.DETUNING.value: 0.8,
                   UnifiedParameters.FACTOR_OME.value: factorOME,
                   UnifiedParameters.MAGNETIC_FIELD.value: [1.0, 0.0, 0.0]}

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

dqdSystem = DQDSystem(fixedParameters, iterationParameters)
dqdSystem.runSimulation()
dqdSystem.plotSimulation(title=titleOptions, options=plotOptions, saveData=True, saveFigure=True)

# Calcula y muestra el tiempo total de ejecución
timeEnd = time() - timeStart
print("Total time: {}".format(formatComputationTime(timeEnd)))
