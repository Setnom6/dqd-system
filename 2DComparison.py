from time import time

import numpy as np

from src.DQDSystemFactory import DQDSystemFactory
from src.UnifiedParameters import UnifiedParameters
from src.base.auxiliaryMethods import formatComputationTime

# Inicia el temporizador
timeStart = time()

# Define los arrays de iteración
xArray = np.linspace(-3, 3, 50)  # Para zeeman

DQDSystemFactory.changeParameter(UnifiedParameters.DETUNING.value, 0.8)

dqdSystem = DQDSystemFactory.ZeemanXvsZeemanZ(xArray, xArray)
dqdSystem.runSimulation()
otherSystem = DQDSystemFactory.ZeemanXvsZeemanY(xArray, xArray)

# Define las opciones de ploteo
plotOptions = {
    "grid": True,
    "applyToAll": False,
    "colorBarMin": 0.02,
    "colorBarMax": 0.35,
    "plotOnly": None,
    "logColorBar": False,
    "gaussianFilter": False
}

# Opcional: título personalizado como lista de strings para concatenar
titleOptions = [UnifiedParameters.DETUNING.value]

dqdSystem.compareSimulationsAndPlot(
    otherSystemDict=otherSystem,
    title=titleOptions,
    options=plotOptions,
    saveData=True,  # Guarda los datos como .npz
    saveFigure=True  # Guarda la figura como .pdf
)

# Calcula y muestra el tiempo total de ejecución
timeEnd = time() - timeStart
print("Total time: {}".format(formatComputationTime(timeEnd)))
