from time import time

import numpy as np

from src.DQDSystem import DQDSystem
from src.base.DoubleQuantumDot import DQDAttributes
from src.base.auxiliaryMethods import formatComputationTime

# Inicia el temporizador
timeStart = time()

# Define los arrays de iteración
xArray = np.linspace(-3, 3, 50)  # Para zeeman

# Define los parámetros de iteración
iterationParameters = [
    {"array": xArray, "features": DQDAttributes.ZEEMAN.value + "X"},
    {"array": xArray, "features": DQDAttributes.ZEEMAN.value + "Y"},
]

# Define los parámetros fijos
fixedParameters = {DQDAttributes.DETUNING.value: 0.08}

dqdSystem = DQDSystem(fixedParameters, iterationParameters)

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
titleOptions = [DQDAttributes.DETUNING.value]

iterationParametersForCompare = [
    {"array": xArray, "features": DQDAttributes.ZEEMAN.value + "X"},
    {"array": xArray, "features": DQDAttributes.ZEEMAN.value + "Z"},
]
# Ejecuta la simulación y genera los gráficos

otherDQD = DQDSystem(fixedParameters, iterationParametersForCompare)
dqdSystem.runSimulation()
dqdSystem.compareSimulationsAndPlot(
    otherSystemDict=otherDQD,
    title=titleOptions,
    options=plotOptions,
    saveData=True,  # Guarda los datos como .npz
    saveFigure=True  # Guarda la figura como .pdf
)

# Calcula y muestra el tiempo total de ejecución
timeEnd = time() - timeStart
print("Total time: {}".format(formatComputationTime(timeEnd)))
