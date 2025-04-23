from time import time

import numpy as np

from src.DQDSystemFactory import DQDSystemFactory
from src.UnifiedParameters import UnifiedParameters
from src.base.DoubleQuantumDot import DQDAttributes
from src.base.auxiliaryMethods import formatComputationTime

# Inicia el temporizador
timeStart = time()

DQDSystemFactory.changeParameter(UnifiedParameters.DETUNING.value, 0.8)

arrayValues = np.linspace(-3, 3, 300)  # Para zeeman

dqdSystem = DQDSystemFactory.ZeemanZ(arrayValues)
dqdSystem.runSimulation()

plotOptions = {
    "grid": True,
    "applyToAll": False,
    "colorBarMin": 0.02,
    "colorBarMax": 0.35,
    "plotOnly": None,
    "logColorBar": False,
    "gaussianFilter": False
}

titleOptions = [DQDAttributes.DETUNING.value]

# Ejecuta la simulación y genera los gráficos
dqdSystem.plotSimulation(
    title=titleOptions,
    options=plotOptions,
    saveData=True,  # Guarda los datos como .npz
    saveFigure=True  # Guarda la figura como .pdf
)

# Calcula y muestra el tiempo total de ejecución
timeEnd = time() - timeStart
print("Total time: {}".format(formatComputationTime(timeEnd)))
