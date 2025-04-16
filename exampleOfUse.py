from src.DQDSystem import DQDSystem
import numpy as np
from time import time
from src.base.DoubleQuantumDot import DQDAttributes

# Inicia el temporizador
timeStart = time()

# Define los arrays de iteración
xArray = np.linspace(-4, 4, 100)  # Para detuning
zArray = np.linspace(-2, 2, 80)   # Para zeemanZLeft

# Define los parámetros de iteración
iterationParameters = [
    {"array": xArray, "features": DQDAttributes.DETUNING.value},
    {"array": zArray, "features": DQDAttributes.ZEEMAN.value + "Z"},
]

# Permite seleccionar si usar uno o ambos arrays
useBothParameters = True  # Cambia a False para prueba unidimensional

# Define los parámetros fijos
fixedParameters = {}

# Configura el sistema DQD
if useBothParameters:
    dqdSystem = DQDSystem(fixedParameters, iterationParameters)
else:
    dqdSystem = DQDSystem(fixedParameters, [{"array": xArray, "features": DQDAttributes.DETUNING.value}])

# Define las opciones de ploteo
plotOptions = {
    "grid": True,
    "applyToAll": True,
    "colorBarMin": 0.02,
    "colorBarMax": 0.35,
    "plotOnly": None,
    "logColorBar": True,
    "gaussianFilter": True
}

# Opcional: título personalizado como lista de strings para concatenar
titleOptions = []

# Ejecuta la simulación y genera los gráficos
dqdSystem.simulateAndPlot(
    title=titleOptions,
    options=plotOptions,
    saveData=True,       # Guarda los datos como .npz
    saveFigure=True      # Guarda la figura como .pdf
)

# Calcula y muestra el tiempo total de ejecución
timeEnd = time() - timeStart
print("Total time: {:.2f} seconds".format(timeEnd))


