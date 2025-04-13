from src.DQDSystem import DQDSystem
import numpy as np
import matplotlib.pyplot as plt
from src.base.auxiliaryMethods import formatComputationTime
from src.base.DoubleQuantumDot import DQDAttributes
from time import time

timeStart = time()

xArray = np.linspace(-4, 4, 100)
zArray = np.linspace(-2, 2, 100)


iterationParameters = [
    {"array": xArray,
        "features": DQDAttributes.DETUNING.value,
     "label": "delta"},
    {"array": zArray,
        "features": DQDAttributes.ZEEMAN.value + "Z",
     "label": "zeeman"},
]

dqdSystem = DQDSystem({}, iterationParameters)
current, polarity = dqdSystem.bidimensionalSimulation()


timeEnd = time() - timeStart
print("Total time: ", formatComputationTime(timeEnd))
# Crear la gráfica pcolor
plt.figure(figsize=(8, 6))
plt.pcolor(xArray, zArray, current, shading='auto', cmap='viridis')
plt.colorbar(label='Sum Current')
plt.xlabel('X Axis (delta)')
plt.ylabel('Y Axis (zeemanZ)')
plt.title('Pcolor Plot of Sum Current')
plt.show()

