from src.DQDSystem import DQDSystem
import numpy as np
import matplotlib.pyplot as plt
from src.base.auxiliaryMethods import formatComputationTime
from src.base.DoubleQuantumDot import DQDAttributes
from time import time

timeStart = time()

xArray = np.linspace(-4, 4, 50)
yArray = np.linspace(0.0, 6, 50)
zArray = np.linspace(-2, 2, 50)


arrays = [xArray, yArray, zArray]
iterationParameters = [
    {"features": DQDAttributes.DETUNING.value,
     "label": "delta"},
    {"features": DQDAttributes.AC_AMPLITUDE.value,
     "label": "acAmp"},
    {"features": DQDAttributes.ZEEMAN.value + "Z",
     "label": "zeeman"},
]

dqdSystem = DQDSystem({}, arrays, iterationParameters)
current, polarity = dqdSystem.bidimensionalSimulation(0,2)


timeEnd = time() - timeStart
print("Total time: ", formatComputationTime(timeEnd))
# Crear la gráfica pcolor
plt.figure(figsize=(8, 6))
plt.pcolor(xArray, yArray, current, shading='auto', cmap='viridis')
plt.colorbar(label='Sum Current')
plt.xlabel('X Axis (delta)')
plt.ylabel('Y Axis (zeemanZ)')
plt.title('Pcolor Plot of Sum Current')
plt.show()

