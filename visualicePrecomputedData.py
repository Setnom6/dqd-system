from src.DQDSystem import DQDSystem
from src.base.DoubleQuantumDot import DQDAttributes

# Define los parámetros de iteración
iterationParameters = [
    {"features": DQDAttributes.ZEEMAN.value + "X"},
    {"features": DQDAttributes.ZEEMAN.value + "Y"},
]

otherSystemDict = {
    "fixedParameters": {DQDAttributes.DETUNING.value: 0.08},
    "iterationParameters": [{"features": DQDAttributes.ZEEMAN.value + "X"},
                            {"features": DQDAttributes.ZEEMAN.value + "Z"}]
}

# Carga el DQDSystem precomputado   

dqdSystem = DQDSystem.createDQDSystemFromPrecomputedData(iterationParameters, otherSystemDict=otherSystemDict)

# Ver simulación directamente

dqdSystem.compareSimulationsAndPlot(saveData=False, saveFigure=True)
