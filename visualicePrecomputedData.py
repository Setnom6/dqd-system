from src.DQDSystem import DQDSystem
from src.base.DoubleQuantumDot import DQDAttributes

# Define los parámetros de iteración
iterationParameters = [
    {"features": DQDAttributes.ZEEMAN.value + "X"},
    {"features": DQDAttributes.ZEEMAN.value + "Y"},
]

# Carga el DQDSystem precomputado   

dqdSystem = DQDSystem.loadData(iterationParameters)

# Ver simulación directamente

dqdSystem.plotSimulation(saveData=False, saveFigure=True)
