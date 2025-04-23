from src.DQDSystem import DQDSystem
from src.UnifiedParameters import UnifiedParameters

# Define los parámetros de iteración
iterationParameters = [
    {"features": UnifiedParameters.SCAN_ANGLE.value},
    {"features": UnifiedParameters.MAGNETIC_FIELD.value + "M"},
]

# Carga el DQDSystem precomputado   

dqdSystem = DQDSystem.loadData(iterationParameters)

# Ver simulación directamente

dqdSystem.plotSimulation(saveData=False, saveFigure=True)
