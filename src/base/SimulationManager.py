from joblib import Parallel, delayed
import numpy as np
from typing import Callable, List, Tuple

class SimulationManager:
    def __init__(
        self,
        lenArrays: List[int],
        updateParameters: Callable[[int, int], None],
        simulatePoint: Callable[[], Tuple[float, float]]
    ):
        self.lenArrays = lenArrays  # [lenX, lenY]
        self.updateParameters = updateParameters
        self.simulatePoint = simulatePoint

    def bidimensionalSimulation(self) -> Tuple[np.ndarray, np.ndarray]:
        lenX, lenY = self.lenArrays

        # Ejecutamos las simulaciones en paralelo
        results = Parallel(n_jobs=-1)(
            delayed(self._simulate_point)(i, j)
            for i in range(lenX)
            for j in range(lenY)
        )

        # Inicializamos los arrays de resultados
        sumCurrent = np.zeros((lenY, lenX))
        polarity = np.zeros((lenY, lenX))

        # Llenamos con los resultados
        for i, j, currentSum, currentPol in results:
            sumCurrent[j, i] = currentSum
            polarity[j, i] = currentPol

        return sumCurrent, polarity

    def _simulate_point(self, i: int, j: int) -> Tuple[int, int, float, float]:
        self.updateParameters(i, j)
        currentSum, currentPol = self.simulatePoint()
        return i, j, currentSum, currentPol