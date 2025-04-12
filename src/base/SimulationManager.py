from joblib import Parallel, delayed
import numpy as np
from typing import Callable, List, Tuple

class SimulationManager:
    def __init__(self, simulationArrays: List[np.ndarray], simulate_point: Callable[[int, int], Tuple[float, float]]):
        """
        Args:
            simulationArrays (List[np.ndarray]): Arrays de iteración para la simulación.
            simulate_point (Callable): Función que simula un punto dado (i, j).
        """
        self.simulationArrays = simulationArrays
        self.simulate_point = simulate_point

    def bidimensionalSimulation(self) -> Tuple[np.ndarray, np.ndarray]:
        arrayX = self.simulationArrays[0]
        arrayY = self.simulationArrays[1]
        numXValues = len(arrayX)
        numYValues = len(arrayY)

        # Parallelize the simulation over all grid points
        results = Parallel(n_jobs=-1)(
            delayed(self.simulate_point)(i, j) for i in range(numXValues) for j in range(numYValues)
        )

        # Initialize the current array
        currentArray = np.zeros((numYValues, numXValues, 2))

        # Populate the current array with the results
        for j, i, current1, current2 in results:
            currentArray[j, i, 0] = current1
            currentArray[j, i, 1] = current2

        # Compute sumCurrent and polarity
        sumCurrent = self.computeSumCurrent(currentArray)
        polarity = self.computePolarity(currentArray)

        return sumCurrent, polarity

    def computeSumCurrent(self, currentArray: np.ndarray) -> np.ndarray:
        return np.sum(currentArray, axis=2)

    def computePolarity(self, currentArray: np.ndarray) -> np.ndarray:
        return np.divide(
            (currentArray[:, :, 0] - currentArray[:, :, 1]),
            (currentArray[:, :, 0] + currentArray[:, :, 1]),
            out=np.zeros_like(currentArray[:, :, 0]),
            where=(currentArray[:, :, 0] + currentArray[:, :, 1]) != 0
        )