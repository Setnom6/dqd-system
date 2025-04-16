from joblib import Parallel, delayed
import numpy as np
from typing import Callable, List, Tuple, Any


class SimulationManager:
    def __init__(
        self,
        lenArrays: List[int],
        updateParameters: Callable[..., None],
        simulatePoint: Callable[..., Tuple[Any, ...]]
    ):
        """
        Args:
            lenArrays (List[int]): Lengths of the arrays for each dimension (e.g., [lenX], [lenX, lenY], [lenX, lenY, lenZ]).
            updateParameters (Callable): Function to update parameters based on indices.
            simulatePoint (Callable): Function to simulate a point, returning a tuple of results.
        """
        self.lenArrays = lenArrays
        self.updateParameters = updateParameters
        self.simulatePoint = simulatePoint

    def runSimulation(self) -> List[np.ndarray]:
        """
        Runs the simulation based on the dimensionality of lenArrays.

        Returns:
            List[np.ndarray]: A list of arrays, one for each value returned by simulatePoint.
        """
        gridShape = tuple(self.lenArrays)

        # Run simulations in parallel
        results = Parallel(n_jobs=-1)(
            delayed(self._simulatePoint)(*indices)
            for indices in np.ndindex(gridShape)
        )

        # Determine the number of outputs from simulatePoint
        numOutputs = len(results[0]) - len(gridShape)

        # Initialize result arrays
        resultArrays = [np.zeros(gridShape) for _ in range(numOutputs)]

        # Populate result arrays
        for result in results:
            indices = result[:len(gridShape)]
            outputs = result[len(gridShape):]
            for k, output in enumerate(outputs):
                resultArrays[k][tuple(indices)] = output

        return resultArrays

    def _simulatePoint(self, *indices: int) -> Tuple[int, ...]:
        """
        Simulates a single point in the grid.

        Args:
            indices (int): Indices for the current point in the grid.

        Returns:
            Tuple[int, ...]: Indices and the results of simulatePoint.
        """
        self.updateParameters(*indices)
        outputs = self.simulatePoint()
        return (*indices, *outputs)