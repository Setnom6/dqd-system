import os
from typing import Callable, List, Tuple, Any

import numpy as np
from dotenv import load_dotenv
from joblib import Parallel, delayed


class SimulationManager:
    def __init__(
            self,
            lenArrays: List[int],
            updateParameters: Callable[..., None],
            simulatePoint: Callable[..., Tuple[Any, ...]]
    ):
        self.lenArrays = lenArrays
        self.updateParameters = updateParameters
        self.simulatePoint = simulatePoint
        self.parallelEnabled = self._checkParallelEnabled()

    @staticmethod
    def _checkParallelEnabled() -> bool:
        """
        Checks whether parallel execution should be enabled based on an environment variable.
        Set DQD_PARALLEL=0 to disable.
        """
        load_dotenv()
        envVar = int(os.getenv("DQD_PARALLEL", "1").strip())
        return bool(envVar)

    def runSimulation(self) -> List[np.ndarray]:
        gridShape = tuple(self.lenArrays)

        if self.parallelEnabled:
            results = Parallel(n_jobs=-1)(
                delayed(self._simulatePoint)(*indices)
                for indices in np.ndindex(gridShape)
            )
        else:
            results = [self._simulatePoint(*indices) for indices in np.ndindex(gridShape)]
            print("No parallel execution is taking place.")

        numOutputs = len(results[0]) - len(gridShape)
        resultArrays = [np.zeros(gridShape) for _ in range(numOutputs)]

        for result in results:
            indices = result[:len(gridShape)]
            outputs = result[len(gridShape):]
            for k, output in enumerate(outputs):
                resultArrays[k][tuple(indices)] = output

        return resultArrays

    def _simulatePoint(self, *indices: int) -> Tuple[int, ...]:
        self.updateParameters(*indices)
        outputs = self.simulatePoint()
        return (*indices, *outputs)
