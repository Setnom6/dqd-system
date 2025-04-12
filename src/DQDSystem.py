import numpy as np
from typing import Dict, Any, List, Tuple
from src.base.attributeInterpreter import processFixedParameters, processIterationParameters, obtainNewVariableToSet
from src.base.DoubleQuantumDot import DoubleQuantumDot
from src.base.SimulationManager import SimulationManager


class DQDSystem:
    def __init__(self, fixedParameters: Dict[str, Any], arrays: List[np.ndarray],
                 iterationParameters: List[Dict[str, str]]) -> None:
        if len(arrays) != len(iterationParameters):
            raise ValueError("The number of arrays must match the number of iteration parameters.")

        for i, array in enumerate(arrays):
            if not isinstance(array, np.ndarray):
                raise TypeError(f"The array at index {i} is not a numpy array.")

        self.dqdObject = self.createDqdObject(fixedParameters)
        self.iterationArrays = arrays
        self.parametersInfo = processIterationParameters(iterationParameters)

    def createDqdObject(self, fixedParams: Dict[str, Any]):
        adjustedParams = processFixedParameters(fixedParams)
        return DoubleQuantumDot(adjustedParams)

    def simulatePoint(self, i: int, j: int, parameterX: int, parameterY: int) -> Tuple[int, int, float, float]:
        """
        Simulates a single point in the grid.

        Args:
            i (int): Index for the X parameter.
            j (int): Index for the Y parameter.
            parameterX (int): Index of the X parameter in iterationArrays.
            parameterY (int): Index of the Y parameter in iterationArrays.

        Returns:
            Tuple[int, int, float, float]: Indices and current values.
        """
        newXDictToSet = obtainNewVariableToSet(self.parametersInfo, self.iterationArrays, self.dqdObject, i, parameterX)
        newYDictToSet = obtainNewVariableToSet(self.parametersInfo, self.iterationArrays, self.dqdObject, j, parameterY)
        self.dqdObject.setParameters(newXDictToSet)
        self.dqdObject.setParameters(newYDictToSet)

        result = self.dqdObject.simulate()
        current1 = np.mean(result.expect[3])
        current2 = np.mean(result.expect[4])
        return j, i, current1, current2

    def bidimensionalSimulation(self, parameterX: int = 0, parameterY: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        simulationArrays = [self.iterationArrays[parameterX], self.iterationArrays[parameterY]]
        simulationManager = SimulationManager(
            simulationArrays,
            lambda i, j: self.simulatePoint(i, j, parameterX, parameterY)
        )
        return simulationManager.bidimensionalSimulation()


