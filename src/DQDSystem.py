import numpy as np
from typing import Dict, Any, List, Tuple
from src.base.AttributeInterpreter import AttributeInterpreter
from src.base.DoubleQuantumDot import DoubleQuantumDot
from src.base.SimulationManager import SimulationManager


class DQDSystem:
    def __init__(self, fixedParameters: Dict[str, Any], iterationParameters: List[Dict[str, Any]]) -> None:
        self.attributeInterpreter = AttributeInterpreter(fixedParameters, iterationParameters)
        self.dqdObject = self.createDqdObject()

    def createDqdObject(self):
        adjustedParams = self.attributeInterpreter.fixedParameters
        return DoubleQuantumDot(adjustedParams)

    def updateDQDParameters(self, i:int, j:int):
        previousXValue = self.dqdObject.getAttributeValue(self.attributeInterpreter.xName())
        previousYValue = self.dqdObject.getAttributeValue(self.attributeInterpreter.yName())

        updaterFunctions = self.attributeInterpreter.getUpdateFunctions(i,j)
        newDictX = updaterFunctions[0](previousXValue)
        newDictY = updaterFunctions[1](previousYValue)

        self.dqdObject.setParameters(newDictX)
        self.dqdObject.setParameters(newDictY)

    def simulatePoint(self) -> Tuple[float, float]:
        sumCurrent, polarity  = self.dqdObject.getCurrent()
        return sumCurrent, polarity

    def bidimensionalSimulation(self) -> Tuple[np.ndarray, np.ndarray]:
        lenX = len(self.attributeInterpreter.iterationArrays[0])
        lenY = len(self.attributeInterpreter.iterationArrays[1])
        simulationManager = SimulationManager(
            [lenX, lenY],
            self.updateDQDParameters,
            self.simulatePoint
        )
        return simulationManager.bidimensionalSimulation()


