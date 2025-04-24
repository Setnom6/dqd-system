import re
from collections import defaultdict
from enum import Enum
from functools import lru_cache
from typing import Dict, Any, List, Tuple, Optional, Callable

import numpy as np

from src.base.DoubleQuantumDot import DQDAttributes


class Axis(Enum):
    X = 0
    Y = 1
    Z = 2
    M = 3


class Side(Enum):
    LEFT = 0
    RIGHT = 1


class NoAttributeParameters(Enum):
    SCAN_ANGLE = "scanAngle"


class DQDParameterInterpreter:
    def __init__(self, fixedParameters: Dict[str, Any], iterationParameters: List[Dict[str, Any]]):
        """
            Interprets and processes attributes for a Double Quantum Dot (DQD) system.

            This class handles fixed and iteration parameters, provides utilities for parsing and formatting attributes,
            and generates update functions for simulation parameters.

        """
        self.fixedParameters = DQDParameterInterpreter.processFixedParameters(fixedParameters)
        self.iterationArrays, self.iterationFeatures = self._processIterationParameters(iterationParameters)
        self._originalIterationParameters = iterationParameters

    def getFixedParameters(self) -> Dict[str, Any]:
        return self.fixedParameters

    def getIndependentArrays(self) -> List[np.ndarray]:
        return self.iterationArrays

    def getSimulationName(self) -> str:
        return "_".join(self.iterationFeatures)

    def getIterationFeatures(self) -> List[str]:
        return self.iterationFeatures

    def getUpdateFunctions(self, *indices: int) -> List[Tuple[Callable, str]]:
        """
                Generates update functions for simulation parameters based on indices.

                Args:
                    *indices (int): Indices for the current point in the simulation grid.

                Returns:
                    List[Tuple[Callable, str]]: A list of update functions and their corresponding parameter names.

                Raises:
                    IndexError: If an index is out of bounds for the corresponding iteration array.
        """
        updateFunctions = []
        for idx, index in enumerate(indices):
            if index >= len(self.iterationArrays[idx]):
                raise IndexError(f"Index {index} out of bounds for array of size {len(self.iterationArrays[idx])}")
            parameterName, axis, side = self.parseAttributeString(self.iterationFeatures[idx])
            newValue = self.iterationArrays[idx][index]
            updater = self._getUpdaterFunction(parameterName, axis, side, newValue)
            if parameterName == NoAttributeParameters.SCAN_ANGLE.value:
                parameterName = (
                    DQDAttributes.MAGNETIC_FIELD.value,
                    DQDAttributes.G_FACTOR.value
                )
            updateFunctions.append((updater, parameterName))
        return updateFunctions

    @staticmethod
    @lru_cache(maxsize=None)
    def parseAttributeString(attributeString: str) -> Tuple[str, Optional[int], Optional[int]]:
        """
                Parses an attribute string to extract the base name, axis, and side.

                Args:
                    attributeString (str): The attribute string to parse (e.g., "zeemanXLeft").

                Returns:
                    Tuple[str, int, int]: A tuple containing:
                        - The base name of the attribute (e.g., "zeeman").
                        - The axis index (e.g., 0 for "X").
                        - The side index (e.g., 0 for "Left").
        """
        side = None
        axis = None

        if attributeString.endswith("Left"):
            side = Side.LEFT.value
            attributeString = attributeString[:-4]
        elif attributeString.endswith("Right"):
            side = Side.RIGHT.value
            attributeString = attributeString[:-5]

        if attributeString[-1] in Axis.__members__:
            axis = Axis[attributeString[-1]].value
            attributeString = attributeString[:-1]

        return attributeString, axis, side

    @staticmethod
    def processFixedParameters(parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
                Processes and adjusts fixed parameters for the DQD system.

                Args:
                    parameters (Dict[str, Any]): Fixed parameters to process.

                Returns:
                    Dict[str, Any]: Adjusted fixed parameters.
        """

        temp = defaultdict(dict)
        adjusted = {}

        for key, value in parameters.items():
            match = re.match(r"(.+)(Left|Right)$", key)
            if match:
                baseKey, side = match.groups()
                temp[baseKey][side] = value
            else:
                adjusted[key] = value

        for baseKey, sides in temp.items():
            adjusted[baseKey] = [sides.get("Left"), sides.get("Right")]

        for key, value in adjusted.items():
            if key == DQDAttributes.GAMMA.value:
                adjusted[key] = np.array(value).reshape((2, 1))
            elif key == DQDAttributes.ZEEMAN.value:
                adjusted[key] = np.array(value).reshape((2, 3))
            elif key == DQDAttributes.MAGNETIC_FIELD.value:
                adjusted[key] = np.array(value).reshape((3,))
            elif key == DQDAttributes.G_FACTOR.value:
                value = np.array(value)
                if value.shape == (2, 3):
                    g = np.zeros((2, 3, 3))
                    for i in range(2):
                        np.fill_diagonal(g[i], value[i])
                    adjusted[key] = g
                elif value.shape == (2, 3, 3):
                    adjusted[key] = value
                else:
                    raise ValueError(f"Invalid shape for G_FACTOR: {value.shape}")
        return adjusted

    def _processIterationParameters(
            self, iterationParameters: List[Dict[str, Any]]
    ) -> Tuple[List[np.ndarray], List[str]]:
        """
                Processes iteration parameters into arrays and features.

                Args:
                    iterationParameters (List[Dict[str, Any]]): Iteration parameters to process.

                Returns:
                    Tuple[List[np.ndarray], List[str]]: Arrays of iteration parameters and their features.
        """
        arrays = []
        features = []
        for param in iterationParameters:
            features.append(param["features"])
            arrays.append(param["array"] if "array" in param else np.array([]))
        return arrays, features

    def _getUpdaterFunction(
            self, attribute: str, axis: Optional[int], side: Optional[int], newValue
    ) -> Callable:
        """
        Returns an updater function for a given attribute.

        Args:
            attribute (str): The name of the attribute to update.
            axis (Optional[int]): The axis to update (if applicable).
            side (Optional[int]): The side to update (if applicable).
            newValue: The new value to set.

        Returns:
            Callable: A function that updates the attribute.
        """
        if attribute == DQDAttributes.GAMMA.value:
            def updater(currentValue):
                return {attribute: self._adjustGamma(np.copy(currentValue), newValue, side)}
        elif attribute == DQDAttributes.ZEEMAN.value:
            def updater(currentValue):
                return {attribute: self._adjustZeeman(np.copy(currentValue), newValue, axis, side)}
        elif attribute == DQDAttributes.MAGNETIC_FIELD.value:
            def updater(currentValue):
                return {attribute: self._adjustMagneticField(np.copy(currentValue), newValue, axis)}
        elif attribute == DQDAttributes.G_FACTOR.value:
            def updater(currentValue):
                return {attribute: self._adjustGFactor(np.copy(currentValue), newValue, axis, side)}
        elif attribute == NoAttributeParameters.SCAN_ANGLE.value:
            def updater(currentValues):
                zeemanValue, magneticFieldValue = self._adjustScanAngleDict(np.copy(currentValues[0]),
                                                                            np.copy(currentValues[1]), newValue)
                return {DQDAttributes.ZEEMAN.value: zeemanValue, DQDAttributes.MAGNETIC_FIELD.value: magneticFieldValue}
        else:
            def updater(currentValue):
                return {attribute: newValue}

        return updater

    def _adjustGamma(self, completeValue: np.ndarray, newValue: float, side: Optional[int]) -> np.ndarray:
        """
        Adjusts the gamma values for the specified side.

        Args:
            completeValue (np.ndarray): The current gamma values.
            newValue (float): The new value to set.
            side (Optional[int]): The side to adjust (0 or 1), or None for both sides.

        Returns:
            np.ndarray: The adjusted gamma values.
        """
        if side is None:
            completeValue[:, 0] = newValue
        else:
            completeValue[side, 0] = newValue
        return completeValue

    def _adjustZeeman(self, completeValue: np.ndarray, newValue: float, axis: Optional[int],
                      side: Optional[int]) -> np.ndarray:
        """
        Adjusts the Zeeman splitting values based on the specified axis and side.

        Args:
            completeValue (np.ndarray): The current Zeeman splitting values.
            newValue (float): The new value to set.
            axis (Optional[int]): The axis to adjust (or magnitude if Axis.M.value).
            side (Optional[int]): The side to adjust (0 or 1), or None for both sides.

        Returns:
            np.ndarray: The adjusted Zeeman splitting values.
        """
        if axis == Axis.M.value:  # Adjust magnitude
            if side is None:
                for s in range(completeValue.shape[0]):
                    norm = np.linalg.norm(completeValue[s])
                    direction = completeValue[s] / norm if norm != 0 else np.zeros_like(completeValue[s])
                    completeValue[s] = direction * newValue
            else:
                norm = np.linalg.norm(completeValue[side])
                direction = completeValue[side] / norm if norm != 0 else np.zeros_like(completeValue[side])
                completeValue[side] = direction * newValue
        else:
            if side is None:
                completeValue[:, axis] = newValue
            else:
                completeValue[side, axis] = newValue
        return completeValue

    def _adjustMagneticField(self, completeValue: np.ndarray, newValue: float, axis: Optional[int]) -> np.ndarray:
        """
        Adjusts the magnetic field values based on the specified axis.

        Args:
            completeValue (np.ndarray): The current magnetic field values.
            newValue (float): The new value to set.
            axis (Optional[int]): The axis to adjust (or magnitude if Axis.M.value).

        Returns:
            np.ndarray: The adjusted magnetic field values.
        """
        if axis == Axis.M.value:  # Adjust magnitude
            norm = np.linalg.norm(completeValue)
            direction = completeValue / norm if norm != 0 else np.zeros_like(completeValue)
            completeValue = direction * newValue
        else:
            completeValue[axis] = newValue
        return completeValue

    def _adjustGFactor(self, completeValue: np.ndarray, newValue: float, axis: Optional[int],
                       side: Optional[int]) -> np.ndarray:
        """
        Adjusts the g-factor values for the specified axis and side.

        Args:
            completeValue (np.ndarray): The current g-factor values.
            newValue (float): The new value to set.
            axis (Optional[int]): The axis to adjust.
            side (Optional[int]): The side to adjust (0 or 1), or None for both sides.

        Returns:
            np.ndarray: The adjusted g-factor values.
        """
        if side is None:
            completeValue[:, axis] = newValue
        else:
            completeValue[side, axis] = newValue
        return completeValue

    def _adjustScanAngleDict(self, completeMagneticField: np.ndarray, completeGFactor: np.ndarray,
                             newAngleValue: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Adjusts the Zeeman splitting values based on a new scan angle.

        Args:
            completeMagneticField (np.ndarray): The current magnetic field values.
            completeGFactor (np.ndarray): The current g-factor values.
            newAngleValue (float): The new scan angle value.

        Returns:
            np.ndarray: The adjusted Zeeman splitting values.
        """
        newAngleValue = newAngleValue * np.pi
        BModule = np.linalg.norm(completeMagneticField)
        XComponent = BModule * np.cos(newAngleValue)
        YComponent = BModule * np.sin(newAngleValue)
        BVector = np.array([XComponent, YComponent, 0])
        ZLeft = list(completeGFactor[0] @ BVector)
        ZRight = list(completeGFactor[1] @ BVector)
        completeZeeman = np.array([ZLeft, ZRight])
        return completeZeeman, BVector
