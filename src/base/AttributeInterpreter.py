import re
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from typing import Dict, Any, Tuple, List, Optional, Callable

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


@dataclass
class Parameter:
    name: str
    value: Any
    axis: Optional[int] = None
    side: Optional[int] = None


class AttributeInterpreter:
    LATEX_SYMBOLS = {
        DQDAttributes.SUM_CURRENT.value: r"$I/ e \Gamma$",
        DQDAttributes.POLARITY.value: r"$P$",
        DQDAttributes.DETUNING.value: r"$\delta$",
        DQDAttributes.ZEEMAN.value: r"$Z$",
        DQDAttributes.AC_AMPLITUDE.value: r"$A_{ac}$",
        DQDAttributes.TAU.value: r"$\tau$",
        DQDAttributes.CHI.value: r"$\chi$",
        DQDAttributes.GAMMA.value: r"$\gamma$",
        DQDAttributes.MAGNETIC_FIELD.value: r"$B$",
        DQDAttributes.G_FACTOR.value: r"$g$",
        DQDAttributes.ALPHA_THETA_ANGLE.value: r"$\theta_{SO}$",
        DQDAttributes.ALPHA_PHI_ANGLE.value: r"$\phi_{SO}$",
        "acFrequency": r"$\omega$",
        "muB": r"$\mu_B$",
    }

    AXIS_SYMBOLS = [axis.name for axis in Axis]
    SIDE_SYMBOLS = [side.name for side in Side]

    DIVIDED_BY_OMEGA = [
        DQDAttributes.DETUNING.value,
        DQDAttributes.ZEEMAN.value,
        DQDAttributes.AC_AMPLITUDE.value,
        DQDAttributes.TAU.value,
        DQDAttributes.GAMMA.value,
        DQDAttributes.MAGNETIC_FIELD.value,
    ]

    DIVIDED_BY_MUB = [DQDAttributes.MAGNETIC_FIELD.value]

    def __init__(self, fixedParameters: Dict[str, Any], iterationParameters: List[Dict[str, Any]]):
        self.fixedParameters = self._processFixedParameters(fixedParameters)
        self.iterationArrays, self.iterationParameterFeatures = self._processIterationParameters(iterationParameters)

    def _processFixedParameters(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        fixedParamsWithoutLeftRight = self._processLeftRightAttributes(parameters)
        adjustedParams = {}

        for key, value in fixedParamsWithoutLeftRight.items():
            if key == DQDAttributes.GAMMA.value:
                adjustedParams[key] = np.array(value).reshape((2, 1))
            elif key == DQDAttributes.ZEEMAN.value:
                adjustedParams[key] = np.array(value).reshape((2, 3))
            elif key == DQDAttributes.MAGNETIC_FIELD.value:
                adjustedParams[key] = np.array(value).reshape((3,))
            elif key == DQDAttributes.G_FACTOR.value:
                value = np.array(value)
                if value.shape == (2, 3):
                    gFactor = np.zeros((2, 3, 3))
                    for i in range(2):
                        np.fill_diagonal(gFactor[i], value[i])
                    adjustedParams[key] = gFactor
                elif value.shape == (2, 3, 3):
                    adjustedParams[key] = np.array(value).reshape((2, 3, 3))
                else:
                    raise ValueError(f"Invalid shape for G_FACTOR: {value.shape}. Expected (2, 3) or (2, 3, 3).")
            elif key == DQDAttributes.FACTOR_OME.value:
                raise ValueError("OME term cannot be determined a priori.")
            else:
                adjustedParams[key] = value

        return adjustedParams

    def _processLeftRightAttributes(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        from collections import defaultdict

        tempStorage = defaultdict(dict)
        adjustedParams = {}

        for key, value in parameters.items():
            match = re.match(r"(.+)(Left|Right)$", key)
            if match:
                baseKey, position = match.groups()
                tempStorage[baseKey][position] = value
            else:
                adjustedParams[key] = value

        for baseKey, positions in tempStorage.items():
            adjustedParams[baseKey] = [
                positions.get("Left"),
                positions.get("Right"),
            ]

        return adjustedParams

    def _processIterationParameters(self, iterationParameters: List[Dict[str, Any]]) -> Tuple[
        List[np.ndarray], List[str]]:
        iterationArrays = []
        iterationParameterFeatures = []

        for parameterDict in iterationParameters:
            iterationParameterFeatures.append(parameterDict["features"])
            iterationArrays.append(parameterDict["array"])

        return iterationArrays, iterationParameterFeatures

    @lru_cache(maxsize=None)
    def parseAttributeString(self, attributeString: str) -> Tuple[str, int, int]:
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

    def getIndependentArrays(self) -> List[np.ndarray]:
        return self.iterationArrays

    def getSimulationName(self) -> str:
        return "_".join(self.iterationParameterFeatures)

    def formatLatexLabel(self, feature: str) -> str:
        name, axis, side = self.parseAttributeString(feature)
        latexSymbol = self.LATEX_SYMBOLS.get(name, feature)

        if axis is not None:
            latexSymbol += r"$_{{" + self.AXIS_SYMBOLS[axis] + "}}$"
        if side is not None:
            latexSymbol += r"$_{{" + self.SIDE_SYMBOLS[side] + "}}$"

        if name in self.DIVIDED_BY_OMEGA and name in self.DIVIDED_BY_MUB:
            latexSymbol += "/" + self.LATEX_SYMBOLS["acFrequency"] + self.LATEX_SYMBOLS["muB"]
        elif name in self.DIVIDED_BY_MUB:
            latexSymbol += "/" + self.LATEX_SYMBOLS["muB"]
        elif name in self.DIVIDED_BY_OMEGA:
            latexSymbol += "/" + self.LATEX_SYMBOLS["acFrequency"]

        return latexSymbol

    def getLabels(self) -> List[str]:
        return [self.formatLatexLabel(feature) for feature in self.iterationParameterFeatures]

    def getDependentLabels(self) -> List[str]:
        return [self.formatLatexLabel(DQDAttributes.SUM_CURRENT.value),
                self.formatLatexLabel(DQDAttributes.POLARITY.value)]

    def getTitle(self, titleOptions: List[str]) -> Dict[str, str]:
        titleParts = []
        placeholders = []

        for feature in titleOptions:
            formattedLabel = self.formatLatexLabel(feature)
            name, _, _ = self.parseAttributeString(feature)
            if name in [attr.value for attr in DQDAttributes]:
                titleParts.append(f"{formattedLabel} = {{}}")
                placeholders.append(feature)
            else:
                titleParts.append(formattedLabel)

        return {"title": ", ".join(titleParts), "placeholders": placeholders}

    def getUpdateFunctions(self, *indices: int) -> List[Tuple[Callable, str]]:
        updateFunctions = []
        for idx, index in enumerate(indices):
            if index >= len(self.iterationArrays[idx]):
                raise IndexError(f"Index {index} is out of bounds for array of size {len(self.iterationArrays[idx])}")

            parameterName, parameterAxis, parameterSide = self.parseAttributeString(
                self.iterationParameterFeatures[idx])
            newValue = self.iterationArrays[idx][index]
            updater = self._getUpdaterFunction(parameterName, parameterAxis, parameterSide, newValue)
            updateFunctions.append((updater, parameterName))
        return updateFunctions

    def _getUpdaterFunction(self, attribute: str, axis: Optional[int], side: Optional[int], newValue):
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
        else:
            def updater(currentValue):
                return {attribute: newValue}

        return updater

    def lenIterationArrays(self) -> List[int]:
        return [len(array) for array in self.iterationArrays]

    def _adjustGamma(self, completeValue: np.ndarray, newValue: float, side: Optional[int]) -> np.ndarray:
        if side is None:
            completeValue[:, 0] = newValue
        else:
            completeValue[side, 0] = newValue
        return completeValue

    def _adjustZeeman(self, completeValue: np.ndarray, newValue: float, axis: Optional[int],
                      side: Optional[int]) -> np.ndarray:
        if axis == Axis.M.value:  # Adjust magnitude
            if side is None:
                for s in range(2):
                    direction = completeValue[s] / np.linalg.norm(completeValue[s])
                    completeValue[s] = direction * newValue
            else:
                direction = completeValue[side] / np.linalg.norm(completeValue[side])
                completeValue[side] = direction * newValue
        else:
            if side is None:
                completeValue[:, axis] = newValue
            else:
                completeValue[side, axis] = newValue
        return completeValue

    def _adjustMagneticField(self, completeValue: np.ndarray, newValue: float, axis: Optional[int]) -> np.ndarray:
        if axis == Axis.M.value:  # Adjust magnitude
            direction = completeValue / np.linalg.norm(completeValue)
            completeValue = direction * newValue
        else:
            completeValue[axis] = newValue
        return completeValue

    def _adjustGFactor(self, completeValue: np.ndarray, newValue: float, axis: Optional[int],
                       side: Optional[int]) -> np.ndarray:
        if side is None:
            completeValue[:, axis] = newValue
        else:
            completeValue[side, axis] = newValue
        return completeValue
