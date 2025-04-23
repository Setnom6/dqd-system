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


class NoAttributeParameters(Enum):
    SCAN_ANGLE = "scanAngle"


class AnnotationsType(Enum):
    EXPECTED_MODULE_RESONANCES = 'ExpectedModuleResonances'


@dataclass
class Parameter:
    name: str
    value: Any
    axis: Optional[int] = None
    side: Optional[int] = None


class AttributeInterpreter:
    """
    Interprets and processes attributes for a Double Quantum Dot (DQD) system.

    This class handles fixed and iteration parameters, provides utilities for parsing and formatting attributes,
    and generates update functions for simulation parameters.

    Attributes:
        LATEX_SYMBOLS (Dict[str, str]): Mapping of attributes to their LaTeX representations.
        AXIS_SYMBOLS (List[str]): List of axis names (X, Y, Z, M).
        SIDE_SYMBOLS (List[str]): List of side names (LEFT, RIGHT).
        DIVIDED_BY_OMEGA (List[str]): Attributes divided by the driving frequency.
        DIVIDED_BY_MUB (List[str]): Attributes divided by the Bohr magneton.
    """
    LATEX_SYMBOLS = {
        DQDAttributes.SUM_CURRENT.value: r"$I/ e \Gamma$",
        DQDAttributes.POLARITY.value: r"$P$",
        DQDAttributes.DETUNING.value: r"$\delta$",
        DQDAttributes.ZEEMAN.value: r"$Z$",
        DQDAttributes.AC_AMPLITUDE.value: r"$A_{{ac}}$",
        DQDAttributes.TAU.value: r"$\tau$",
        DQDAttributes.CHI.value: r"$\chi$",
        DQDAttributes.GAMMA.value: r"$\gamma$",
        DQDAttributes.MAGNETIC_FIELD.value: r"$B$",
        DQDAttributes.G_FACTOR.value: r"$g$",
        DQDAttributes.FACTOR_OME.value: r"$k_{{OME}}$",
        DQDAttributes.ALPHA_THETA_ANGLE.value: r"$\theta_{{SO}}$",
        DQDAttributes.ALPHA_PHI_ANGLE.value: r"$\phi_{{SO}}$",
        "acFrequency": r"$\omega$",
        "muB": r"$\mu_B$",
        NoAttributeParameters.SCAN_ANGLE.value: r"$\theta_{{XY}} / \pi$",
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
        """
        Initializes the AttributeInterpreter with fixed and iteration parameters.

        Args:
            fixedParameters (Dict[str, Any]): Fixed parameters for the DQD system.
            iterationParameters (List[Dict[str, Any]]): Parameters to iterate over during the simulation.
        """
        self.fixedParameters = self._processFixedParameters(fixedParameters)
        self.iterationArrays, self.iterationParameterFeatures = self._processIterationParameters(iterationParameters)
        self._iterationParametersOriginalFormat = iterationParameters

    def returnOriginalterationParameters(self) -> List[Dict[str, Any]]:
        """
        Returns the original iteration parameters in their initial format.

        Returns:
            List[Dict[str, Any]]: The original iteration parameters.
        """
        return self._iterationParametersOriginalFormat

    def _processFixedParameters(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Processes and adjusts fixed parameters for the DQD system.

        Args:
            parameters (Dict[str, Any]): Fixed parameters to process.

        Returns:
            Dict[str, Any]: Adjusted fixed parameters.
        """
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
            else:
                adjustedParams[key] = value

        return adjustedParams

    def _processLeftRightAttributes(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Processes attributes with "Left" and "Right" suffixes into a unified format.

        Args:
            parameters (Dict[str, Any]): Parameters to process.

        Returns:
            Dict[str, Any]: Adjusted parameters with unified "Left" and "Right" attributes.
        """
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
        """
        Processes iteration parameters into arrays and features.

        Args:
            iterationParameters (List[Dict[str, Any]]): Iteration parameters to process.

        Returns:
            Tuple[List[np.ndarray], List[str]]: Arrays of iteration parameters and their features.
        """
        iterationArrays = []
        iterationParameterFeatures = []

        for parameterDict in iterationParameters:
            iterationParameterFeatures.append(parameterDict["features"])
            if "array" in parameterDict:
                iterationArrays.append(parameterDict["array"])
            else:
                iterationArrays.append(np.array([]))

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
        """
        Returns the arrays of independent variables for the simulation.

        Returns:
            List[np.ndarray]: Arrays of independent variables.
        """
        return self.iterationArrays

    def getSimulationName(self) -> str:
        """
        Generates a simulation name based on iteration parameter features.

        Returns:
            str: The simulation name.
        """
        return "_".join(self.iterationParameterFeatures)

    def formatLatexLabel(self, feature: str) -> str:
        """
        Formats a feature into a LaTeX-compatible label.

        Args:
            feature (str): The feature to format.

        Returns:
            str: The LaTeX-compatible label.
        """
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
        """
        Returns LaTeX labels for iteration parameter features.

        Returns:
            List[str]: LaTeX labels for iteration parameter features.
        """
        return [self.formatLatexLabel(feature) for feature in self.iterationParameterFeatures]

    def getDependentLabels(self) -> List[str]:
        """
        Returns LaTeX labels for dependent variables.

        Returns:
            List[str]: LaTeX labels for dependent variables.
        """
        return [self.formatLatexLabel(DQDAttributes.SUM_CURRENT.value),
                self.formatLatexLabel(DQDAttributes.POLARITY.value)]

    def getTitle(self, titleOptions: List[str]) -> Dict[str, str]:
        """
        Generates a title string and placeholders for a plot.

        Args:
            titleOptions (List[str]): Features to include in the title.

        Returns:
            Dict[str, str]: A dictionary containing the title string and placeholders.
        """
        titleParts = []
        placeholders = []

        for feature in titleOptions:
            formattedLabel = self.formatLatexLabel(feature)
            name, _, _ = self.parseAttributeString(feature)
            if name in [attr.value for attr in DQDAttributes]:
                titlePart = f"{formattedLabel} = {{}}"
                placeholders.append(feature)
            else:
                titlePart = formattedLabel
            if name in DQDAttributes.ALPHA_PHI_ANGLE.value or name in DQDAttributes.ALPHA_THETA_ANGLE.value:
                titlePart += "º"
            titleParts.append(titlePart)

        return {"title": ", ".join(titleParts), "placeholders": placeholders}

    def lenIterationArrays(self) -> List[int]:
        """
        Returns the lengths of the iteration arrays.

        Returns:
            List[int]: A list of lengths for each iteration array.
        """
        return [len(array) for array in self.iterationArrays]

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
                raise IndexError(f"Index {index} is out of bounds for array of size {len(self.iterationArrays[idx])}")

            parameterName, parameterAxis, parameterSide = self.parseAttributeString(
                self.iterationParameterFeatures[idx])
            newValue = self.iterationArrays[idx][index]
            updater = self._getUpdaterFunction(parameterName, parameterAxis, parameterSide, newValue)
            if parameterName == NoAttributeParameters.SCAN_ANGLE.value:
                parameterName = tuple((DQDAttributes.MAGNETIC_FIELD.value, DQDAttributes.G_FACTOR.value))
            updateFunctions.append((updater, parameterName))
        return updateFunctions

    def _getUpdaterFunction(self, attribute: str, axis: Optional[int], side: Optional[int], newValue) -> Callable:
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
                return {DQDAttributes.ZEEMAN.value: self._adjustScanAngleDict(np.copy(currentValues[0]),
                                                                              np.copy(currentValues[1]), newValue)}
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
                             newAngleValue: float) -> np.ndarray:
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
        return completeZeeman

    def decideWhichAnnotations(self) -> str:
        """
        Decides the type of annotations to use based on the iterationParameterFeatures.

        Returns:
            str: The type of annotations to use, or an empty string if no match is found.
        """
        parsedParameters = [self.parseAttributeString(feature)[0] for feature in self.iterationParameterFeatures]

        # Check for specific combinations of parameters
        if (
            NoAttributeParameters.SCAN_ANGLE.value in parsedParameters and
            DQDAttributes.MAGNETIC_FIELD.value in parsedParameters
        ):
            return AnnotationsType.EXPECTED_MODULE_RESONANCES.value

        return ""
