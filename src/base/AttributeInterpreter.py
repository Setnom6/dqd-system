from typing import Dict, Any, Tuple, List, Optional, Callable
import re
from src.base.DoubleQuantumDot import DQDAttributes
import numpy as np


class AttributeInterpreter:
    LATEX_SYMBOLS = {
        "current": r"$I/ e \Gamma$",
        "polarity": r"$P$",
        "detuning": r"$\delta$",
        "zeeman": r"$Z$",
        "acAmplitude": r"$A_{ac}$",
        "tau": r"$\tau$",
        "chi": r"$\chi$",
        "gamma": r"$\gamma$",
        "acFrequency": r"$\omega$",
        "magneticField": r"$B$",
        "gFactor": r"$g$",
        "X": r"$_{X,",
        "Y": r"$_{Y,",
        "Z": r"$_{Z,",
        "M": r"$_{mod}$",
        "Left": r"\text{Left}}$",
        "Right": r"\text{Right}$",
        "alphaThetaAngle": r"$\theta_{SO}$",
        "alphaPhiAngle": r"$\phi_{SO}$",
    }

    def __init__(self, fixedParameters: Dict[str, Any], iterationParameters: List[Dict[str, Any]]):
        self.fixedParameters = self.processFixedParameters(fixedParameters)
        self.iterationArrays, self.iterationParametersFeatures = self.processIterationParameters(iterationParameters)

    def processFixedParameters(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        fixedParamsWithoutLeftRight = self.processLeftRightAttributes(parameters)
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

    def processLeftRightAttributes(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        adjustedParams = {}
        tempStorage = {}

        for key, value in parameters.items():
            match = re.match(r"(.+)(Left|Right)$", key)
            if match:
                baseKey, position = match.groups()
                if baseKey not in tempStorage:
                    tempStorage[baseKey] = {}
                tempStorage[baseKey][position] = value
            else:
                adjustedParams[key] = value

        for baseKey, positions in tempStorage.items():
            leftValue = positions.get("Left")
            rightValue = positions.get("Right")
            if leftValue is not None and rightValue is not None:
                adjustedParams[baseKey] = [leftValue, rightValue]
            elif leftValue is not None:
                adjustedParams[baseKey] = [leftValue]
            elif rightValue is not None:
                adjustedParams[baseKey] = [rightValue]

        return adjustedParams

    def processIterationParameters(self, iterationParameters: List[Dict[str, Any]]) -> Tuple[List[np.ndarray], List[str]]:
        iterationArrays = []
        iterationParametersFeatures = []

        for parameterDict in iterationParameters:
            iterationParametersFeatures.append(parameterDict["features"])
            iterationArrays.append(parameterDict["array"])

        return iterationArrays, iterationParametersFeatures

    def parseAttributeString(self, attributeString: str) -> Tuple[str, int, int]:
        side = None
        axis = None

        if attributeString.endswith("Left"):
            side = 0
            attributeString = attributeString[:-4]
        elif attributeString.endswith("Right"):
            side = 1
            attributeString = attributeString[:-5]

        axisMapping = {"X": 0, "Y": 1, "Z": 2, "M": 3}
        if attributeString[-1] in axisMapping:
            axis = axisMapping[attributeString[-1]]
            attributeString = attributeString[:-1]

        name = attributeString
        return name, axis, side
    
    def getIndependentArrays(self) -> List[np.ndarray]:
        """
        Returns the independent arrays used for iteration.

        Returns:
            List[np.ndarray]: A list of independent arrays.
        """
        return self.iterationArrays
    
    def getSimulationName(self) -> str:
        """
        Generates a simulation name based on the fixed parameters.

        Returns:
            str: A formatted string representing the simulation name.
        """
        # Concatenate the names of the iteration parameters
        return "_".join([features for features in self.iterationParametersFeatures])
    

    def formatLatexLabel(self, feature: str) -> str:
        """
        Formats a LaTeX label for a given feature, including axis and side information if present.

        Args:
            feature (str): The feature name in "features" format.

        Returns:
            str: The LaTeX-formatted label.
        """
        # Check if the feature matches a known parameter
        if feature in self.LATEX_SYMBOLS:
            # Simple case: feature is directly in LATEX_SYMBOLS
            return self.LATEX_SYMBOLS[feature]
        else:
            # Complex case: feature includes axis and side (e.g., "zeemanXLeft")
            baseFeature = ''.join([char for char in feature if not char.isupper() and char not in ["Left", "Right"]])
            axis = next((i for i, axis in enumerate(["X", "Y", "Z"]) if axis in feature), None)
            side = next((i for i, side in enumerate(["Left", "Right"]) if side in feature), None)

            if baseFeature in self.LATEX_SYMBOLS and axis is not None and side is not None:
                latexSymbol = self.LATEX_SYMBOLS[baseFeature]
                axisLabel = ["X", "Y", "Z"][axis]
                sideLabel = ["Left", "Right"][side]
                return rf"{latexSymbol}_{{{axisLabel},{sideLabel}}}"
            else:
                # If the feature is not recognized, return it as-is
                return feature

    def getLabels(self) -> List[str]:
        """
        Generates LaTeX labels for each independent variable based on their features,
        axis, and side information.

        Returns:
            List[str]: A list of LaTeX-formatted labels for the independent variables.
        """
        labels = []
        for features in self.iterationParametersFeatures:
            labels.append(self.formatLatexLabel(features))
        return labels

    def getDependentLabels(self) -> List[str]:
        return [self.formatLatexLabel("current"), self.formatLatexLabel("polarity")]

    def getTitle(self, titleOptions: List[str]) -> Dict[str, str]:
        """
        Generates a LaTeX-formatted title string and instructions for filling placeholders.

        Args:
            titleOptions (List[str]): A list of variable names in "features" format.

        Returns:
            Dict[str, Any]: A dictionary containing the formatted title string and instructions
                            for filling placeholders with values from dqdObject.
        """
        titleParts = []
        placeholders = []

        for feature in titleOptions:
            formattedLabel = self.formatLatexLabel(feature)

            # Parse the feature to extract name, axis, and side
            name, axis, side = self.parseAttributeString(feature)

            if "{}" in formattedLabel:
                titleParts.append(f"{formattedLabel} = {{}}")
                placeholders.append(feature)
            else:
                titleParts.append(formattedLabel)

        titleStr = ", ".join(titleParts)
        return {"title": titleStr, "placeholders": placeholders}


    def getUpdateFunctions(self, *indices: int) -> List[Tuple[Callable, str]]:
        """
        Generates updater functions for all iteration parameters based on the provided indices.

        Args:
            indices (int): Indices corresponding to the current point in the simulation grid.

        Returns:
            List[Tuple[Callable, str]]: A list of tuples containing updater functions and their corresponding attribute names.
        """
        updateFunctions = []
        for idx, index in enumerate(indices):
            if index >= len(self.iterationArrays[idx]):
                raise IndexError(f"Index {index} is out of bounds for array of size {len(self.iterationArrays[idx])}")

            parameterName, parameterAxis, parameterSide = self.parseAttributeString(self.iterationParametersFeatures[idx])
            newValue = self.iterationArrays[idx][index]
            updater = self._getUpdaterFunction(
                parameterName,
                parameterAxis,
                parameterSide,
                newValue
            )
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
        """
        Returns the lengths of all iteration arrays.

        Returns:
            List[int]: A list of lengths for each iteration array.
        """
        return [len(array) for array in self.iterationArrays]

    def _adjustGamma(self, completeValue: np.ndarray, newValue: float, side: Optional[int]) -> np.ndarray:
        if side is None:
            completeValue[:, 0] = newValue
        else:
            completeValue[side, 0] = newValue
        return completeValue

    def _adjustZeeman(self, completeValue: np.ndarray, newValue: float, axis: Optional[int],
                      side: Optional[int]) -> np.ndarray:
        if axis == 3:  # Adjust magnitude
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
        if axis == 3:  # Adjust magnitude
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