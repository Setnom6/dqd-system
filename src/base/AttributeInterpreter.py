from typing import Dict, Any, Tuple, List, Optional, Callable
import re
from src.base.DoubleQuantumDot import DQDAttributes
import numpy as np


class AttributeInterpreter:

    def __init__(self, fixedParameters: Dict[str, Any], iterationParameters: List[Dict[str, Any]]):
        self.fixedParameters = self.processFixedParameters(fixedParameters)
        self.iterationArrays, self.iterationParametersInfo = self.processIterationParameters(iterationParameters)

    def processFixedParameters(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        fixedParamsWithoutLeftRight = self.processLeftRightAttributes(parameters)
        adjustedParams = {}

        for key, value in fixedParamsWithoutLeftRight.items():
            if key == DQDAttributes.GAMMA.value:
                # Ensure gamma is a (2, 1) array
                adjustedParams[key] = np.array(value).reshape((2, 1))

            elif key == DQDAttributes.ZEEMAN.value:
                # Ensure zeeman is a (2, 3) array
                adjustedParams[key] = np.array(value).reshape((2, 3))

            elif key == DQDAttributes.MAGNETIC_FIELD.value:
                # Ensure magneticField is a (3,) array
                adjustedParams[key] = np.array(value).reshape((3,))

            elif key == DQDAttributes.G_FACTOR.value:
                # Ensure gFactor is a (2, 3, 3) array
                value = np.array(value)
                if value.shape == (2, 3):  # Case where it's two lists of 3 elements
                    gFactor = np.zeros((2, 3, 3))
                    for i in range(2):
                        np.fill_diagonal(gFactor[i], value[i])
                    adjustedParams[key] = gFactor
                elif value.shape == (2, 3, 3):  # Case where it's already 3x3 matrices
                    adjustedParams[key] = np.array(value).reshape((2, 3, 3))
                else:
                    raise ValueError(f"Invalid shape for G_FACTOR: {value.shape}. Expected (2, 3) or (2, 3, 3).")

            elif key == DQDAttributes.FACTOR_OME.value:
                raise ValueError("OME term cannot be determined a priori.")

            else:
                # For scalar values, just pass them directly
                adjustedParams[key] = value

            return adjustedParams

    def processLeftRightAttributes(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Processes custom parameters containing 'Left' or 'Right' in their keys.
        Combines the corresponding values into a single key without 'Left' or 'Right'.

        Args:
            parameters (Dict[str, Any]): Dictionary of parameters provided by the user.

        Returns:
            Dict[str, Any]: Dictionary with adjusted parameters.
        """
        adjustedParams = {}
        tempStorage = {}

        # Iterate through all keys in the dictionary
        for key, value in parameters.items():
            # Look for keys containing 'Left' or 'Right'
            match = re.match(r"(.+)(Left|Right)$", key)
            if match:
                baseKey, position = match.groups()
                if baseKey not in tempStorage:
                    tempStorage[baseKey] = {}
                tempStorage[baseKey][position] = value
            else:
                # If it doesn't contain 'Left' or 'Right', copy directly
                adjustedParams[key] = value

        # Combine 'Left' and 'Right' keys into a single key
        for baseKey, positions in tempStorage.items():
            leftValue = positions.get("Left")
            rightValue = positions.get("Right")
            if leftValue is not None and rightValue is not None:
                # Combine the values into a list or array
                adjustedParams[baseKey] = [leftValue, rightValue]
            elif leftValue is not None:
                # If only 'Left' exists, use it as the sole value
                adjustedParams[baseKey] = [leftValue]
            elif rightValue is not None:
                # If only 'Right' exists, use it as the sole value
                adjustedParams[baseKey] = [rightValue]

        return adjustedParams

    def processIterationParameters(self, iterationParameters: List[Dict[str, Any]]) -> Tuple[List[np.ndarray], List[Dict[str, Any]]]:
        allParameters = []
        iterationArrays = []
        for parameterDict in iterationParameters:
            parameterInfo = {}

            parameterInfo["name"], parameterInfo["axis"], parameterInfo["side"] = self.parseAttributeString(
                parameterDict["features"])

            parameterInfo["label"] = parameterDict["label"]

            allParameters.append(parameterInfo)
            iterationArrays.append(parameterDict["array"])

        return iterationArrays, allParameters

    def parseAttributeString(self, attributeString: str) -> Tuple[str, int, int]:
        """
        Parses a string with the structure "attributeName" + "Axis" + "Side".

        Args:
            attributeString (str): The input string to parse.

        Returns:
            Tuple[str, int, int]: A tuple containing:
                - name (str): The attribute name.
                - axis (int): The axis index (0 for "X", 1 for "Y", 2 for "Z", 3 for "M").
                - side (int): The side index (0 for "Left", 1 for "Right", None if not present).
        """
        # Initialize variables
        side = None
        axis = None

        # Check for "Left" or "Right" at the end of the string
        if attributeString.endswith("Left"):
            side = 0
            attributeString = attributeString[:-4]  # Remove "Left"
        elif attributeString.endswith("Right"):
            side = 1
            attributeString = attributeString[:-5]  # Remove "Right"

        # Check for axis in the last character
        axisMapping = {"X": 0, "Y": 1, "Z": 2, "M": 3}
        if attributeString[-1] in axisMapping:
            axis = axisMapping[attributeString[-1]]
            attributeString = attributeString[:-1]  # Remove the axis character

        # The remaining string is the attribute name
        name = attributeString

        return name, axis, side

    def xName(self) -> str:
        return self.iterationParametersInfo[0]["name"]

    def yName(self) -> str:
        return self.iterationParametersInfo[1]["name"]

    def xAxis(self) -> int:
        return self.iterationParametersInfo[0]["axis"]

    def yAxis(self) -> int:
        return self.iterationParametersInfo[1]["axis"]

    def xSide(self) -> int:
        return self.iterationParametersInfo[0]["side"]

    def ySide(self) -> int:
        return self.iterationParametersInfo[1]["side"]

    def lenX(self) -> int:
        return len(self.iterationParametersInfo[0])

    def lenY(self) -> int:
        return len(self.iterationParametersInfo[1])


    def getUpdateFunctions(self, i: int, j: int) -> List[Callable]:
        xNewValue = self.iterationArrays[0][i]
        yNewValue = self.iterationArrays[0][j]
        return [self._getUpdaterFunction(self.xName(), self.xAxis(), self.xSide(), xNewValue),
                self._getUpdaterFunction(self.yName(), self.yAxis(), self.ySide(), yNewValue)]

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