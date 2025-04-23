from typing import Dict, Any

import numpy as np

from src.DQDSystem import DQDSystem
from src.UnifiedParameters import UnifiedParameters
from src.base.DoubleQuantumDot import DQDAttributes


class DQDSystemFactory:
    """
    Factory class for creating preconfigured DQDSystem instances.

    This class provides methods to create DQDSystem instances with predefined configurations.
    Users can customize the independent arrays and plot options.
    """

    # Global dictionary for default fixed parameters
    defaultFixedParameters: Dict[str, Any] = {
        UnifiedParameters.DETUNING.value: 0.08,
        UnifiedParameters.FACTOR_OME.value: 0.0,
        UnifiedParameters.MAGNETIC_FIELD.value: [1.0, 0.0, 0.0],
    }

    @staticmethod
    def changeParameter(parameter: str, value: Any) -> None:
        """
        Changes the value of a fixed parameter in the global dictionary.

        Args:
            parameter (str): The name of the parameter to change.
            value (Any): The new value for the parameter.

        Raises:
            KeyError: If the parameter does not exist in the global dictionary.
        """
        if parameter not in DQDSystemFactory.defaultFixedParameters:
            raise KeyError(f"Parameter '{parameter}' does not exist in the default fixed parameters.")
        DQDSystemFactory.defaultFixedParameters[parameter] = value

    @staticmethod
    def ZeemanXvsZeemanY(xArray: np.ndarray, yArray: np.ndarray, plotOptions: Dict[str, Any] = None) -> DQDSystem:
        """
        Creates a DQDSystem instance for ZeemanX vs ZeemanY.

        Args:
            xArray (np.ndarray): Array for ZeemanX.
            yArray (np.ndarray): Array for ZeemanY.
            plotOptions (Dict[str, Any]): Plotting options.

        Returns:
            DQDSystem: Configured DQDSystem instance.
        """
        iterationParameters = [
            {"array": xArray, "features": DQDAttributes.ZEEMAN.value + "X"},
            {"array": yArray, "features": DQDAttributes.ZEEMAN.value + "Y"},
        ]
        fixedParameters = DQDSystemFactory.defaultFixedParameters.copy()

        dqdSystem = DQDSystem(fixedParameters, iterationParameters)
        dqdSystem.plottingInfo["options"] = plotOptions or {}
        return dqdSystem

    @staticmethod
    def ScanAngleVsMagneticField(xArray: np.ndarray, yArray: np.ndarray,
                                 plotOptions: Dict[str, Any] = None) -> DQDSystem:
        """
        Creates a DQDSystem instance for ScanAngle vs MagneticField.

        Args:
            xArray (np.ndarray): Array for ScanAngle (in units of PI).
            yArray (np.ndarray): Array for MagneticField.
            plotOptions (Dict[str, Any]): Plotting options.

        Returns:
            DQDSystem: Configured DQDSystem instance.
        """
        iterationParameters = [
            {"array": xArray, "features": UnifiedParameters.SCAN_ANGLE.value},
            {"array": yArray, "features": UnifiedParameters.MAGNETIC_FIELD.value + "M"},
        ]
        fixedParameters = DQDSystemFactory.defaultFixedParameters.copy()

        dqdSystem = DQDSystem(fixedParameters, iterationParameters)
        dqdSystem.plottingInfo["options"] = plotOptions or {}
        return dqdSystem

    @staticmethod
    def ZeemanXvsZeemanZ(xArray: np.ndarray, yArray: np.ndarray, plotOptions: Dict[str, Any] = None) -> DQDSystem:
        """
        Creates a DQDSystem instance for ZeemanX vs ZeemanZ.

        Args:
            xArray (np.ndarray): Array for ZeemanX.
            yArray (np.ndarray): Array for ZeemanZ.
            plotOptions (Dict[str, Any]): Plotting options.

        Returns:
            DQDSystem: Configured DQDSystem instance.
        """
        iterationParameters = [
            {"array": xArray, "features": DQDAttributes.ZEEMAN.value + "X"},
            {"array": yArray, "features": DQDAttributes.ZEEMAN.value + "Z"},
        ]
        fixedParameters = DQDSystemFactory.defaultFixedParameters.copy()

        dqdSystem = DQDSystem(fixedParameters, iterationParameters)
        dqdSystem.plottingInfo["options"] = plotOptions or {}
        return dqdSystem

    @staticmethod
    def ZeemanX(xArray: np.ndarray, plotOptions: Dict[str, Any] = None) -> DQDSystem:
        """
        Creates a DQDSystem instance for ZeemanX as the only independent parameter.

        Args:
            xArray (np.ndarray): Array for ZeemanX.
            plotOptions (Dict[str, Any]): Plotting options.

        Returns:
            DQDSystem: Configured DQDSystem instance.
        """
        iterationParameters = [
            {"array": xArray, "features": DQDAttributes.ZEEMAN.value + "X"},
        ]
        fixedParameters = DQDSystemFactory.defaultFixedParameters.copy()

        dqdSystem = DQDSystem(fixedParameters, iterationParameters)
        dqdSystem.plottingInfo["options"] = plotOptions or {}
        return dqdSystem

    @staticmethod
    def ZeemanY(yArray: np.ndarray, plotOptions: Dict[str, Any] = None) -> DQDSystem:
        """
        Creates a DQDSystem instance for ZeemanY as the only independent parameter.

        Args:
            yArray (np.ndarray): Array for ZeemanY.
            plotOptions (Dict[str, Any]): Plotting options.

        Returns:
            DQDSystem: Configured DQDSystem instance.
        """
        iterationParameters = [
            {"array": yArray, "features": DQDAttributes.ZEEMAN.value + "Y"},
        ]
        fixedParameters = DQDSystemFactory.defaultFixedParameters.copy()

        dqdSystem = DQDSystem(fixedParameters, iterationParameters)
        dqdSystem.plottingInfo["options"] = plotOptions or {}
        return dqdSystem

    @staticmethod
    def ZeemanZ(zArray: np.ndarray, plotOptions: Dict[str, Any] = None) -> DQDSystem:
        """
        Creates a DQDSystem instance for ZeemanZ as the only independent parameter.

        Args:
            zArray (np.ndarray): Array for ZeemanZ.
            plotOptions (Dict[str, Any]): Plotting options.

        Returns:
            DQDSystem: Configured DQDSystem instance.
        """
        iterationParameters = [
            {"array": zArray, "features": DQDAttributes.ZEEMAN.value + "Z"},
        ]
        fixedParameters = DQDSystemFactory.defaultFixedParameters.copy()

        dqdSystem = DQDSystem(fixedParameters, iterationParameters)
        dqdSystem.plottingInfo["options"] = plotOptions or {}
        return dqdSystem

    @staticmethod
    def Detuning(detuningArray: np.ndarray, plotOptions: Dict[str, Any] = None) -> DQDSystem:
        """
        Creates a DQDSystem instance for Detuning as the only independent parameter.

        Args:
            detuningArray (np.ndarray): Array for Detuning.
            plotOptions (Dict[str, Any]): Plotting options.

        Returns:
            DQDSystem: Configured DQDSystem instance.
        """
        iterationParameters = [
            {"array": detuningArray, "features": UnifiedParameters.DETUNING.value},
        ]
        fixedParameters = DQDSystemFactory.defaultFixedParameters.copy()

        dqdSystem = DQDSystem(fixedParameters, iterationParameters)
        dqdSystem.plottingInfo["options"] = plotOptions or {}
        return dqdSystem

    @staticmethod
    def ZeemanYvsZeemanZ(yArray: np.ndarray, zArray: np.ndarray, plotOptions: Dict[str, Any] = None) -> DQDSystem:
        """
        Creates a DQDSystem instance for ZeemanY vs ZeemanZ.

        Args:
            yArray (np.ndarray): Array for ZeemanY.
            zArray (np.ndarray): Array for ZeemanZ.
            plotOptions (Dict[str, Any]): Plotting options.

        Returns:
            DQDSystem: Configured DQDSystem instance.
        """
        iterationParameters = [
            {"array": yArray, "features": DQDAttributes.ZEEMAN.value + "Y"},
            {"array": zArray, "features": DQDAttributes.ZEEMAN.value + "Z"},
        ]
        fixedParameters = DQDSystemFactory.defaultFixedParameters.copy()

        dqdSystem = DQDSystem(fixedParameters, iterationParameters)
        dqdSystem.plottingInfo["options"] = plotOptions or {}
        return dqdSystem
