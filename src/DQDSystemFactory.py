from typing import Dict, Any, List

import numpy as np

from src.DQDSystem import DQDSystem
from src.PlotOptionsManager import PlotOptionsManager
from src.base.DQDParameterInterpreter import NoAttributeParameters, DQDParameterInterpreter
from src.base.DoubleQuantumDot import DQDAttributes


class DQDSystemFactory:
    defaultFixedParameters: Dict[str, Any] = {
        DQDAttributes.DETUNING.value: 0.08,
        DQDAttributes.AC_AMPLITUDE.value: 1.2,
        DQDAttributes.CHI.value: 0.1,
        DQDAttributes.TAU.value: 0.1,
        DQDAttributes.GAMMA.value: [[0.01], [0.01]],
        DQDAttributes.ZEEMAN.value: [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        DQDAttributes.G_FACTOR.value: [[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                                       [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]],
        DQDAttributes.SOC_THETA_ANGLE.value: np.pi / 2,
        DQDAttributes.SOC_PHI_ANGLE.value: 0.0,
        DQDAttributes.FACTOR_OME.value: 0.0,
        DQDAttributes.MAGNETIC_FIELD.value: [0.0, 0.0, 0.0],
    }

    _plotOptionsManager = PlotOptionsManager()
    _titleFields: List[str] = []

    @staticmethod
    def getPlotOptionsForSystem() -> Dict[str, Any]:
        return DQDSystemFactory._plotOptionsManager.getAllOptions()

    @staticmethod
    def addToPlotOptions(key: str, value: Any) -> None:
        DQDSystemFactory._plotOptionsManager.setOption(key, value)

    @staticmethod
    def resetPlotOptions() -> None:
        DQDSystemFactory._plotOptionsManager.resetOptions()

    @staticmethod
    def getTitleForSystem() -> List[str]:
        return DQDSystemFactory._titleFields.copy()

    @staticmethod
    def addToTitle(field: str) -> None:
        if field not in DQDSystemFactory._titleFields:
            DQDSystemFactory._titleFields.append(field)

    @staticmethod
    def resetTitle() -> None:
        DQDSystemFactory._titleFields.clear()

    @staticmethod
    def changeParameter(paramName: str, value: Any) -> None:
        base, _, side = DQDParameterInterpreter.parseAttributeString(paramName)
        if side == 0:
            if base not in DQDSystemFactory.defaultFixedParameters:
                DQDSystemFactory.defaultFixedParameters[base] = [None, None]
            DQDSystemFactory.defaultFixedParameters[base][0] = value
        elif side == 1:
            if base not in DQDSystemFactory.defaultFixedParameters:
                DQDSystemFactory.defaultFixedParameters[base] = [None, None]
            DQDSystemFactory.defaultFixedParameters[base][1] = value
        else:
            DQDSystemFactory.defaultFixedParameters[paramName] = value

    @staticmethod
    def create(iterationParameters: List[Dict[str, Any]]) -> DQDSystem:
        fixedParameters = DQDSystemFactory.defaultFixedParameters.copy()
        return DQDSystem(fixedParameters, iterationParameters)

    # --- Predefined systems ---

    @staticmethod
    def zeemanX(xArray: np.ndarray) -> DQDSystem:
        return DQDSystemFactory.create([
            {"array": xArray, "features": DQDAttributes.ZEEMAN.value + "X"}
        ])

    @staticmethod
    def zeemanY(xArray: np.ndarray) -> DQDSystem:
        return DQDSystemFactory.create([
            {"array": xArray, "features": DQDAttributes.ZEEMAN.value + "Y"}
        ])

    @staticmethod
    def zeemanZ(xArray: np.ndarray) -> DQDSystem:
        return DQDSystemFactory.create([
            {"array": xArray, "features": DQDAttributes.ZEEMAN.value + "Z"}
        ])

    @staticmethod
    def detuning(xArray: np.ndarray) -> DQDSystem:
        return DQDSystemFactory.create([
            {"array": xArray, "features": DQDAttributes.DETUNING.value}
        ])

    @staticmethod
    def detuningvsZeemanX(xArray: np.ndarray, yArray: np.ndarray) -> DQDSystem:
        return DQDSystemFactory.create([
            {"array": xArray, "features": DQDAttributes.DETUNING.value},
            {"array": yArray, "features": DQDAttributes.ZEEMAN.value + "X"},
        ])

    @staticmethod
    def detuningvsZeemanY(xArray: np.ndarray, yArray: np.ndarray) -> DQDSystem:
        return DQDSystemFactory.create([
            {"array": xArray, "features": DQDAttributes.DETUNING.value},
            {"array": yArray, "features": DQDAttributes.ZEEMAN.value + "Y"},
        ])

    @staticmethod
    def detuningvsZeemanZ(xArray: np.ndarray, yArray: np.ndarray) -> DQDSystem:
        return DQDSystemFactory.create([
            {"array": xArray, "features": DQDAttributes.DETUNING.value},
            {"array": yArray, "features": DQDAttributes.ZEEMAN.value + "Z"},
        ])

    @staticmethod
    def zeemanXvsZeemanY(xArray: np.ndarray, yArray: np.ndarray) -> DQDSystem:
        return DQDSystemFactory.create([
            {"array": xArray, "features": DQDAttributes.ZEEMAN.value + "X"},
            {"array": yArray, "features": DQDAttributes.ZEEMAN.value + "Y"},
        ])

    @staticmethod
    def zeemanXvsZeemanZ(xArray: np.ndarray, yArray: np.ndarray) -> DQDSystem:
        return DQDSystemFactory.create([
            {"array": xArray, "features": DQDAttributes.ZEEMAN.value + "X"},
            {"array": yArray, "features": DQDAttributes.ZEEMAN.value + "Z"},
        ])

    @staticmethod
    def zeemanYvsZeemanZ(xArray: np.ndarray, yArray: np.ndarray) -> DQDSystem:
        return DQDSystemFactory.create([
            {"array": xArray, "features": DQDAttributes.ZEEMAN.value + "Y"},
            {"array": yArray, "features": DQDAttributes.ZEEMAN.value + "Z"},
        ])

    @staticmethod
    def magneticFieldXvsMagneticFieldY(xArray: np.ndarray, yArray: np.ndarray) -> DQDSystem:
        return DQDSystemFactory.create([
            {"array": xArray, "features": DQDAttributes.MAGNETIC_FIELD.value + "X"},
            {"array": yArray, "features": DQDAttributes.MAGNETIC_FIELD.value + "Y"},
        ])

    @staticmethod
    def magneticFieldXvsMagneticFieldZ(xArray: np.ndarray, yArray: np.ndarray) -> DQDSystem:
        return DQDSystemFactory.create([
            {"array": xArray, "features": DQDAttributes.MAGNETIC_FIELD.value + "X"},
            {"array": yArray, "features": DQDAttributes.MAGNETIC_FIELD.value + "Z"},
        ])

    @staticmethod
    def magneticFieldYvsMagneticFieldZ(xArray: np.ndarray, yArray: np.ndarray) -> DQDSystem:
        return DQDSystemFactory.create([
            {"array": xArray, "features": DQDAttributes.MAGNETIC_FIELD.value + "Y"},
            {"array": yArray, "features": DQDAttributes.MAGNETIC_FIELD.value + "Z"},
        ])

    @staticmethod
    def scanAngleVsMagneticFieldModule(xArray: np.ndarray, yArray: np.ndarray) -> DQDSystem:
        fixedParameters = DQDSystemFactory.defaultFixedParameters.copy()
        fixedParameters[DQDAttributes.MAGNETIC_FIELD.value] = [1.0, 1.0, 0.0]

        return DQDSystem(fixedParameters, [
            {"array": xArray, "features": NoAttributeParameters.SCAN_ANGLE.value},
            {"array": yArray, "features": DQDAttributes.MAGNETIC_FIELD.value + "M"}
        ])
