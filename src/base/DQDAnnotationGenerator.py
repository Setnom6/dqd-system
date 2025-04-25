from itertools import product
from typing import Dict, Any, List, Union

import numpy as np

from src.base.DQDParameterInterpreter import Axis, DQDParameterInterpreter
from src.base.DoubleQuantumDot import DQDAttributes, DoubleQuantumDot


class DQDAnnotationGenerator:
    def __init__(self, dqdObject: DoubleQuantumDot, iterationFeatures: Union[str, List[str]]):
        self.dqdObject = dqdObject
        self.iterationFeatures = (
            iterationFeatures if isinstance(iterationFeatures, list)
            else iterationFeatures.split("_")
        )

    def generateAnnotations(self) -> List[Dict[str, Any]]:
        annotationType = self._decideAnnotationType()
        if annotationType == "ExpectedModuleResonances":
            return self._expectedModuleResonances()
        elif annotationType == "ExpectedGTensorResonances":
            return self._expectedGTensorResonances()
        return []

    def _decideAnnotationType(self) -> str:
        if (
                "scanAngle" in self.iterationFeatures and
                "magneticFieldM" in self.iterationFeatures
        ):
            return "ExpectedModuleResonances"

        pairs = [("magneticFieldX", "magneticFieldY"),
                 ("magneticFieldX", "magneticFieldZ"),
                 ("magneticFieldY", "magneticFieldZ")]

        for a, b in pairs:
            if a in self.iterationFeatures and b in self.iterationFeatures:
                return "ExpectedGTensorResonances"

        return ""

    def _expectedModuleResonances(self) -> List[Dict[str, Any]]:
        detuning = self.dqdObject.getAttributeValue(DQDAttributes.DETUNING.value)
        nValues = range(0, 2)
        annotations = []

        def referenceLines(d):
            return [
                (lambda n: n + d, 'red', 'solid'),
                (lambda n: n - d, 'blue', 'dashed'),
                (lambda n: n + d, 'red', 'dashed'),
                (lambda n: n - d, 'blue', 'dotted'),
                (lambda n: n + d, 'red', 'dotted'),
            ]

        for n in nValues:
            for func, color, style in referenceLines(detuning):
                annotations.append({
                    "type": "line",
                    "data": {"y": func(n)},
                    "style": {"color": color, "linestyle": style, "linewidth": 1},
                    "axis": 1
                })
        return annotations

    def _expectedGTensorResonances(self) -> List[Dict[str, Any]]:
        detuning = self.dqdObject.getAttributeValue(DQDAttributes.DETUNING.value)
        gLeft = self.dqdObject.getAttributeValue(DQDAttributes.G_FACTOR.value)[0]
        gRight = self.dqdObject.getAttributeValue(DQDAttributes.G_FACTOR.value)[1]
        socTheta = self.dqdObject.getAttributeValue(DQDAttributes.SOC_THETA_ANGLE.value) * np.pi / 180
        socPhi = self.dqdObject.getAttributeValue(DQDAttributes.SOC_PHI_ANGLE.value) * np.pi / 180

        points = {Axis.X.value: [], Axis.Y.value: [], Axis.Z.value: []}

        def getPoint(axis, n, sign, spinFlip):
            if axis == Axis.X.value:
                B = np.array([1, 0, 0])
                if spinFlip and abs(socTheta) < 1e-3 and abs(socPhi) < 1e-3:
                    return None
            elif axis == Axis.Y.value:
                B = np.array([0, 1, 0])
                if spinFlip and abs(socTheta - np.pi / 2) < 1e-3 and abs(socPhi) < 1e-3:
                    return None
            else:
                B = np.array([0, 0, 1])
                if spinFlip and abs(socPhi - np.pi / 2) < 1e-3:
                    return None

            effGL = np.linalg.norm(gLeft @ B)
            effGR = np.linalg.norm(gRight @ B)
            denom = abs(effGL + effGR) if spinFlip else abs(effGL - effGR)
            if denom < 1e-10:
                return None

            delta = detuning
            value = 2 * (n + delta if sign == '+' else n - delta) / denom
            return value

        axisIdx = [DQDParameterInterpreter.parseAttributeString(feature)[1] for feature in self.iterationFeatures]

        for axis in axisIdx:
            for n, sign, spinFlip in product(range(5), ['+', '-'], [False, True]):
                v = getPoint(axis, n, sign, spinFlip)
                if v is not None:
                    points[axis].append({'value': v, 'n': n, 'sign': sign, 'flip': spinFlip})

        def style(n, sign, flip):
            colors = {0: 'red', 1: 'blue', 2: 'green', 3: 'purple', 4: 'orange'}
            markers = {'+': 'o', '-': 's'} if not flip else {'+': '^', '-': 'v'}
            return colors[n], markers[sign]

        annotations = []

        for idx, axis in enumerate(axisIdx):
            for point in points[axis]:
                color, marker = style(point['n'], point['sign'], point['flip'])
                coord = {"x": point['value'], "y": 0} if idx == 0 else {"x": 0, "y": point['value']}
                annotations.append({
                    "type": "point",
                    "data": coord,
                    "style": {"color": color, "marker": marker, "markersize": 5},
                    "axis": idx
                })

        return annotations
