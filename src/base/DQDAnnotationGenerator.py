from itertools import product
from typing import Dict, Any, List, Union, Optional

import numpy as np

from src.base.DQDParameterInterpreter import Axis, DQDParameterInterpreter
from src.base.DoubleQuantumDot import DQDAttributes, DoubleQuantumDot


class DQDAnnotationGenerator:
    def __init__(self, dqdObject: DoubleQuantumDot, iterationFeatures: Union[str, List[str]],
                 independentArrays: Optional[List[np.ndarray]] = None):
        self.dqdObject = dqdObject
        self.iterationFeatures = (
            iterationFeatures if isinstance(iterationFeatures, list)
            else iterationFeatures.split("_")
        )
        self.independentArrays = independentArrays

    def generateAnnotations(self) -> List[Dict[str, Any]]:
        annotationType = self._decideAnnotationType()
        if annotationType == "ExpectedModuleResonances":
            return self._expectedModuleResonances()
        elif annotationType == "ExpectedGTensorResonances":
            return self._expectedGTensorResonances()
        elif annotationType == "SpinConservingAndSpinFlipDetuningMagnetic":
            return self._spinConservingAndSpinFlipDetuningMagnetic()
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

        if (
                len(self.iterationFeatures) == 2 and
                self.iterationFeatures[0] == "detuning" and
                self.iterationFeatures[1] in ["magneticFieldX", "magneticFieldY", "magneticFieldZ"]
        ):
            return "SpinConservingAndSpinFlipDetuningMagnetic"

        return ""

    def _expectedModuleResonances(self) -> List[Dict[str, Any]]:
        detuning = self.dqdObject.getAttributeValue(DQDAttributes.DETUNING.value)
        nValues = range(0, 3)
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

    def _spinConservingAndSpinFlipDetuningMagnetic(self) -> List[Dict[str, Any]]:
        if self.independentArrays is None:
            return []

        gLeft, gRight = self.dqdObject.getAttributeValue(DQDAttributes.G_FACTOR.value)
        fieldStr = self.iterationFeatures[1]
        detuningStr = self.iterationFeatures[0]

        axisMap = {
            "magneticFieldX": np.array([1, 0, 0]),
            "magneticFieldY": np.array([0, 1, 0]),
            "magneticFieldZ": np.array([0, 0, 1])
        }

        componentIdxMap = {
            "magneticFieldX": 0,
            "magneticFieldY": 1,
            "magneticFieldZ": 2
        }

        if fieldStr not in axisMap:
            return []

        direction = axisMap[fieldStr]
        componentIdx = componentIdxMap[fieldStr]

        deltaVals = np.linspace(self.independentArrays[0][0], self.independentArrays[0][-1], num=300)
        fieldVals = np.linspace(self.independentArrays[1][0], self.independentArrays[1][-1], num=300)

        # Preparar mapa de colores para los 4 tipos
        colorMap = {
            "diffPlus": "darkblue",
            "diffMinus": "lightblue",
            "sumPlus": "red",
            "sumMinus": "orange"
        }

        # Línea distinta por n
        lineStyles = ['-', '--', '-.', ':', (0, (3, 5, 1, 5)), (0, (1, 1))]

        eps = 0.05  # tolerancia
        annotations = []
        groupedLines = {}  # (color, linestyle) -> list of (delta, Bmag)

        for delta in deltaVals:
            for Bmag in fieldVals:
                Bvec = Bmag * direction
                gLeftProj = gLeft @ Bvec
                gRightProj = gRight @ Bvec
                gDiffProj = gRightProj - gLeftProj
                gSumProj = gRightProj + gLeftProj

                diffComp = abs(gDiffProj[componentIdx])
                sumComp = abs(gSumProj[componentIdx])

                for n in range(5):  # ajusta si quieres más o menos líneas
                    targetPlus = 2 * (n + delta)
                    targetMinus = 2 * (n - delta)
                    style = lineStyles[n % len(lineStyles)]

                    # 1. Azul oscuro
                    if abs(diffComp - targetPlus) < eps:
                        groupedLines.setdefault(("darkblue", style), []).append((delta, Bmag))
                    # 2. Azul claro
                    if abs(diffComp - targetMinus) < eps:
                        groupedLines.setdefault(("lightblue", style), []).append((delta, Bmag))
                    # 3. Rojo
                    if abs(sumComp - targetPlus) < eps:
                        groupedLines.setdefault(("red", style), []).append((delta, Bmag))
                    # 4. Naranja
                    if abs(sumComp - targetMinus) < eps:
                        groupedLines.setdefault(("orange", style), []).append((delta, Bmag))

        # Detecta qué eje es X
        xIsDelta = (detuningStr.lower() == "detuning")

        # Límite de salto para dividir líneas (relativo al tamaño del grafo)
        deltaValsSorted = np.sort(deltaVals)
        fieldValsSorted = np.sort(fieldVals)
        deltaStep = np.min(np.diff(deltaValsSorted)) if len(deltaValsSorted) > 1 else 0.1
        fieldStep = np.min(np.diff(fieldValsSorted)) if len(fieldValsSorted) > 1 else 0.1
        maxJump = 2 * max(deltaStep, fieldStep)

        for (color, linestyle), points in groupedLines.items():
            # Ordenar
            if xIsDelta:
                points = sorted(points, key=lambda p: p[0])
            else:
                points = sorted(points, key=lambda p: p[1])

            # Segmentar
            segment = [points[0]]
            for i in range(1, len(points)):
                dx = points[i][0] - points[i - 1][0]
                dy = points[i][1] - points[i - 1][1]
                if np.hypot(dx, dy) > maxJump:
                    if len(segment) > 1:
                        xVals, yVals = zip(*segment)
                        annotations.append({
                            "type": "line",
                            "data": {"x": xVals, "y": yVals},
                            "style": {
                                "color": color,
                                "linestyle": linestyle,
                                "linewidth": 1
                            },
                            "absoluteCoordinates": True
                        })
                    segment = []
                segment.append(points[i])

            # Último segmento
            if len(segment) > 1:
                xVals, yVals = zip(*segment)
                annotations.append({
                    "type": "line",
                    "data": {"x": xVals, "y": yVals},
                    "style": {
                        "color": color,
                        "linestyle": linestyle,
                        "linewidth": 1
                    },
                    "absoluteCoordinates": True
                })

        return annotations
