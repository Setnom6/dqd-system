from typing import Dict, List, Tuple

from src.base.DQDParameterInterpreter import NoAttributeParameters, Side, Axis
from src.base.DoubleQuantumDot import DQDAttributes


class DQDLabelFormatter:
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
        DQDAttributes.SOC_THETA_ANGLE.value: r"$\theta_{{SO}}$",
        DQDAttributes.SOC_PHI_ANGLE.value: r"$\phi_{{SO}}$",
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

    DEGREES = [DQDAttributes.SOC_THETA_ANGLE.value, DQDAttributes.SOC_PHI_ANGLE.value]

    def __init__(self, iterationFeatures: List[str]):
        self.iterationFeatures = iterationFeatures

    def getLabels(self) -> List[str]:
        return [self.formatLatexLabel(feature) for feature in self.iterationFeatures]

    def getDependentLabels(self) -> List[str]:
        return [self.formatLatexLabel(DQDAttributes.SUM_CURRENT.value),
                self.formatLatexLabel(DQDAttributes.POLARITY.value)]

    def getTitle(self, titleOptions: List[str]) -> Dict[str, str]:
        titleParts = []
        placeholders = []
        for feature in titleOptions:
            name, _, _ = self.parseAttributeString(feature)
            if name in [attr.value for attr in DQDAttributes]:
                formattedLabel = self.formatLatexLabel(feature)
                titlePart = f"{formattedLabel} = {{}}"
                placeholders.append(feature)
                if name in self.DEGREES:
                    titlePart += "º"
            else:
                titlePart = feature
            titleParts.append(titlePart)
        return {"title": ", ".join(titleParts), "placeholders": placeholders}

    def formatLatexLabel(self, feature: str) -> str:
        name, axis, side = self.parseAttributeString(feature)
        symbol = self.LATEX_SYMBOLS.get(name, feature)

        if axis is not None:
            symbol += r"$_{{" + self.AXIS_SYMBOLS[axis] + "}}$"
        if side is not None:
            symbol += r"$_{{" + self.SIDE_SYMBOLS[side] + "}}$"

        if name in self.DIVIDED_BY_OMEGA and name in self.DIVIDED_BY_MUB:
            symbol += "/" + self.LATEX_SYMBOLS["acFrequency"] + self.LATEX_SYMBOLS["muB"]
        elif name in self.DIVIDED_BY_MUB:
            symbol += "/" + self.LATEX_SYMBOLS["muB"]
        elif name in self.DIVIDED_BY_OMEGA:
            symbol += "/" + self.LATEX_SYMBOLS["acFrequency"]

        return symbol

    @staticmethod
    def parseAttributeString(attributeString: str) -> Tuple[str, int, int]:
        side = None
        axis = None

        if attributeString.endswith("Left"):
            side = Side.LEFT.value
            attributeString = attributeString[:-4]
        elif attributeString.endswith("Right"):
            side = Side.RIGHT.value
            attributeString = attributeString[:-5]

        if attributeString and attributeString[-1] in Axis.__members__:
            axis = Axis[attributeString[-1]].value
            attributeString = attributeString[:-1]

        return attributeString, axis, side

    @staticmethod
    def retrieveFeaturesFromSimulationName(simulationName: str) -> List[str]:
        features = simulationName.split("_")
        return [f for f in features]
