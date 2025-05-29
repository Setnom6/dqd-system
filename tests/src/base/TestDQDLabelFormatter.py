import unittest

from src.base.DQDLabelFormatter import DQDLabelFormatter
from src.base.DQDParameterInterpreter import NoAttributeParameters
from src.base.DoubleQuantumDot import DQDAttributes


class TestDQDLabelFormatter(unittest.TestCase):

    def testFormatLatexLabelSimple(self):
        """Test formatting of a simple LaTeX label."""
        formatter = DQDLabelFormatter([])
        label = formatter.formatLatexLabel("detuning")
        self.assertEqual(label, r"$\delta$/$\omega$")

    def testFormatLatexLabelWithAxis(self):
        """Test formatting of a LaTeX label with an axis."""
        formatter = DQDLabelFormatter([])
        label = formatter.formatLatexLabel("zeemanX")
        self.assertEqual(label, r"$Z$$_{{X}}$/$\omega$")

    def testFormatLatexLabelWithSide(self):
        """Test formatting of a LaTeX label with a side."""
        formatter = DQDLabelFormatter([])
        label = formatter.formatLatexLabel("gammaRight")
        self.assertEqual(label, r"$\gamma$$_{{RIGHT}}$/$\omega$")

    def testFormatLatexLabelWithAxisAndSide(self):
        """Test formatting of a LaTeX label with both axis and side."""
        formatter = DQDLabelFormatter([])
        label = formatter.formatLatexLabel("gFactorZLeft")
        self.assertEqual(label, r"$g$$_{{Z}}$$_{{LEFT}}$")

    def testFormatLatexLabelDividedByOmegaandMultipliedByMuB(self):
        """Test formatting of a LaTeX label divided by both muB and omega."""
        formatter = DQDLabelFormatter([])
        label = formatter.formatLatexLabel("magneticField")
        self.assertEqual(label, r"$\mu_B$$B$/$\omega$")

    def testGetLabels(self):
        """Test retrieval of formatted labels for iteration features."""
        features = ["detuning", "zeemanX", "gammaRight"]
        formatter = DQDLabelFormatter(features)
        labels = formatter.getLabels()
        self.assertEqual(len(labels), 3)
        self.assertEqual(labels[0], r"$\delta$/$\omega$")
        self.assertEqual(labels[1], r"$Z$$_{{X}}$/$\omega$")
        self.assertEqual(labels[2], r"$\gamma$$_{{RIGHT}}$/$\omega$")

    def testGetDependentLabels(self):
        """Test retrieval of dependent labels."""
        formatter = DQDLabelFormatter([])
        depLabels = formatter.getDependentLabels()
        self.assertIn(r"$I/ e \Gamma$", depLabels)
        self.assertIn(r"$P$", depLabels)

    def testGetTitleWithPlaceholders(self):
        """Test generation of a title with placeholders."""
        titleOptions = [
            DQDAttributes.DETUNING.value,
            DQDAttributes.SOC_THETA_ANGLE.value,
            DQDAttributes.SOC_PHI_ANGLE.value
        ]
        formatter = DQDLabelFormatter([])
        titleDict = formatter.getTitle(titleOptions)

        self.assertIn("title", titleDict)
        self.assertIn("placeholders", titleDict)
        self.assertIn(r"$\delta$", titleDict["title"])
        self.assertIn("º", titleDict["title"])  # Angles must include degree symbol
        self.assertEqual(len(titleDict["placeholders"]), 3)

    def testParseAttributeStringBasic(self):
        """Test parsing of a basic attribute string."""
        attr, axis, side = DQDLabelFormatter.parseAttributeString("zeemanXLeft")
        self.assertEqual(attr, "zeeman")
        self.assertEqual(axis, 0)
        self.assertEqual(side, 0)

    def testParseAttributeStringNoAxisOrSide(self):
        """Test parsing of an attribute string with no axis or side."""
        attr, axis, side = DQDLabelFormatter.parseAttributeString("tau")
        self.assertEqual(attr, "tau")
        self.assertIsNone(axis)
        self.assertIsNone(side)

    def testRetrieveFeaturesFromSimulationName(self):
        """Test retrieval of features from a simulation name."""
        simName = "zeemanX_zeemanZ"
        features = DQDLabelFormatter.retrieveFeaturesFromSimulationName(simName)
        self.assertEqual(features, ["zeemanX", "zeemanZ"])

    def testFormatLatexLabelWithScanAngle(self):
        """Test formatting of a LaTeX label for SCAN_ANGLE."""
        formatter = DQDLabelFormatter([])
        label = formatter.formatLatexLabel(NoAttributeParameters.SCAN_ANGLE.value)
        self.assertEqual(label, r"$\theta_{{XY}} / \pi$")

    def testGetTitleWithDegrees(self):
        """Test generation of a title with degree symbols."""
        titleOptions = [
            DQDAttributes.SOC_THETA_ANGLE.value,
            DQDAttributes.SOC_PHI_ANGLE.value
        ]
        formatter = DQDLabelFormatter([])
        titleDict = formatter.getTitle(titleOptions)

        self.assertIn("title", titleDict)
        self.assertIn("placeholders", titleDict)
        self.assertIn("º", titleDict["title"])  # Degree symbol must be included

    def testFormatLatexLabelUnknownFeature(self):
        """Test formatting of an unknown feature."""
        formatter = DQDLabelFormatter([])
        label = formatter.formatLatexLabel("unknownFeature")
        self.assertEqual(label, "unknownFeature")


if __name__ == "__main__":
    unittest.main()
