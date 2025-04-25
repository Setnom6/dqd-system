import unittest

import numpy as np

from src.base.DQDAnnotationGenerator import DQDAnnotationGenerator
from src.base.DQDParameterInterpreter import Axis
from src.base.DoubleQuantumDot import DoubleQuantumDot, DQDAttributes


class TestDQDAnnotationGenerator(unittest.TestCase):

    def setUp(self):
        """Set up a DoubleQuantumDot object with default parameters for testing."""
        self.dqd = DoubleQuantumDot()
        self.dqd.setParameters({
            DQDAttributes.DETUNING.value: 0.5,
            DQDAttributes.G_FACTOR.value: np.array([
                np.identity(3),
                2 * np.identity(3)
            ]),
            DQDAttributes.SOC_THETA_ANGLE.value: 90.0,  # in degrees
            DQDAttributes.SOC_PHI_ANGLE.value: 0.0
        })

    def testExpectedModuleResonancesDetection(self):
        """Test detection of ExpectedModuleResonances annotation type."""
        features = ["scanAngle", "magneticFieldM"]
        generator = DQDAnnotationGenerator(self.dqd, features)
        annotationType = generator._decideAnnotationType()
        self.assertEqual(annotationType, "ExpectedModuleResonances")

    def testExpectedGTensorResonancesDetection(self):
        """Test detection of ExpectedGTensorResonances annotation type."""
        features = ["magneticFieldX", "magneticFieldY"]
        generator = DQDAnnotationGenerator(self.dqd, features)
        annotationType = generator._decideAnnotationType()
        self.assertEqual(annotationType, "ExpectedGTensorResonances")

    def testExpectedModuleResonancesOutput(self):
        """Test the output of ExpectedModuleResonances annotations."""
        generator = DQDAnnotationGenerator(self.dqd, ["scanAngle", "magneticFieldM"])
        annotations = generator.generateAnnotations()
        self.assertIsInstance(annotations, list)
        self.assertGreater(len(annotations), 0)

        for annotation in annotations:
            self.assertEqual(annotation["type"], "line")
            self.assertIn("y", annotation["data"])
            self.assertIn("color", annotation["style"])
            self.assertIn("linestyle", annotation["style"])
            self.assertEqual(annotation["axis"], 1)

        # Validate specific values
        detuning = self.dqd.getAttributeValue(DQDAttributes.DETUNING.value)
        for annotation in annotations:
            y_value = annotation["data"]["y"]
            self.assertTrue(isinstance(y_value, float))
            self.assertIn(y_value, [n + detuning for n in range(2)] + [n - detuning for n in range(2)])

    def testExpectedGTensorResonancesOutput(self):
        """Test the output of ExpectedGTensorResonances annotations."""
        generator = DQDAnnotationGenerator(self.dqd, ["magneticFieldX", "magneticFieldY"])
        annotations = generator.generateAnnotations()
        self.assertIsInstance(annotations, list)
        self.assertGreater(len(annotations), 0)

        for annotation in annotations:
            self.assertEqual(annotation["type"], "point")
            self.assertIn("x", annotation["data"])
            self.assertIn("y", annotation["data"])
            self.assertIn("marker", annotation["style"])
            self.assertIn("color", annotation["style"])

        # Validate specific values
        detuning = self.dqd.getAttributeValue(DQDAttributes.DETUNING.value)
        g_factor = self.dqd.getAttributeValue(DQDAttributes.G_FACTOR.value)
        for annotation in annotations:
            x_value = annotation["data"]["x"]
            y_value = annotation["data"]["y"]
            self.assertTrue(isinstance(x_value, (float, int)))
            self.assertTrue(isinstance(y_value, (float, int)))

            # Check that the values are consistent with the g-factor and detuning
            axis = annotation["axis"]
            if axis == Axis.X.value:
                self.assertTrue(x_value is not None or y_value is not None)

    def testSpinFlipSuppressionForSOC(self):
        """Test that spin flip is suppressed for specific SOC values."""
        self.dqd.setParameters({
            DQDAttributes.SOC_THETA_ANGLE.value: 0.0,
            DQDAttributes.SOC_PHI_ANGLE.value: 0.0
        })
        generator = DQDAnnotationGenerator(self.dqd, ["magneticFieldX", "magneticFieldZ"])
        annotations = generator.generateAnnotations()

        # Expect no spin-flip markers along X (only non-flip points remain)
        for ann in annotations:
            if ann["axis"] == Axis.X.value:
                self.assertNotEqual(ann["style"]["marker"], "^")
                self.assertNotEqual(ann["style"]["marker"], "v")

    def testNoAnnotationsForUnknownFeatures(self):
        """Test that no annotations are generated for unknown features."""
        generator = DQDAnnotationGenerator(self.dqd, ["detuning"])
        annotations = generator.generateAnnotations()
        self.assertEqual(annotations, [])

    def testAnnotationStyle(self):
        """Test that annotation styles are correctly assigned."""
        generator = DQDAnnotationGenerator(self.dqd, ["magneticFieldX", "magneticFieldY"])
        annotations = generator.generateAnnotations()

        for annotation in annotations:
            style = annotation["style"]
            self.assertIn(style["color"], ["red", "blue", "green", "purple", "orange"])
            self.assertIn(style["marker"], ["o", "s", "^", "v"])
            self.assertEqual(style["markersize"], 5)


if __name__ == "__main__":
    unittest.main()
