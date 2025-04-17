import unittest

import numpy as np

from src.base.AttributeInterpreter import AttributeInterpreter, Axis, Side
from src.base.DoubleQuantumDot import DQDAttributes


class TestAttributeInterpreter(unittest.TestCase):

    def setUp(self):
        # Fixed parameters for initialization
        self.fixedParams = {
            DQDAttributes.GAMMA.value + "Left": 0.1,
            DQDAttributes.ZEEMAN.value: [[1, 0, 0], [0, 1, 0]],
            DQDAttributes.MAGNETIC_FIELD.value: [1.0, 2.0, 3.0],
        }

        # Iteration parameters for initialization
        self.iterParams = [
            {"features": DQDAttributes.GAMMA.value + "Right", "array": np.array([0.1, 0.2, 0.3])},
            {"features": DQDAttributes.MAGNETIC_FIELD.value + "Z", "array": np.array([1.0, 2.0, 3.0])},
        ]

        # Initialize AttributeInterpreter
        self.interpreter = AttributeInterpreter(self.fixedParams, self.iterParams)

    def testInitialization(self):
        # Test fixed parameters processing
        fixed = self.interpreter.fixedParameters
        self.assertIn(DQDAttributes.GAMMA.value, fixed)
        self.assertIn(DQDAttributes.ZEEMAN.value, fixed)
        self.assertIn(DQDAttributes.MAGNETIC_FIELD.value, fixed)

        self.assertEqual(fixed[DQDAttributes.GAMMA.value][0], 0.1)  # Left side
        self.assertTrue(np.all(fixed[DQDAttributes.ZEEMAN.value] == np.array([[1, 0, 0], [0, 1, 0]])))
        self.assertTrue(np.all(fixed[DQDAttributes.MAGNETIC_FIELD.value] == np.array([1.0, 2.0, 3.0])))

        # Test iteration parameters processing
        self.assertEqual(len(self.interpreter.iterationArrays), 2)
        self.assertEqual(len(self.interpreter.iterationParameterFeatures), 2)
        self.assertTrue(np.all(self.interpreter.iterationArrays[0] == np.array([0.1, 0.2, 0.3])))
        self.assertTrue(np.all(self.interpreter.iterationArrays[1] == np.array([1.0, 2.0, 3.0])))

    def testGetIterableAttributes(self):
        # Test retrieval of iterable attributes
        iterables = self.interpreter.getIndependentArrays()
        self.assertEqual(len(iterables), 2)
        self.assertTrue(np.all(iterables[0] == np.array([0.1, 0.2, 0.3])))
        self.assertTrue(np.all(iterables[1] == np.array([1.0, 2.0, 3.0])))

    def testParseAttributeString(self):
        # Test parsing of attribute strings
        name, axis, side = self.interpreter.parseAttributeString(DQDAttributes.GAMMA.value + "Right")
        self.assertEqual(name, DQDAttributes.GAMMA.value)
        self.assertIsNone(axis)
        self.assertEqual(side, Side.RIGHT.value)

        name, axis, side = self.interpreter.parseAttributeString(DQDAttributes.ZEEMAN.value + "MLeft")
        self.assertEqual(name, DQDAttributes.ZEEMAN.value)
        self.assertEqual(axis, Axis.M.value)
        self.assertEqual(side, Side.LEFT.value)

        name, axis, side = self.interpreter.parseAttributeString(DQDAttributes.MAGNETIC_FIELD.value + "Z")
        self.assertEqual(name, DQDAttributes.MAGNETIC_FIELD.value)
        self.assertEqual(axis, Axis.Z.value)
        self.assertIsNone(side)

    def testFormatLatexLabel(self):
        # Test LaTeX label formatting
        label = self.interpreter.formatLatexLabel(DQDAttributes.GAMMA.value + "Right")
        self.assertIn(r"$_{{RIGHT}}$", label)
        self.assertIn(r"\gamma", label)

        label = self.interpreter.formatLatexLabel(DQDAttributes.ZEEMAN.value + "MLeft")
        self.assertIn(r"$_{{M}}$", label)
        self.assertIn(r"$_{{LEFT}}$", label)
        self.assertIn(r"Z", label)

        label = self.interpreter.formatLatexLabel(DQDAttributes.MAGNETIC_FIELD.value + "Z")
        self.assertIn(r"$_{{Z}}$", label)
        self.assertIn(r"B", label)

    def testGetLabels(self):
        # Test retrieval of labels for iteration parameters
        labels = self.interpreter.getLabels()
        self.assertEqual(len(labels), 2)
        self.assertIn(r"$_{{RIGHT}}$", labels[0])
        self.assertIn(r"\gamma", labels[0])
        self.assertIn(r"$_{{Z}}$", labels[1])
        self.assertIn(r"B", labels[1])

    def testGetSimulationName(self):
        # Test generation of simulation name
        simName = self.interpreter.getSimulationName()
        expected = f"{DQDAttributes.GAMMA.value}Right_{DQDAttributes.MAGNETIC_FIELD.value}Z"
        self.assertEqual(simName, expected)

    def testGetTitle(self):
        # Test title generation
        titleOptions = [DQDAttributes.GAMMA.value + "Right", DQDAttributes.MAGNETIC_FIELD.value + "Z"]
        title = self.interpreter.getTitle(titleOptions)
        self.assertIn(r"\gamma", title["title"])
        self.assertIn(r"B", title["title"])
        self.assertIn(r"$_{{RIGHT}}$", title["title"])
        self.assertIn(r"$_{{Z}}$", title["title"])
        self.assertEqual(len(title["placeholders"]), 2)

    def testGetUpdateFunctions(self):
        # Test generation of update functions
        updateFunctions = self.interpreter.getUpdateFunctions(0, 1)
        self.assertEqual(len(updateFunctions), 2)

        # Test first update function (gammaRight)
        updater, paramName = updateFunctions[0]
        self.assertEqual(paramName, DQDAttributes.GAMMA.value)
        updatedValue = updater(np.array([[0.5], [0.6]]))
        self.assertTrue(np.all(updatedValue[DQDAttributes.GAMMA.value] == np.array([[0.5], [0.1]])))

        # Test second update function (magneticFieldZ)
        updater, paramName = updateFunctions[1]
        self.assertEqual(paramName, DQDAttributes.MAGNETIC_FIELD.value)
        updatedValue = updater(np.array([1.0, 2.0, 3.0]))
        self.assertTrue(np.all(updatedValue[DQDAttributes.MAGNETIC_FIELD.value] == np.array([1.0, 2.0, 2.0])))


if __name__ == "__main__":
    unittest.main()
