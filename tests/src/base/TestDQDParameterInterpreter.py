import unittest

import numpy as np

from src.base.DQDParameterInterpreter import DQDParameterInterpreter, Axis, Side
from src.base.DoubleQuantumDot import DQDAttributes


class TestDQDParameterInterpreter(unittest.TestCase):

    def setUp(self):
        # Fixed parameters for testing
        self.fixedParams = {
            DQDAttributes.AC_AMPLITUDE.value: 1.2,
            DQDAttributes.GAMMA.value + "Left": [0.1],
            DQDAttributes.GAMMA.value + "Right": [0.2],
            DQDAttributes.G_FACTOR.value: [
                [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                [[2, 0, 0], [0, 2, 0], [0, 0, 2]]
            ],
            DQDAttributes.ZEEMAN.value: [[0.0, 0.0, 1.0], [0.0, 0.0, 2.0]],
        }
        self.iterationParams = [
            {"features": DQDAttributes.AC_AMPLITUDE.value, "array": np.array([1.0, 2.0, 3.0])}
        ]
        self.interpreter = DQDParameterInterpreter(self.fixedParams, self.iterationParams)

    def testGetFixedParametersProcessed(self):
        """Test that fixed parameters are processed correctly and values match expectations."""
        params = self.interpreter.getFixedParameters()

        # Check GAMMA
        self.assertIn(DQDAttributes.GAMMA.value, params)
        expected_gamma = np.array([[0.1], [0.2]])
        self.assertTrue(np.allclose(params[DQDAttributes.GAMMA.value], expected_gamma))

        # Check G_FACTOR
        self.assertIn(DQDAttributes.G_FACTOR.value, params)
        expected_g_factor = np.array([
            [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            [[2, 0, 0], [0, 2, 0], [0, 0, 2]]
        ])
        self.assertTrue(np.allclose(params[DQDAttributes.G_FACTOR.value], expected_g_factor))

        # Check ZEEMAN
        self.assertIn(DQDAttributes.ZEEMAN.value, params)
        expected_zeeman = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 2.0]])
        self.assertTrue(np.allclose(params[DQDAttributes.ZEEMAN.value], expected_zeeman))

    def testParseAttributeString(self):
        """Test parsing of attribute strings into base name, axis, and side."""
        base, axis, side = DQDParameterInterpreter.parseAttributeString("zeemanXLeft")
        self.assertEqual(base, "zeeman")
        self.assertEqual(axis, Axis.X.value)
        self.assertEqual(side, Side.LEFT.value)

        base, axis, side = DQDParameterInterpreter.parseAttributeString("gFactorZRight")
        self.assertEqual(base, "gFactor")
        self.assertEqual(axis, Axis.Z.value)
        self.assertEqual(side, Side.RIGHT.value)

        base, axis, side = DQDParameterInterpreter.parseAttributeString("tau")
        self.assertEqual(base, "tau")
        self.assertIsNone(axis)
        self.assertIsNone(side)

    def testGetUpdateFunctions(self):
        """Test generation of update functions for iteration parameters."""
        funcs = self.interpreter.getUpdateFunctions(1)  # Index 1 => value 2.0 in array
        self.assertEqual(len(funcs), 1)
        func, name = funcs[0]
        updated = func(1.0)
        self.assertEqual(name, DQDAttributes.AC_AMPLITUDE.value)
        self.assertEqual(updated[DQDAttributes.AC_AMPLITUDE.value], 2.0)

    def testSimulationName(self):
        """Test generation of simulation name based on iteration features."""
        self.assertEqual(self.interpreter.getSimulationName(), DQDAttributes.AC_AMPLITUDE.value)

    def testIndependentArrays(self):
        """Test retrieval of independent arrays."""
        arrays = self.interpreter.getIndependentArrays()
        self.assertEqual(len(arrays), 1)
        self.assertTrue(np.allclose(arrays[0], np.array([1.0, 2.0, 3.0])))

    def testAdjustGamma(self):
        """Test adjustment of gamma values."""
        gamma = np.array([[0.1], [0.2]])
        updated = self.interpreter._adjustGamma(gamma, 0.5, side=Side.LEFT.value)
        expected = np.array([[0.5], [0.2]])
        self.assertTrue(np.allclose(updated, expected))

        updated = self.interpreter._adjustGamma(gamma, 0.7, side=None)
        expected = np.array([[0.7], [0.7]])
        self.assertTrue(np.allclose(updated, expected))

    def testAdjustZeeman(self):
        """Test adjustment of Zeeman splitting values."""
        zeeman = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 2.0]])
        updated = self.interpreter._adjustZeeman(zeeman, 0.5, axis=Axis.Z.value, side=Side.LEFT.value)
        expected = np.array([[0.0, 0.0, 0.5], [0.0, 0.0, 2.0]])
        self.assertTrue(np.allclose(updated, expected))

        updated = self.interpreter._adjustZeeman(zeeman, 1.0, axis=Axis.M.value, side=None)
        self.assertTrue(np.allclose(np.linalg.norm(updated[0]), 1.0))
        self.assertTrue(np.allclose(np.linalg.norm(updated[1]), 1.0))

    def testAdjustMagneticField(self):
        """Test adjustment of magnetic field values."""
        magnetic_field = np.array([1.0, 0.0, 0.0])
        updated = self.interpreter._adjustMagneticField(magnetic_field, 2.0, axis=Axis.X.value)
        expected = np.array([2.0, 0.0, 0.0])
        self.assertTrue(np.allclose(updated, expected))

        updated = self.interpreter._adjustMagneticField(magnetic_field, 1.0, axis=Axis.M.value)
        self.assertTrue(np.allclose(np.linalg.norm(updated), 1.0))

    def testAdjustScanAngleDict(self):
        """Test adjustment of Zeeman splitting and magnetic field based on scan angle."""
        magnetic_field = np.array([1.0, 0.0, 0.0])
        g_factor = np.array([[[1, 0, 0], [0, 1, 0], [0, 0, 1]], [[2, 0, 0], [0, 2, 0], [0, 0, 2]]])
        zeeman, b_vector = self.interpreter._adjustScanAngleDict(magnetic_field, g_factor, 0.5)

        # Expected magnetic field
        expected_b_vector = np.array([np.cos(0.5 * np.pi), np.sin(0.5 * np.pi), 0])
        self.assertTrue(np.allclose(b_vector, expected_b_vector))

        # Expected Zeeman splitting
        expected_zeeman = np.array([
            g_factor[0] @ expected_b_vector,
            g_factor[1] @ expected_b_vector
        ])
        self.assertTrue(np.allclose(zeeman, expected_zeeman))


if __name__ == "__main__":
    unittest.main()
