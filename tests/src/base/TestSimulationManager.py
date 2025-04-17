import unittest

import numpy as np

from src.base.SimulationManager import SimulationManager

# Define mock functions globally to make them pickleable
updatedParameters = []


def mockUpdateParameters(*indices):
    updatedParameters.append(indices)


def mockSimulatePoint():
    # Handle cases where updatedParameters[-1] might have a single index
    if len(updatedParameters[-1]) == 1:
        return (updatedParameters[-1][0], updatedParameters[-1][0])  # Sum and product are the same in 1D
    else:
        return (sum(updatedParameters[-1]), np.prod(updatedParameters[-1]))


class TestSimulationManager(unittest.TestCase):
    def setUp(self):
        # Clear the updated parameters list before each test
        global updatedParameters
        updatedParameters = []

    def testRunSimulation1D(self):
        # Test a 1D simulation
        lenArrays = [5]  # 1D grid with 5 points
        manager = SimulationManager(lenArrays, mockUpdateParameters, mockSimulatePoint)

        resultArrays = manager.runSimulation()

        # Check the shape of the result arrays
        self.assertEqual(len(resultArrays), 2)  # Two outputs from simulatePoint
        self.assertEqual(resultArrays[0].shape, (5,))
        self.assertEqual(resultArrays[1].shape, (5,))

        # Check the values in the result arrays
        expectedSums = np.arange(5)  # Sum of indices
        expectedProducts = np.arange(5)  # Product of indices (same as indices in 1D)
        self.assertTrue(np.all(resultArrays[0] == expectedSums))
        self.assertTrue(np.all(resultArrays[1] == expectedProducts))

    def testRunSimulation2D(self):
        # Test a 2D simulation
        lenArrays = [3, 4]  # 2D grid with shape (3, 4)
        manager = SimulationManager(lenArrays, mockUpdateParameters, mockSimulatePoint)

        resultArrays = manager.runSimulation()

        # Check the shape of the result arrays
        self.assertEqual(len(resultArrays), 2)  # Two outputs from simulatePoint
        self.assertEqual(resultArrays[0].shape, (3, 4))
        self.assertEqual(resultArrays[1].shape, (3, 4))

        # Check the values in the result arrays
        expectedSums = np.zeros((3, 4))
        expectedProducts = np.zeros((3, 4))
        for i in range(3):
            for j in range(4):
                expectedSums[i, j] = i + j
                expectedProducts[i, j] = i * j
        self.assertTrue(np.all(resultArrays[0] == expectedSums))
        self.assertTrue(np.all(resultArrays[1] == expectedProducts))

    def testRunSimulation3D(self):
        # Test a 3D simulation
        lenArrays = [2, 3, 4]  # 3D grid with shape (2, 3, 4)
        manager = SimulationManager(lenArrays, mockUpdateParameters, mockSimulatePoint)

        resultArrays = manager.runSimulation()

        # Check the shape of the result arrays
        self.assertEqual(len(resultArrays), 2)  # Two outputs from simulatePoint
        self.assertEqual(resultArrays[0].shape, (2, 3, 4))
        self.assertEqual(resultArrays[1].shape, (2, 3, 4))

        # Check the values in the result arrays
        expectedSums = np.zeros((2, 3, 4))
        expectedProducts = np.zeros((2, 3, 4))
        for i in range(2):
            for j in range(3):
                for k in range(4):
                    expectedSums[i, j, k] = i + j + k
                    expectedProducts[i, j, k] = i * j * k
        self.assertTrue(np.all(resultArrays[0] == expectedSums))
        self.assertTrue(np.all(resultArrays[1] == expectedProducts))

    def testSimulatePoint(self):
        # Test the _simulatePoint method
        lenArrays = [3, 4]  # 2D grid with shape (3, 4)
        manager = SimulationManager(lenArrays, mockUpdateParameters, mockSimulatePoint)

        # Simulate a single point
        result = manager._simulatePoint(1, 2)

        # Check the result
        self.assertEqual(result[:2], (1, 2))  # Indices
        self.assertEqual(result[2], 3)  # Sum of indices
        self.assertEqual(result[3], 2)  # Product of indices


if __name__ == "__main__":
    unittest.main()
