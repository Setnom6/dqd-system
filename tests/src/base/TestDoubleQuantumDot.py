from unittest.mock import patch

import numpy as np
import pytest

from src.base.DoubleQuantumDot import DoubleQuantumDot


class TestDoubleQuantumDot:
    # Default DoubleQuantumDot instance for checks

    AC_AMPLITUDE = 2.0
    GAMMA = np.array([[0.5], [0.74]])
    MAGNETICFIELD = np.array([1.0, 2.0, 3.0])
    GFACTOR = np.array([[[2.0, 0.0, 0.0],
                         [0.0, 4.0, 0.0],
                         [0.0, 0.0, 3.0]],
                        [[5.0, 0.0, 0.0],
                         [0.0, -4.0, 0.0],
                         [0.0, 0.0, 8.0]]])
    ZEEMAN = np.array([[2.0, 8.0, 9.0],
                       [5.0, -8.0, 24.0]])

    def testInitialization(self):
        dqd = DoubleQuantumDot()
        assert dqd.acAmplitude == 1.2
        assert dqd.chi == 0.1
        assert dqd.tau == 0.1
        assert dqd.groundRightEnergy == 1.0
        assert dqd.detuning == 0.0
        assert np.all(dqd.gamma == np.array([[0.01], [0.01]]))
        assert np.all(dqd.zeeman == np.zeros((2, 3)))
        assert np.all(dqd.gFactor[0] == np.identity(3))
        assert np.all(dqd.gFactor[1] == np.identity(3))
        assert np.all(dqd.magneticField == np.zeros(3))
        assert np.all(dqd.OME == np.zeros((2, 3)))
        assert dqd.factorBetweenOMEAndZeeman == 0.5
        assert dqd.alphaThetaAngle == pytest.approx(np.pi / 2)
        assert dqd.alphaPhiAngle == 0.0
        assert dqd.sumCurrent is None
        assert dqd.polarity is None

    def testSetParameters(self):
        """Test setting parameters in DoubleQuantumDot."""
        dqd = DoubleQuantumDot()
        params = {
            "acAmplitude": self.AC_AMPLITUDE,
            "gamma": self.GAMMA,
            "magneticField": self.MAGNETICFIELD,
            "gFactor": self.GFACTOR,
        }
        dqd.setParameters(params)
        assert dqd.acAmplitude == self.AC_AMPLITUDE
        assert np.all(dqd.gamma == self.GAMMA)
        assert np.all(dqd.gFactor == self.GFACTOR)
        assert np.all(dqd.magneticField == self.MAGNETICFIELD)

        # Check that the Zeeman effect is updated correctly
        expectedZeeman = np.zeros((2, 3))
        expectedZeeman[0] = self.GFACTOR[0] @ self.MAGNETICFIELD
        expectedZeeman[1] = self.GFACTOR[1] @ self.MAGNETICFIELD
        assert np.allclose(dqd.zeeman, expectedZeeman)

        # Test invalid parameter
        with pytest.raises(AttributeError, match="'invalidParam' is not a valid attribute"):
            dqd.setParameters({"invalidParam": 1.0})

        # Test type mismatch
        with pytest.raises(ValueError, match="'acAmplitude' must be a numeric value"):
            dqd.setParameters({"acAmplitude": "invalidType"})

        # Test invalid gamma shape
        with pytest.raises(ValueError, match="'gamma' must be a numpy array with shape \\(2, 1\\)"):
            dqd.setParameters({"gamma": np.array([0.5, 0.74])})

        # Test invalid magneticField shape
        with pytest.raises(ValueError, match="'magneticField' must be a numpy array with shape \\(3,\\)"):
            dqd.setParameters({"magneticField": np.array([1.0, 2.0])})

        # Test invalid gFactor shape
        with pytest.raises(ValueError, match="'gFactor' must be a numpy array with shape \\(2, 3, 3\\)"):
            dqd.setParameters({"gFactor": np.identity(3)})

        # Test negative acAmplitude
        with pytest.raises(ValueError, match="'acAmplitude' must be non-negative"):
            dqd.setParameters({"acAmplitude": -1.0})

    def testTimeIndependentHamiltonian(self):
        dqd = DoubleQuantumDot()
        expectedH0 = np.array([
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, -0.09, -0.01],
            [0.0, 0.0, 1.0, 0.01, -0.09],
            [0.0, -0.09, 0.01, 1.0, 0.0],
            [0.0, -0.01, -0.09, 0.0, 1.0]
        ], dtype=complex)

        h0 = dqd._timeIndependentHamiltonian()
        assert h0.shape == (5, 5)
        assert np.allclose(h0, expectedH0)  # Compare with the expected Hamiltonian
        assert np.allclose(h0, h0.T.conj())  # Hermitian check

    def testOscillatoryHamiltonian(self):
        dqd = DoubleQuantumDot()
        expectedH1 = np.array([
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.2, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.2, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, -0.0]
        ], dtype=complex)

        with patch.object(dqd, '_calculateOMEFromZeeman', wraps=dqd._calculateOMEFromZeeman) as mock_calculate_OME:
            h1 = dqd._oscillatoryHamiltonian()
            mock_calculate_OME.assert_called_once()  # Ensure the function is called
            assert h1.shape == (5, 5)
            assert np.allclose(h1, expectedH1)  # Compare with the expected Hamiltonian
            assert np.allclose(h1, h1.T.conj())  # Hermitian check

    def testCalculateOMEFromZeeman(self):
        dqd = DoubleQuantumDot()
        dqd.setParameters({"gFactor": self.GFACTOR,
                           "magneticField": self.MAGNETICFIELD})
        expectedOME = np.array([
            [-4.50000000e+00, 2.75545530e-16, 1.00000000e+00],
            [-1.20000000e+01, 7.34788079e-16, 2.50000000e+00]
        ])

        dqd._calculateOMEFromZeeman()
        assert np.allclose(dqd.OME, expectedOME)  # Compare with the expected OME

    def testComputeAndGetCurrent(self):
        dqd = DoubleQuantumDot()
        with patch.object(dqd, '_timeIndependentHamiltonian', wraps=dqd._timeIndependentHamiltonian) as mock_h0, \
                patch.object(dqd, '_oscillatoryHamiltonian', wraps=dqd._oscillatoryHamiltonian) as mock_h1, \
                patch.object(dqd, '_collapseOperators', wraps=dqd._collapseOperators) as mock_collapse, \
                patch.object(dqd, '_getChargeObservables', wraps=dqd._getChargeObservables) as mock_observables:
            dqd.computeCurrent(iterationsPerPeriod=10)
            mock_h0.assert_called_once()  # Ensure _timeIndependentHamiltonian is called
            mock_h1.assert_called_once()  # Ensure _oscillatoryHamiltonian is called
            mock_collapse.assert_called_once()  # Ensure _collapseOperators is called
            mock_observables.assert_called_once()  # Ensure _getChargeObservables is called

        sumCurrent, polarity = dqd.getCurrent()
        assert sumCurrent is not None
        assert polarity is not None

    def testToDict(self):
        dqd = DoubleQuantumDot()
        dqdDict = dqd.toDict()
        assert isinstance(dqdDict, dict)
        assert "acAmplitude" in dqdDict
        assert dqdDict["acAmplitude"] == 1.2

    def testGetAttributeValue(self):
        dqd = DoubleQuantumDot()
        assert dqd.getAttributeValue("acAmplitude") == 1.2
        assert np.all(dqd.getAttributeValue("gamma") == np.array([[0.01], [0.01]]))


if __name__ == "__main__":
    pytest.main()
