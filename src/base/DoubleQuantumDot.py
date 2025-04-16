import numpy as np
from typing import Dict, Any, List, Tuple
from qutip import *
from enum import Enum

class DQDAttributes(Enum):
    AC_AMPLITUDE = "acAmplitude"
    CHI = "chi"
    TAU = "tau"
    GAMMA = "gamma"
    ZEEMAN = "zeeman"
    MAGNETIC_FIELD = "magneticField"
    G_FACTOR = "gFactor"
    FACTOR_OME = "factorBetweenOMEAndZeeman"
    ALPHA_THETA_ANGLE = "alphaThetaAngle"
    ALPHA_PHI_ANGLE = "alphaPhiAngle"
    DETUNING = "detuning"
    GROUND_RIGHT_ENERGY = "groundRightEnergy"

class DoubleQuantumDot:
    """Class representing a Double Quantum Dot with defined parameters.

    Frecuency of the driving is considered undetermined. Quantities are given in terms of this driving frequency.
    """

    acAmplitude: float
    chi: float
    tau: float
    gamma: np.ndarray
    zeeman: np.ndarray
    magneticField: np.ndarray
    gFactor: np.ndarray
    factorBetweenOMEAndZeeman: float
    alphaThetaAngle: float
    alphaPhiAngle: float
    OME: np.ndarray
    detuning: float
    groundRightEnergy: float
    sumCurrent: float
    polarity: float

    def __init__(self, parameters: Dict[str, Any] = None) -> None:
        self._initializeDefaultParameters()
        if parameters is not None:
            self.setParameters(parameters)

        self.sumCurrent = None
        self.polarity = None

    def _initializeDefaultParameters(self) -> None:
        """Initialize default parameters for the Double Quantum Dot system."""
        self.acAmplitude = 1.2
        self.chi = 0.1
        self.tau = 0.1
        self.groundRightEnergy = 1.0
        self.detuning = 0.0
        self.gamma = np.zeros((2,1))
        self.gamma[0][0] = 0.01
        self.gamma[1][0] = 0.01
        self.zeeman = np.zeros((2,3))
        self.gFactor = np.zeros((2,3,3))
        self.gFactor[0] = np.identity(3)
        self.gFactor[1] = np.identity(3)
        self.magneticField = np.zeros((3,))
        self.OME = np.zeros((2,3))
        self.factorBetweenOMEAndZeeman = 0.5
        self.alphaThetaAngle = np.pi/2
        self.alphaPhiAngle = 0.0

    def setParameters(self, parameters: Dict[str, Any]) -> None:
        for key, value in parameters.items():
            if hasattr(self, key):
                self._validateParameters(key, value)  # Validate the parameter
                attr = getattr(self, key)
                if isinstance(value, type(attr)):
                    setattr(self, key, value)
                else:
                    raise AttributeError(
                        f"Type mismatch for attribute '{key}': expected {type(attr).__name__}, got {type(value).__name__}")
            else:
                raise AttributeError(f"'{key}' is not a valid attribute")

            if key in ["gFactor", "magneticField"]:
                self.zeeman[0] = self.gFactor[0] @ self.magneticField
                self.zeeman[1] = self.gFactor[1] @ self.magneticField

        if parameters is not None or parameters is not {}:
            self.sumCurrent = None
            self.polarity = None

    def _validateParameters(self, key: str, value: Any) -> None:
        """Validate the parameters being set to ensure correctness."""
        if key == "gamma":
            if not isinstance(value, np.ndarray) or value.shape != (2, 1):
                raise ValueError(
                    f"'gamma' must be a numpy array with shape (2, 1), got {value.shape if isinstance(value, np.ndarray) else type(value)}")
        elif key == "zeeman":
            if not isinstance(value, np.ndarray) or value.shape != (2, 3):
                raise ValueError(
                    f"'zeeman' must be a numpy array with shape (2, 3), got {value.shape if isinstance(value, np.ndarray) else type(value)}")
        elif key == "gFactor":
            if not isinstance(value, np.ndarray) or value.shape != (2, 3, 3):
                raise ValueError(
                    f"'gFactor' must be a numpy array with shape (2, 3, 3), got {value.shape if isinstance(value, np.ndarray) else type(value)}")
        elif key == "magneticField":
            if not isinstance(value, np.ndarray) or value.shape != (3,):
                raise ValueError(
                    f"'magneticField' must be a numpy array with shape (3,), got {value.shape if isinstance(value, np.ndarray) else type(value)}")
        elif key == "OME":
            if not isinstance(value, np.ndarray) or value.shape != (2, 3):
                raise ValueError(
                    f"'OME' must be a numpy array with shape (2, 3), got {value.shape if isinstance(value, np.ndarray) else type(value)}")
        elif key in ["acAmplitude", "chi", "tau", "factorBetweenOMEAndZeeman", "alphaThetaAngle", "alphaPhiAngle",
                     "detuning", "groundRightEnergy"]:
            if not isinstance(value, (int, float)):
                raise ValueError(f"'{key}' must be a numeric value, got {type(value)}")
            if key in ["acAmplitude", "chi", "tau", "factorBetweenOMEAndZeeman"] and value < 0:
                raise ValueError(f"'{key}' must be non-negative, got {value}")

    def _timeIndependentHamiltonian(self) -> np.ndarray:
        """Calculate the time-independent Hamiltonian as a numpy matrix."""
        H0 = np.zeros((5, 5), dtype=complex)

        groundLeftEnergy = self.groundRightEnergy - self.detuning
        tau0 = self.tau - self.tau * self.chi
        tauSFModule = self.tau * self.chi
        thetaAngle= self.alphaThetaAngle
        phiAngle = self.alphaPhiAngle
        tauXSF = tauSFModule* np.cos(thetaAngle)*np.cos(phiAngle)
        tauYSF = tauSFModule * np.sin(thetaAngle)*np.cos(phiAngle)
        tauZSF = tauSFModule*np.sin(phiAngle)

        H0[1, 1] = groundLeftEnergy + self.zeeman[0][2]/2
        H0[1, 2] = self.zeeman[0][0] / 2 - 1j * self.zeeman[0][1]/ 2
        H0[1, 3] = -tau0 -1j*tauZSF
        H0[1, 4] = -tauYSF - 1j*tauXSF

        H0[2, 1] = self.zeeman[0][0] / 2 + 1j * self.zeeman[0][1]/ 2
        H0[2, 2] = groundLeftEnergy - self.zeeman[0][2] / 2
        H0[2, 3] = tauYSF - 1j*tauXSF
        H0[2, 4] = -tau0 + 1j*tauZSF

        H0[3, 1] = -tau0+1j*tauZSF
        H0[3, 2] = tauYSF +1j*tauXSF
        H0[3, 3] = self.groundRightEnergy + self.zeeman[1][2] / 2
        H0[3, 4] = self.zeeman[1][0] / 2 - 1j * self.zeeman[1][1] / 2

        H0[4, 1] = -tauYSF + 1j*tauXSF
        H0[4, 2] = -tau0-1j*tauZSF
        H0[4, 3] = self.zeeman[1][0] / 2 + 1j * self.zeeman[1][1] / 2
        H0[4, 4] = self.groundRightEnergy - self.zeeman[1][2] / 2

        return H0

    def _oscillatoryHamiltonian(self) -> np.ndarray:
        """Calculate the oscillatory Hamiltonian a numpy matrix proportional to cos(omega t)."""
        H1 = np.zeros((5, 5), dtype=complex)
        self._calculateOMEFromZeeman()

        # As the SOC vector is directed into axis y, the Ey does not contribute to the OME terms
        H1[1, 1] = self.acAmplitude + self.OME[0][2] / 2
        H1[2, 2] = self.acAmplitude - self.OME[0][2] / 2
        H1[1, 2] = self.OME[0][0] / 2 - 1j*self.OME[0][1] / 2
        H1[2, 1] = self.OME[0][0] / 2 + 1j*self.OME[0][1] / 2
        H1[3, 3] = self.OME[1][2] / 2
        H1[4, 4] = -self.OME[1][2] / 2
        H1[3, 4] = self.OME[1][0] / 2 - 1j*self.OME[1][1]/ 2
        H1[4, 3] = self.OME[1][0] / 2 + 1j*self.OME[1][1] / 2

        return H1

    def _calculateOMEFromZeeman(self) -> None:
        """Calculate the OME from the Zeeman effect as a propotionality factor of its correspondent Zeeman component.
            The OME terms are supposed to be factor*(-Zz)*alphay*sigmax, factor*(Zz)*alphax*sigmay, factor*(-Zy*alphax+Zx*alphay)*sigmaz
        """

        alphax = self.factorBetweenOMEAndZeeman*np.cos(self.alphaThetaAngle)*np.cos(self.alphaPhiAngle)
        alphay = self.factorBetweenOMEAndZeeman*np.sin(self.alphaThetaAngle)*np.cos(self.alphaPhiAngle)
        alphaz = self.factorBetweenOMEAndZeeman*np.sin(self.alphaPhiAngle)

        self.OME[0][0] = -alphay *self.zeeman[0][2] +alphaz*self.zeeman[0][1]
        self.OME[0][1] = alphax * self.zeeman[0][2] -alphaz*self.zeeman[0][0]
        self.OME[0][2] = alphay * self.zeeman[0][0] -alphax * self.zeeman[0][1]

        self.OME[1][0] = -alphay * self.zeeman[1][2] + alphaz * self.zeeman[1][1]
        self.OME[1][1] = alphax * self.zeeman[1][2] - alphaz * self.zeeman[1][0]
        self.OME[1][2] = alphay * self.zeeman[1][0] -alphax * self.zeeman[1][1]

    def _collapseOperators(self) -> List[np.ndarray]:
        """Generate the collapse operators for the system."""
        kappaComponents = [np.zeros((5, 5), dtype=complex) for i in range(4)]

        # Left lead to left spin up dot (creation operator)
        kappaComponents[0][1, 0] = np.sqrt(self.gamma[0][0])

        # Left lead to left spin down dot (creation operator)
        kappaComponents[1][2, 0] = np.sqrt(self.gamma[0][0])

        # Right lead to right spin up dot (destruction operator)
        kappaComponents[2][0, 3] = np.sqrt(self.gamma[1][0])

        # Right lead to right spin down dot (destruction operator)
        kappaComponents[3][0, 4] = np.sqrt(self.gamma[1][0])

        return kappaComponents

    def _getChargeObservables(self) -> List[np.ndarray]:
        """Generate the charge observables for the system."""
        chargeObservables = [np.zeros((5, 5), dtype=complex) for i in range(5)]

        for i in range(5):
            chargeObservables[i][i, i] = 1

        return chargeObservables

    def computeCurrent(self, iterationsPerPeriod = 20) -> None:
        """Run the simulation for the Double Quantum Dot system."""
        T = 2 * np.pi
        H0 = Qobj(self._timeIndependentHamiltonian())
        H1 = Qobj(self._oscillatoryHamiltonian())
        H = [H0, [H1, 'cos(t)']]

        collapseOperatorsList = self._collapseOperators()
        collapseOperatorsList = [Qobj(kappa) for kappa in collapseOperatorsList]

        observables = self._getChargeObservables()
        observables = [Qobj(observable) for observable in observables]

        propagatorU = propagator(H, T, collapseOperatorsList)
        rhoSS = propagator_steadystate(propagatorU)

        times = np.linspace(0, T, int(iterationsPerPeriod))

        results = mesolve(H, rhoSS, times, c_ops=collapseOperatorsList, e_ops=observables)

        spinUp = np.mean(results.expect[3])
        spinDown = np.mean(results.expect[4])
        self.sumCurrent = spinUp + spinDown
        self.polarity = (spinUp - spinDown) / (spinUp + spinDown)
        return None

    def getCurrent(self,  iterationsPerPeriod = 20) -> Tuple[float, float]:
        if self.sumCurrent is None or self.polarity is None:
            self.computeCurrent(iterationsPerPeriod)
        return self.sumCurrent, self.polarity

    def toDict(self) -> Dict[str, Any]:
        dictToReturn = {
            key: (value.tolist() if isinstance(value, np.ndarray)
                  else int(value) if isinstance(value, np.integer)
            else float(value) if isinstance(value, np.floating)
            else value)
            for key, value in self.__dict__.items()
            if not key.startswith("_")
        }
        return dictToReturn

    def getAttributeValue(self, key: str):
        """
        Get the value of an attribute by its key.

        Args:
            key (str): The name of the attribute.

        Returns:
            The value of the attribute. If the value is mutable (e.g., an array), a copy is returned.
        """
        value = getattr(self, key)
        # Check if the value is mutable (e.g., an array) and return a copy if possible
        if isinstance(value, (np.ndarray, list)):
            return value.copy()
        # For scalar values, return them directly
        return value