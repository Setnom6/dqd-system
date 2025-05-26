import numpy as np


def rotateMatrix(matrix, theta, axis):
    """
    Method to misalign a g-tensor by rotating it aroung the specified axis with the specified angle.
    """
    axis = axis.upper()
    if axis == 'X':
        rotationMatrix = np.array([
            [1, 0, 0],
            [0, np.cos(theta), -np.sin(theta)],
            [0, np.sin(theta), np.cos(theta)]
        ])
    elif axis == 'Y':
        rotationMatrix = np.array([
            [np.cos(theta), 0, np.sin(theta)],
            [0, 1, 0],
            [-np.sin(theta), 0, np.cos(theta)]
        ])
    elif axis == 'Z':
        rotationMatrix = np.array([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1]
        ])
    else:
        raise ValueError("Axis must be 'X', 'Y', or 'Z'.")

    # Rotación activa del tensor en el espacio
    return rotationMatrix @ matrix @ rotationMatrix.T


def applySpinConservingRotation(matrix, tau, chi, socTheta, socPhi):
    """
    Given the total tau = t0 + tsf, chi=1/(1+t0/tsf) and the recipe to build the SOC vector, it rotates the given gR-tensor
    to the spin conserving frame with respect to the gL-tensor. It returns also the total effective tau0 for the tunneling matrix
    """
    t0 = tau * (1 - chi)
    tsf = tau * chi
    tx = tsf * np.cos(socTheta) * np.cos(socPhi)
    ty = tsf * np.sin(socTheta) * np.cos(socPhi)
    tz = tsf * np.sin(socPhi)

    modtSquared = t0 ** 2 + tx ** 2 + ty ** 2 + tz ** 2
    tEffective = np.sqrt(modtSquared)

    identity = np.identity(3)

    # Simétrica: 2 * outer(t, t)
    outerTT = 2 * np.array([
        [tx * tx, tx * ty, tx * tz],
        [ty * tx, ty * ty, ty * tz],
        [tz * tx, tz * ty, tz * tz]
    ])

    # Antisimétrica: 2 * t0 * cross-product matrix
    antisymmetric = 2 * t0 * np.array([
        [0, tz, -ty],
        [-tz, 0, tx],
        [ty, -tx, 0]
    ])

    R = (1 / modtSquared) * ((t0 ** 2 - (tx ** 2 + ty ** 2 + tz ** 2)) * identity + outerTT - antisymmetric)

    Rinv = np.linalg.inv(R)

    return Rinv @ matrix, tEffective


def computeGeometricalClass(gLeft: np.ndarray, gRightRotated: np.ndarray) -> str:
    """
    Computes the geometrical class of the system based on the eigenvalues of the matrix M.

    Parameters
    ----------
    gLeft : np.ndarray, shape (3, 3)
        The left g-tensor matrix.
    gRightRotated : np.ndarray, shape (3, 3)
        The rotated right g-tensor matrix.

    Returns
    -------
    str
        The geometrical class of the system:
        - "A": Two complex eigenvalues and one real positive eigenvalue.
        - "B": Two real negative eigenvalues and one real positive eigenvalue.
        - "C": Three real positive eigenvalues.
        - "D": Two complex eigenvalues and one real negative eigenvalue.
        - "E": Three real negative eigenvalues.
        - "F": Two real positive eigenvalues and one real negative eigenvalue.

    Raises
    ------
    ValueError
        If gRightRotated is not invertible or M does not match any of the specified configurations.
    """
    try:
        # Step 1: Compute M = Inverse[gRightRotated] · gLeft
        gRightRotatedInv = np.linalg.inv(gRightRotated)
        M = gRightRotatedInv @ gLeft
    except np.linalg.LinAlgError:
        raise ValueError("gRightRotated is not invertible.")

    # Step 2: Compute the eigenvalues of M
    eigenvalues = np.linalg.eigvals(M)

    # Step 3: Classify the eigenvalues
    countPositive = 0
    countNegative = 0
    countComplex = 0

    for eigenvalue in eigenvalues:
        if np.iscomplex(eigenvalue):
            countComplex += 1
        else:
            if eigenvalue > 0:
                countPositive += 1
            elif eigenvalue < 0:
                countNegative += 1

    # Step 4: Determine the geometrical class
    if countComplex == 2 and countPositive == 1:
        return "A"
    elif countComplex == 0 and countNegative == 2 and countPositive == 1:
        return "B"
    elif countComplex == 0 and countPositive == 3:
        return "C"
    elif countComplex == 2 and countNegative == 1:
        return "D"
    elif countComplex == 0 and countNegative == 3:
        return "E"
    elif countComplex == 0 and countPositive == 2 and countNegative == 1:
        return "F"
    else:
        raise ValueError("M does not match any of the specified configurations.")


def returnEffectiveGScalar(gTensor, magneticField):
    """
    It returns the effective gTensor in the given magnetic field direction.
    """
    return np.linalg.norm(gTensor @ magneticField) / np.linalg.norm(magneticField)


def spinConservingPATCondition(gLEff, gREff):
    """
    It computes the difference in energy between the two zeeman vectors in the direction of the magnetic field,
    given by the effective gtensores.
    """
    return abs(gREff - gLEff)


def spinFlipPATCondition(gLEff, gREff):
    """
        It computes the sum in energy between the two zeeman vectors in the direction of the magnetic field,
        given by the effective gtensores.
        """
    return abs(gREff + gLEff)


"""TO organize"""


def computeTunnelingHamiltonian(tau: float, chi: float, alpha: np.ndarray) -> np.ndarray:
    """
    Compute the tunneling Hamiltonian for a given set of parameters.

    Parameters
    ----------
    tau : float
        Scalar quantity representing the tunneling amplitude.
    chi : float
        Dimensionless parameter in [0, 1], defined as ||tsf|| / (t0 + ||tsf||).
    alpha : np.ndarray, shape (3,)
        Unit vector indicating the spatial direction of the tunneling.

    Returns
    -------
    np.ndarray, shape (4, 4)
        The tunneling Hamiltonian matrix.

    Raises
    ------
    ValueError
        If alpha is not a 3-component vector or is the zero vector.
    """
    alpha = np.array(alpha, dtype=float)
    if alpha.shape != (3,):
        raise ValueError("alpha must be a 3-component vector.")
    normAlpha = np.linalg.norm(alpha)
    if normAlpha == 0:
        raise ValueError("alpha cannot be the zero vector.")
    alpha = alpha / normAlpha

    # Compute t0 and tsf from tau and chi
    t0 = tau * (1 - chi)
    tsf = tau * chi

    # Compute tsfVec in the direction of alpha
    tx, ty, tz = tsf * alpha

    # Construct the tunneling Hamiltonian
    Ht = np.array([
        [0, 0, t0 - 1j * tz, -1j * (tx - 1j * ty)],
        [0, 0, -1j * (tx + 1j * ty), t0 + 1j * tz],
        [t0 + 1j * tz, 1j * (tx - 1j * ty), 0, 0],
        [1j * (tx + 1j * ty), t0 - 1j * tz, 0, 0]
    ], dtype=complex)

    return Ht


def obtainTunnelingComponents(Ht: np.ndarray) -> tuple:
    """
    Extract tunneling components (t0, tx, ty, tz) from the tunneling Hamiltonian.

    Parameters
    ----------
    Ht : np.ndarray, shape (4, 4)
        The tunneling Hamiltonian matrix.

    Returns
    -------
    tuple
        A tuple containing the tunneling components (t0, tx, ty, tz).
    """

    # Check if Ht is Hermitian
    if not np.allclose(Ht, Ht.conj().T, atol=1e-10):
        raise ValueError("The tunneling Hamiltonian Ht must be Hermitian.")

    t0 = Ht[0, 2].real
    tz = Ht[1, 3].imag
    ty = Ht[1, 2].real
    tx = Ht[2, 1].imag
    return t0, tx, ty, tz


def obtainRotationMatrices(Ht: np.ndarray) -> tuple:
    """
    Compute the rotation matrices R and W from the tunneling Hamiltonian.

    Parameters
    ----------
    Ht : np.ndarray, shape (4, 4)
        The tunneling Hamiltonian matrix.

    Returns
    -------
    tuple
        A tuple containing the rotation matrix R (3x3) and the spinor rotation matrix W (4x4).
    """
    t0, tx, ty, tz = obtainTunnelingComponents(Ht)
    tsfVec = np.array([tx, ty, tz])
    modt = np.sqrt(t0 ** 2 + tx ** 2 + ty ** 2 + tz ** 2)

    # Construct the spinor rotation matrix W
    W = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, (t0 + 1j * tz) / modt, (ty + 1j * tx) / modt],
        [0, 0, (-ty + 1j * tx) / modt, (t0 - 1j * tz) / modt]
    ], dtype=complex)

    # Construct the SO(3) rotation matrix R
    delta = np.identity(3)
    tVecSquared = np.dot(tsfVec, tsfVec)
    denom = t0 ** 2 + tVecSquared
    R = np.zeros((3, 3))

    epsilon = np.zeros((3, 3, 3))
    epsilon[0, 1, 2] = epsilon[1, 2, 0] = epsilon[2, 0, 1] = 1
    epsilon[2, 1, 0] = epsilon[0, 2, 1] = epsilon[1, 0, 2] = -1

    for i in range(3):
        for j in range(3):
            term1 = (t0 ** 2 - tVecSquared) * delta[i, j]
            term2 = 2 * tsfVec[i] * tsfVec[j]
            term3 = -2 * t0 * sum(epsilon[i, j, k] * tsfVec[k] for k in range(3))
            R[i, j] = (term1 + term2 + term3) / denom

    return R, W


def computePseudoSpinConservingGTensors(gL: np.ndarray, gR: np.ndarray, Ht: np.ndarray) -> tuple:
    """
    Compute the transformed g-tensors in the pseudo-spin conserving gauge.

    Parameters
    ----------
    gL : np.ndarray, shape (3, 3)
        Real 3x3 matrix with non-zero eigenvalues (left matrix).
    gR : np.ndarray, shape (3, 3)
        Real 3x3 matrix with non-zero eigenvalues (right matrix).
    Ht : np.ndarray, shape (4, 4)
        The tunneling Hamiltonian matrix.

    Returns
    -------
    tuple
        A tuple containing:
        - gL : np.ndarray, shape (3, 3)
            The original left matrix.
        - gRRot : np.ndarray, shape (3, 3)
            Rotated version of the right matrix, gRRot = R^{-1} · gR.
        - HtRot : np.ndarray, shape (4, 4)
            Rotated tunneling Hamiltonian.
        - geometricClass : int
            Geometric class of the system based on eigenvalue analysis.

    Raises
    ------
    ValueError
        If gL or gR is not 3x3, has zero eigenvalues, or if alpha is invalid.
    """
    gL = np.array(gL, dtype=float)
    gR = np.array(gR, dtype=float)

    if gL.shape != (3, 3) or gR.shape != (3, 3):
        raise ValueError("gL and gR must be 3x3 matrices.")

    eigvalsGL = np.linalg.eigvals(gL)
    eigvalsGR = np.linalg.eigvals(gR)

    if np.any(np.isclose(eigvalsGL, 0)) or np.any(np.isclose(eigvalsGR, 0)):
        raise ValueError("gL and gR must have all non-zero eigenvalues.")

    R, W = obtainRotationMatrices(Ht)

    gRRot = np.linalg.inv(R) @ gR
    HtRot = W.conj().T @ Ht @ W

    _, _, _, geometricClass = obtainGeometricClass(gL, gRRot)

    return gL, gRRot, HtRot, geometricClass


def obtainGeometricClass(gL, gRRot):
    """
    Determine the geometric class of the system based on eigenvalue analysis.

    Parameters
    ----------
    gL : ndarray, shape (3, 3)
        Left g-tensor matrix.
    gRRot : ndarray, shape (3, 3)
        Rotated right g-tensor matrix.

    Returns
    -------
    M : ndarray
        Matrix product of gRRot^{-1} and gL.
    eigenvaluesM : ndarray
        Eigenvalues of matrix M.
    eigenvectorsM : ndarray
        Eigenvectors of matrix M.
    geometricClass : int
        Geometric class of the system.

    Raises
    ------
    ValueError
        If gRRot is not invertible.
    """
    try:
        gRRotInv = np.linalg.inv(gRRot)
    except np.linalg.LinAlgError:
        raise ValueError("gRRot is not invertible.")

    M = gRRotInv @ gL
    eigenvaluesM, eigenvectorsM = np.linalg.eig(M)

    countPositive = 0
    countNegative = 0
    countComplex = 0
    geometricClass = None
    for eigenvalue in eigenvaluesM:
        if not isinstance(eigenvalue, complex):
            eigenvalue = complex(eigenvalue)
        if eigenvalue.imag != 0:
            countComplex += 1
        else:
            if eigenvalue.real > 0:
                countPositive += 1
            else:
                countNegative += 1

    if countComplex > 0:
        if countPositive == 1 and countNegative == 0:
            geometricClass = 0
        elif countPositive == 0 and countNegative == 1:
            geometricClass = 3
    else:
        if countPositive == 3 and countNegative == 0:
            geometricClass = 2
        elif countPositive == 2 and countNegative == 1:
            geometricClass = 5
        elif countPositive == 1 and countNegative == 2:
            geometricClass = 1
        elif countPositive == 0 and countNegative == 3:
            geometricClass = 4

    return M, eigenvaluesM, eigenvectorsM, geometricClass


def su2_from_so3(R: np.ndarray) -> np.ndarray:
    """
    Dada una rotación R ∈ SO(3), obtiene una matriz U ∈ SU(2) tal que
    U σ_i U^† = ∑_j R_ij σ_j
    """
    theta = np.arccos((np.trace(R) - 1) / 2)
    if np.isclose(theta, 0):
        return np.eye(2)

    rx = R[2, 1] - R[1, 2]
    ry = R[0, 2] - R[2, 0]
    rz = R[1, 0] - R[0, 1]
    r = np.array([rx, ry, rz])
    r = r / (2 * np.sin(theta))

    sx = np.array([[0, 1], [1, 0]], dtype=complex)
    sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sz = np.array([[1, 0], [0, -1]], dtype=complex)
    n = r[0] * sx + r[1] * sy + r[2] * sz

    U = expm(-1j * theta / 2 * n)
    return U


def build_spinor_rotation(U: np.ndarray) -> np.ndarray:
    """
    Extiende U ∈ SU(2) a una matriz W ∈ U(4) que actúa como identidad en L y como U en R.
    """
    W = np.eye(4, dtype=complex)
    W[2:4, 2:4] = U
    return W


def reconstructPhysicalParameters(gL: np.ndarray, gRRot: np.ndarray, HtRot: np.ndarray):
    """
    Reconstruye (gL, gR, Ht, tau, chi, alpha) a partir de gRRot y HtRot
    """

    gL = np.array(gL, dtype=float)
    gRRot = np.array(gRRot, dtype=float)
    HtRot = np.array(HtRot, dtype=complex)

    if gL.shape != (3, 3) or gRRot.shape != (3, 3):
        raise ValueError("gL y gRRot deben ser matrices 3x3.")

    eigvals = np.linalg.eigvals(gRRot)
    if np.any(np.isclose(eigvals, 0)):
        raise ValueError("gRRot no debe tener autovalores nulos.")

    # 1. Diagonalizamos gRRot: R @ gRRot = gR
    eigvals, R = np.linalg.eig(gRRot)
    gR = np.linalg.matrix_transpose(R) @ gRRot

    # 2. Construimos W a partir de R
    U = su2_from_so3(R)
    W = build_spinor_rotation(U)

    # 3. Desrotamos HtRot
    Ht = W.conj().T @ HtRot @ W

    # 4. Obtenemos componentes del tunneling
    t0, tx, ty, tz = obtainTunnelingComponents(Ht)
    tau = np.sqrt(t0 ** 2 + tx ** 2 + ty ** 2 + tz ** 2)
    tsf = np.sqrt(tx ** 2 + ty ** 2 + tz ** 2)
    chi = tsf / (t0 + tsf)

    if tsf == 0:
        alpha = np.array([0, 0, 0])
    else:
        alpha = np.array([tx, ty, tz]) / tsf

    return gL, gR, Ht, tau, chi, alpha


def recover_gR_and_Ht(gRRot, HtRot):
    # Paso 1: Usar SVD para obtener la rotación R
    U, S, Vh = np.linalg.svd(gRRot)
    R = U @ Vh
    gR = R @ gRRot  # Esto será diagonal si R es correctamente ortogonal

    # Asegurar que R es una rotación pura (det=1), no una reflexión
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vh
        gR = R @ gRRot

    # Paso 2: Convertir R ∈ SO(3) a W_R ∈ SU(2)
    rotation = Rot.from_matrix(R)
    x, y, z, w = rotation.as_quat()
    W_R = np.array([
        [w + 1j * z, y + 1j * x],
        [-y + 1j * x, w - 1j * z]
    ])
    W_R /= np.linalg.det(W_R) ** 0.5  # Asegura que W_R ∈ SU(2)

    # Paso 3: Construir W = I_2 ⊕ W_R
    W = np.block([
        [np.eye(2), np.zeros((2, 2))],
        [np.zeros((2, 2)), W_R]
    ])

    # Paso 4: Desrotar HtRot
    Ht = W @ HtRot @ W.conj().T

    return gR, Ht


def obtainAlphaTauChi(Ht):
    t0, tx, ty, tz = obtainTunnelingComponents(Ht)
    tsfVec = np.array([tx, ty, tz])
    modtsf = np.linalg.norm(tsfVec)
    alpha = tsfVec / modtsf
    tau = t0 + modtsf
    chi = modtsf / tau
    return alpha, tau, chi


import numpy as np


def rotationMatrixToAxisAngle(R):
    angle = np.arccos((np.trace(R) - 1) / 2)
    if np.isclose(angle, 0):
        return np.array([1.0, 0.0, 0.0]), 0.0
    elif np.isclose(angle, np.pi):
        eigvals, eigvecs = np.linalg.eigh((R + np.eye(3)) / 2)
        axis = eigvecs[:, np.argmax(eigvals)]
        return axis, angle
    else:
        axis = np.array([
            R[2, 1] - R[1, 2],
            R[0, 2] - R[2, 0],
            R[1, 0] - R[0, 1]
        ]) / (2 * np.sin(angle))
        return axis, angle


def su2FromAxisAngle(axis, angle):
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    c = np.cos(angle / 2)
    s = np.sin(angle / 2)
    return np.array([
        [c - 1j * z * s, -1j * (x - 1j * y) * s],
        [-1j * (x + 1j * y) * s, c + 1j * z * s]
    ])


def safeDiagonalization(g):
    """Diagonaliza sin reordenar ejes, y asegura matriz de rotación en SO(3)."""
    eigvals, eigvecs = np.linalg.eig(g)
    if np.linalg.det(eigvecs) < 0:
        eigvecs[:, 0] *= -1  # Corrige signo si está en O(3) pero no SO(3)
    gRot = eigvecs.T @ g @ eigvecs
    return gRot, eigvecs


def rotateGTensorsAndTunneling(gLeft, gRight, tunnelingHamiltonian):
    gLeftRot, rotL = safeDiagonalization(gLeft)
    gRightRot, rotR = safeDiagonalization(gRight)

    axisL, angleL = rotationMatrixToAxisAngle(rotL)
    axisR, angleR = rotationMatrixToAxisAngle(rotR)
    wL = su2FromAxisAngle(axisL, angleL)
    wR = su2FromAxisAngle(axisR, angleR)

    uTotal = np.block([
        [wL.conj().T, np.zeros((2, 2))],
        [np.zeros((2, 2)), wR.conj().T]
    ])

    tunnelingHamiltonianRotated = uTotal @ tunnelingHamiltonian @ uTotal.conj().T

    return gLeftRot, gRightRot, tunnelingHamiltonianRotated
