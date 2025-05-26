from examples.master_thesis.spinConservingFrame import *

tau = 0.1
chi = 0.1
socVectorTheta = np.pi / 4
socVectorPhi = np.pi / 4
kOmeFactor = 0.5
magneticField = np.array([0.0, 0.0, 1.0])
detuning = 0.8

gL = np.array([[3.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 2.0]])
gROriginal = np.array([[3.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 2.0]])

# G-tensors

axisToRotate = "Y"
convertToRad = np.pi / 180
anglesToRotate = np.array(
    [0 * convertToRad])

for angle in anglesToRotate:
    gRotated = rotateMatrix(gROriginal, angle, axisToRotate)
    gRSC, tauEff = applySpinConservingRotation(gRotated, tau, chi, socVectorPhi, socVectorTheta)

    gLEff = returnEffectiveGScalar(gL, magneticField)
    gREff = returnEffectiveGScalar(gRSC, magneticField)

    condition = spinFlipPATCondition(gLEff, gREff)

    print("Angle: ", f"{angle * 180 / np.pi:.3f}")
    print("Magnetic Field: ", magneticField)
    print("Detuning: ", detuning)
    for n in range(5):
        print(f"n={n}, Plus: ", f"{2 * (n + detuning) / condition:.3f}")
        print(f"n={n}, Minus: ", f"{2 * (n - detuning) / condition:.3f}")
    print("\n\n")
