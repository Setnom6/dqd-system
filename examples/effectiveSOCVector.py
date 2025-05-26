from src.base.auxiliaryMethods import *

tau = 0.1
chi = 0.1
alpha = np.array([0.0, 1.0, 0.0])

Ht = computeTunnelingHamiltonian(tau, chi, alpha)

gL = np.array([[6, 0, 0], [0, 4, 0], [0, 0, 5]])
gR = np.array([[2.92683, 0, 0.439024], [0, 5, 0], [0.439024, 0, 1.95122]])

gLRot, gRRot, HtRot = rotateGTensorsAndTunneling(gL, gR, Ht)
alpha, tau, chi = obtainAlphaTauChi(HtRot)

print("gLRot: ", gL)
print("gRRot: ", gRRot)
print("HtRot: ", HtRot)
print("alpha: ", alpha)
print("tau: ", tau)
print("chi: ", chi)
print("\n\n")
