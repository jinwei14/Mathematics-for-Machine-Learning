import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv

N = 25
X = np.reshape(np.linspace(0, 0.9, N), (N, 1))
Y = np.cos(10*X**2) + 0.1 * np.sin(100*X)
# K = 12  # order of the func
listTrainning = []
for i in np.linspace(0, 0.9, N):
    listTrainning.append(np.cos(10*i**2) + 0.1 * np.sin(100*i))



# implement the Phi function.
def Phi(K):
    Psii = np.zeros((N, 2*K + 1))
    for i in range(N):
        for j in range(0, 2*K+1):
            if j % 2 == 0:
                Psii[i][j] = np.cos(2 * np.pi * (j/2) * X[i][0])
            else:
                Psii[i][j] = np.sin(2 * np.pi * (j+1)/2 * X[i][0])

    return Psii



def lml(alpha, beta, Phi, Y):
    """
    4 marks

    :param alpha: float
    :param beta: float
    :param Phi: array of shape (N, M)
    :param Y: array of shape (N, 1)
    :return: the log marginal likelihood, a scalar
    """
    N = len(Phi)
    M = len(Phi[0])
    part1 = (-N*0.5)*np.log(2*np.pi)
    wholePhi = np.dot(np.dot(Phi, alpha*np.identity(M)),Phi.T)
    wholeBeta = beta*np.identity(N)
    part2 = - 0.5*np.log(np.linalg.det( wholePhi + wholeBeta))

    part3 = -0.5*np.dot(np.dot(Y.T, inv((wholePhi + wholeBeta))),Y)
    logFunc  = part1 + part2 + part3

    return logFunc[0][0]

def grad_lml(alpha, beta, Phi, Y):
    """
    8 marks (4 for each component)

    :param alpha: float
    :param beta: float
    :param Phi: array of shape (N, M)
    :param Y: array of shape (N, 1)
    :return: array of shape (2,). The components of this array are the gradients
    (d_lml_d_alpha, d_lml_d_beta), the gradients of lml with respect to alpha and beta respectively.

    """
    N = len(Phi)
    M = len(Phi[0])
    X = np.dot(np.dot(alpha, Phi), Phi.T) + np.dot(beta, np.identity(N))
    common = 0.5 * np.dot(np.dot(inv(X), np.dot(Y, Y.T) - X), inv(X))
    d_alpha = np.trace(np.dot(common.T, np.dot(Phi, Phi.T)))
    d_bete = np.trace(common.T)

    return np.array([d_alpha, d_bete])