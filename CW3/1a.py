"""
that lml(alpha, beta, Phi, Y)
that returns the log marginal likelihood, and also a function grad_lml(alpha, beta, Phi, Y)
that returns the gradient of the log marginal likelihood with respect to the vector (α,β). 2.
The function should return a numpy vector with the gradient with respect to α in the first component and gradient
with respect to β in the second.

"""

import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv

N = 25
X = np.reshape(np.linspace(0, 0.9, N), (N, 1))
Y = np.cos(10*X**2) + 0.1 * np.sin(100*X)
#K = 12  # order of the func
listTrainning = []
for i in np.linspace(0, 0.9, N):
    listTrainning.append(np.cos(10*i**2) + 0.1 * np.sin(100*i))

def Phi(K):
    Psii = np.zeros((N, K + 1))
    for i in range(N):
        for j in range(0, K+1):
            Psii[i][j] = X[i][0]**j
    return Psii

def lml(alpha, beta, Phi, Y):
    logFunc  = (-N/2)*np.log(2*np.pi) - 0.5*np.log()
    return 0


def grad_lml(alpha, beta, Phi, Y):
    return 2
