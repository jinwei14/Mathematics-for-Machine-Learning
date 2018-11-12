# -*- coding: utf-8 -*-

"""
Use this file for your answers.

This file should been in the root of the repository
(do not move it or change the file name)

"""


import numpy as np
from numpy.linalg import inv

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
    # print(N)
    # print(M)
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
    pass

# Phi = np.array([[1,2],[1,3]])
#
# Y = np.array([[-0.75426779],[-0.5480492]])
#
# print(lml(2.0, 4.0, Phi, Y))