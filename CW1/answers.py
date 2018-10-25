# -*- coding: utf-8 -*-

"""
Use this file for your answers.

This file should been in the root of the repository
(do not move it or change the file name)

"""

# NB this is tested on python 2.7. Watch out for integer division

import numpy as np


def grad_f1(x):
    """
    4 marks


    :param x: input array with shape (2, )   2 row x column
    :return: the gradient of f1, with shape (2, )

    answer is  gradient f(x) = (I+B)x -a -b =0
    """
    matrix_B = np.array( [[3,-1],[-1,3]] )
    matrix_a =np.array( [[1],[0]] )
    matrix_b = np.array( [[0],[1]])
    #fx1 = 2**.dot(x)  - matrix_a + matrix_b

    fx1 = 2*(np.identity(2) + matrix_B).dot(x) - matrix_a + matrix_b

    return fx1

def grad_f2(x):
    """
    6 marks

    :param x: input array with shape (2, )
    :return: the gradient of f2, with shape (2, )
    """
    pass


def grad_f3(x):
    """
    This question is optional. The test will still run (so you can see if you are correct by
    looking at the testResults.txt file), but the marks are for grad_f1 and grad_f2 only.

    Do not delete this function.

    :param x: input array with shape (2, )
    :return: the gradient of f3, with shape (2, )
    """
    pass




print(grad_f1(np.array( [ [3],[-1] ])))