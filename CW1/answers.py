# -*- coding: utf-8 -*-

"""
Use this file for your answers.

This file should been in the root of the repository
(do not move it or change the file name)

"""

# NB this is tested on python 2.7. Watch out for integer division

import autograd.numpy as np  # Thinly-wrapped numpy
from autograd import grad
#import numpy as np
import math
# Thinly wrapped numpy


# Basically everything you need



B = np.array([[3, -1], [-1, 3]])
a = np.array([[1], [0]])
b = np.array([[0], [-1]])

def grad_f1(x):
    """
    4 marks


    :param x: input array with shape (2, )   2 row x column
    :return: the gradient of f1, with shape (2, )

    answer is  gradient f(x) = (I+B)x -a -b =0
    """

    #fx1 = 2**.dot(x)  - matrix_a + matrix_b

    fx1 = 2*(np.identity(2) + B).dot(x) - a + b

    #fx1 = 2*np.matmul((np.identity(2) + matrix_B), x) - matrix_a + matrix_b
    #fx1 = fx1.reshape(2, )
    ans = np.array([fx1[0,0],fx1[0,1]])
    return ans

def grad_f2(x):
    """
    6 marks

    :param x: input array with shape (2, )
    :return: the gradient of f2, with shape (2, )

    gradient f(x) = 2(x-a)Cos((x-a)T (x-a)) + 2B(x-b)
    """
    x=x.reshape(2,1)
    fx = 2*(x-a)*math.cos( np.transpose(x-a).dot(x-a))+ 2*B.dot(x-b)
    # print(fx[1])
    # ans = np.array(fx[0],fx[1])
    fx = fx.reshape(2,)
    return  fx

# f3(x) = 1− exp −(x−a)T(x−a) +exp −(x−b)TB(x−b) − 1 log  1 I+xxT
def f3(x):

    return f3

# Create a function to compute the gradient
def grad_f3(x):
    """
    This question is optional. The test will still run (so you can see if you are correct by
    looking at the testResults.txt file), but the marks are for grad_f1 and grad_f2 only.

    Do not delete this function.

    :param x: input array with shape (2, )
    :return: the gradient of f3, with shape (2, )
    """
    #x = x.reshape(2, 1)

    part1 = np.exp(-1 * np.transpose(x - a).dot(x-a))
    part2 = np.exp(-1 * np.transpose(x - b).dot(x-b))

    part3 = -(1 / 10) * np.log(abs((1 / 100) * np.identity(2) + x.dot(np.transpose(x))))

    f3 = 1 - part1 - part2 + part3 #this is the function

    grad_f3 = grad(f3)

    return grad_f3(x)



x= np.array( [ 1,-1 ])
#print(grad_f1(np.array( [ 1,-1 ])).shape )

print(grad_f2(x).shape )




