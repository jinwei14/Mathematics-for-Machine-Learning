# -*- coding: utf-8 -*-
"""
Use this file for your answers.

This file should been in the root of the repository
(do not move it or change the file name)

"""
# NB this is tested on python 2.7. Watch out for integer division
#import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

import autograd.numpy as np
import math
import matplotlib.pyplot as plt
from autograd import grad
# Thinly wrapped numpy

# Basically everything you need
B = np.array([[3, -1], [-1, 3]])
a = np.array([[1], [0]])
b = np.array([[0], [-1]])
def f1(x):
    """
    this the function of f3
    """
    f1 = np.transpose(x).dot(x) + np.transpose(x).dot(B).dot(x) - np.transpose(a).dot(x) + np.transpose(b).dot(x)
    return f1

def grad_f1(x):
    """
    4 marks
    :param x: input array with shape (2, )   2 row x column
    :return: the gradient of f1, with shape (2, )
    answer is  gradient f(x) = (I+B)x -a -b =0
    """
    x = x.reshape(2, 1)
    #fx1 = 2**.dot(x)  - matrix_a + matrix_b
    fx1 = 2*(np.identity(2) + B).dot(x) - a + b
    ans = fx1.reshape(2,)
    return ans

def f2(x):
    """
    this the function of f3
    """
    x=x.reshape(2,1)
    f2 = math.sin(np.transpose(x-a).dot(x-a)) + np.transpose(x-b).dot(B).dot(x-b)
    return f2

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

def f3(x):
    """
    this the function of f3
    """
    x = x.reshape(2, 1)
    part1 = np.exp(-np.dot((x - a).T, x-a))
    part2 = np.exp(-np.dot (np.dot((x - b).T, B),(x-b)))
    det = np.linalg.det(0.01 * np.identity(2) + np.dot(x, x.T))
    part3 = 0.1 * np.log(det)
    func3 = 1 - part1 - part2 + part3
    return func3

def grad_f3(x):
    """
    This question is optional. The test will still run (so you can see if you are correct by
    looking at the testResults.txt file), but the marks are for grad_f1 and grad_f2 only.
    Do not delete this function.
    :param x: input array with shape (2, )
    :return: the gradient of f3, with shape (2, )
    """
    grad_f3 = grad(f3)
    return grad_f3(x)

x= np.array([1,1])
#print(grad_f1(np.array( [ 1,-1 ])).shape )
#print(grad_f3(x))
print(f2(x))

















# From calculation, it is expected that the local minimum occurs at x=9/4

cur_x = np.array([1, -1]) # The algorithm starts at x=1
gamma = 0.1 # step size multiplier


max_iters = 50 # maximum number of iterations
iters = 0 #iteration counter


x_gd = []
y_gd = []
z_gd = []


while (iters < max_iters):
    x_gd.append(cur_x[0])
    y_gd.append(cur_x[1])
    z_gd.append(f3(cur_x))

    prev_x = np.array([cur_x[0],cur_x[1]])
    cur_x = prev_x - gamma * grad_f3(prev_x)
    iters+=1

print("The local minimum occurs at", cur_x)


xlist = np.linspace(-0.25, 1, 50)
ylist = np.linspace(-1.0, 0, 50)

X,Y = np.meshgrid(xlist,ylist)
#new an array with all 0 inside
Z = np.zeros((50, 50))

for i in range(50):
    for j in range(50):
        Z[i][j] = f3(np.array([xlist[i],ylist[j]]))

print(Z)
plt.contour(X.T,Y.T, Z, 50, cmap = 'jet')
plt.colorbar()

plt.plot(x_gd, y_gd, color='green', marker='v', linestyle='dashed', linewidth=0.4, markersize=3 )
plt.show()


# Plot the surface.

# fig = plt.figure()
# ax = fig.gca(projection='3d')
#
# surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
#                        linewidth=0, antialiased=False)
#
# # Customize the z axis.
# ax.set_zlim(-1.01, 1.01)
# ax.zaxis.set_major_locator(LinearLocator(10))
# ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
#
# # Add a color bar which maps values to colors.
# fig.colorbar(surf, shrink=0.5, aspect=5)
#
# plt.show()





