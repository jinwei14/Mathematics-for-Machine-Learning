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




def PsiiFill(K):
    Psii = np.zeros((N, 2*K + 1))
    for i in range(N):
        for j in range(0, 2*K+1):
            if j % 2 == 0:
                Psii[i][j] = np.cos(2 * np.pi * (j/2) * X[i][0])
            else:
                Psii[i][j] = np.sin(2 * np.pi * (j+1)/2 * X[i][0])

    theta = np.dot(np.dot(inv(np.dot(Psii.T, Psii)), Psii.T), Y)
    return theta

def f(x, order):
    fx = 0
    theta_tem = PsiiFill(order)
    #print(theta_tem)
    for k in range(2*order+1):
        if k % 2 == 0:
            fx = fx + np.dot((np.cos(2 * np.pi * (k//2) * x)), theta_tem[k][0])
        else:
            fx = fx + np.dot((np.sin(2 * np.pi * (k + 1)//2 * x)), theta_tem[k][0])

    return fx

testingX = np.linspace(-1, 1.2, 200)


list_Y_1 = []
for i in testingX:
    # print(f(i))
    list_Y_1.append(f(i, 1))


list_Y_11 = []
for i in testingX:
    # print(f(i))
    list_Y_11.append(f(i, 11))

font = {'family': 'serif',
        'color':  'black',
        'weight': 'normal',
        'size': 12,
        }

fontDot = {'family': 'serif',
        'color':  'red',
        'weight': 'normal',
        'size': 12,
        }
# print(testingX)
plt.plot(np.linspace(0, 0.9, N), listTrainning, 'ro')
plt.text(0.2, 1.5, r'Pink Dot: tranning data', fontdict=fontDot)


plt.plot(testingX, list_Y_1, 'g')
plt.text(0.9, -1.0, r'order 1:green', fontdict=font)


plt.plot(testingX, list_Y_11, 'm')
plt.text(-0.2, -1.0, r'order 11: magenta', fontdict=font)

plt.axis([-0.3, 1.3, -2, 2])
plt.show()