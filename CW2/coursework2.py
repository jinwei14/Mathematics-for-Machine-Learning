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




def PsiiFill(K):
    Psii = np.zeros((N, K + 1))
    for i in range(N):
        for j in range(0, K+1):
            Psii[i][j] = X[i][0]**j

    theta = np.dot(np.dot(inv(np.dot(Psii.T, Psii)), Psii.T), Y)
    return theta

#print(theta)


# def Psi(x):
#     for i in range(K + 1):
#         temp.append([x**i])
#     return np.array(temp)
#
# for r in range(1, N+1):
#     Psii[r].append(Psi(r).T)
#
# Psii = Psii.reshape(N, K)
# # θ∗ = (Φ⊤Φ)−1Φ⊤y
# theta = np.dot(np.dot(inv(np.dot(Psi.T, Psi)), Psi.T), Y)
# print(theta)

def f(x, order):
    fx = 0
    theta_tem = PsiiFill(order)
    #print(theta_tem)
    for k in range(order+1):
        fx = fx + np.dot((x**k), theta_tem[k][0])
    return fx

testingX = np.linspace(-0.3, 1.3, 200)

list_Y_0 = []
for i in testingX:
    # print(f(i))
    list_Y_0.append(f(i, 0))

list_Y_1 = []
for i in testingX:
    # print(f(i))
    list_Y_1.append(f(i, 1))

list_Y_2 = []
for i in testingX:
    # print(f(i))
    list_Y_2.append(f(i, 2))

list_Y_3 = []
for i in testingX:
    # print(f(i))
    list_Y_3.append(f(i, 3))

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
print(testingX)
plt.plot(np.linspace(0, 0.9, N), listTrainning, 'ro')
plt.text(0.2, 3, r'Pink Dot: tranning data', fontdict=fontDot)


plt.plot(testingX, list_Y_0, 'b')
plt.text(-0.2, 0.35, r'order 0', fontdict=font)

plt.plot(testingX, list_Y_1, 'g')
plt.text(1.1, -0.5, r'order 1', fontdict=font)

plt.plot(testingX, list_Y_2, 'r')
plt.text(-0.28, 3.0, r'order 2', fontdict=font)

plt.plot(testingX, list_Y_3, 'c')
plt.text(1.0, 3, r'order 3', fontdict=font)

plt.plot(testingX, list_Y_11, 'm')
plt.text(-0.1, 3.5, r'order 11', fontdict=font)

plt.axis([-0.3, 1.3, -1, 4])
plt.show()