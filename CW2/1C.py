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







# list_Y_1 = []
# for i in testingX:
#     # print(f(i))
#     list_Y_1.append(f(i, 1))
# print(list_Y_1)
#
#
# list_Y_11 = []
# for i in testingX:
#     # print(f(i))
#     list_Y_11.append(f(i, 11))



def sigmaSquare(order):
    trainning_Y = []
    for i in X:
        # print(f(i))
        trainning_Y.append(f(i, order))
    S = 0
    # order 0 to 10
    for i in range(1, N+1):
        S = S + (Y[i-1] - trainning_Y[i-1])**2

    return S/N

# # order 0 to 10
lsitSigmaS = []
orderList = []
for ord in range(11):
    orderList.append(ord)
    lsitSigmaS.append(sigmaSquare(ord))

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
# plt.plot(np.linspace(0, 0.9, N), listTrainning, 'ro')
# plt.text(0.2, 1.5, r'Pink Dot: tranning data', fontdict=fontDot)
#
#

print(lsitSigmaS)
plt.plot(orderList, lsitSigmaS)
plt.title('maximum likelihood value for $\sigma_ML^2$', fontdict=font)
plt.xlabel('Order', fontdict=font)
plt.ylabel(r'$\sigma_ML^2$', fontdict=font)


# plt.plot(testingX, list_Y_11, 'm')
# plt.text(-0.2, -1.0, r'order 11: magenta', fontdict=font)

# plt.axis([-0.3, 1.3, -2, 2])
plt.show()