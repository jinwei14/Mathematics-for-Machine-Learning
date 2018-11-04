import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv

N = 25
X = np.reshape(np.linspace(0, 0.9, N), (N, 1))
Y = np.cos(10*X**2) + 0.1 * np.sin(100*X)
lamda = 0.01
# lamda = 1.5
# lamda = 10.0
#K = 12  # order of the func
listTrainning = []
for i in np.linspace(0, 0.9, N):
    listTrainning.append(np.cos(10*i**2) + 0.1 * np.sin(100*i))



def PsiiFill(K):
    Psii = np.zeros((N, K + 1))
    for i in range(N):
        for j in range(0, K+1):
            Psii[i][j] = np.exp((-(X[i][0]-mul(j))**2) / 2*0.01)

    theta = np.dot(np.dot(inv(np.dot(Psii.T, Psii)+lamda * np.identity(len(np.dot(Psii.T, Psii)))), Psii.T), Y)
    return theta

def mul(j):
    x= np.linspace(0, 1, 20)
    return x[j]

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

def f(x, order):
    fx = 0
    theta_tem = PsiiFill(order)
    #print(theta_tem)
    for k in range(order+1):
        fx = fx + np.dot(np.exp((-(x-mul(k))**2) / 0.02), theta_tem[k][0])
    return fx

testingX = np.linspace(-0.3, 1.3, 200)
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
fontBlue = {'family': 'serif',
        'color':  'blue',
        'weight': 'normal',
        'size': 12,
        }

# list_Y_0 = []
# for i in testingX:
# # print(f(i))
#     list_Y_0.append(f(i, 11))
# print(list_Y_0)
# plt.plot(testingX, list_Y_0)
# plt.text(-0.2, 1, r'order x', fontdict=font)
# # order 0 to 10
lsitSigmaS = []
orderList = []
for ord in range(20):
    orderList.append(ord)
    lsitSigmaS.append(sigmaSquare(ord))

print(lsitSigmaS)
plt.plot(orderList, lsitSigmaS)
plt.text(4, 0.5, r'Blue: maximum likelihood estimate', fontdict=fontBlue)


# print(testingX)
plt.plot(np.linspace(0, 0.9, N), listTrainning, 'ro')
plt.text(0.2, 3, r'Pink Dot: tranning data', fontdict=fontDot)





plt.axis([-0.3, 1.3, -1.5, 4])
plt.show()