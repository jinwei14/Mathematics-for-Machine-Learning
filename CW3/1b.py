import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv

N = 25
X = np.reshape(np.linspace(0, 0.9, N), (N, 1))
Y_train = np.cos(10*X**2) + 0.1 * np.sin(100*X)
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

#order = 1
Phi = Phi(1)
# From calculation, it is expected that the local minimum occurs at x=9/4

cur_x = np.array([0.5, 0.65]) # The algorithm starts at x=1
gamma = 0.01 # step size multiplier


max_iters = 20000 # maximum number of iterations
iters = 0 #iteration counter

x_gd = []
y_gd = []
z_gd = []
while (iters < max_iters):
    x_gd.append(cur_x[0])
    y_gd.append(cur_x[1])
    z_gd.append(lml(cur_x[0],cur_x[1], Phi, Y_train))

    prev_x = np.array([cur_x[0], cur_x[1]])
    #cur_x = prev_x + gamma * grad_f2(prev_x)
    cur_x = prev_x + gamma * grad_lml(prev_x[0], prev_x[1], Phi, Y_train)
    iters += 1

print("The local maximum occurs at", cur_x)

xlist = np.linspace(0.2, 1.0, 50)
ylist = np.linspace(0.3, 0.7, 50)

X, Y = np.meshgrid(xlist, ylist)
# new an array with all 0 inside
Z = np.zeros((50, 50))


for i in range(50):
    for j in range(50):
        #Z[i][j] = f2(np.array([xlist[i], ylist[j]]))
        Z[i][j] = lml(X[i,j], Y[i,j], Phi, Y_train)

# print(Z)
plt.contour(X, Y, Z, 100, cmap='jet')
plt.colorbar()

plt.plot(x_gd, y_gd, color='green', marker='v', linewidth=2, markersize=0)
plt.show()
#print(theta)