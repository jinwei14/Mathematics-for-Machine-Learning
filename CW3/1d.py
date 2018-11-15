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
        Psii[i][0] = 1
        for j in range(1, K+1):
            Psii[i][j] = np.exp((-(X[i][0]-mul(j-1))**2) / 0.02)

    return Psii

def mul(j):
    x = np.linspace(-0.5, 1, 10)
    return x[j]


X_test = np.reshape(np.linspace(-1, 1.5, 200), (200, 1))
def PhiTest(K):
    Psii = np.zeros((200, K + 1))
    for i in range(200):
        Psii[i][0] = 1
        for j in range(1, K+1):
            Psii[i][j] = np.exp((-(X_test[i][0]-mul(j-1))**2) / 0.02)

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
Phi = Phi(10)
print(Phi.shape)
alpha = 1.0
beta = 0.1
S_n = (inv(alpha*np.identity(len(Phi.T)) + (1/beta)*np.dot(Phi.T, Phi)))
# print(S_n.shape)
M_n = np.dot(S_n, (1/beta)*np.dot(Phi.T,  Y_train)).reshape(1,11)
# print(M_n.shape)
samples = np.random.multivariate_normal(M_n[0], S_n, 5)
# print(samples.shape)




Phi_test = PhiTest(10)
print(Phi_test.shape)

index = 1
# ---------------Predictive mean------------------
for sample in samples:
    print(index)
    mean = np.dot(Phi_test, sample)
    # print(sample.shape)
    plt.plot(X_test, mean, label='sample'+str(index))
    index += 1

# ------------------ mean and covariance ------------------

# 求均值和方差(没有noise)
upper = []
lower = []
for i in range(200):
    get_Phi = Phi_test[i]
    miu = np.dot(M_n, np.transpose(get_Phi))
    sigma1 = np.sqrt(np.dot(get_Phi, np.dot(S_n, get_Phi.T)))
    # print(sigma1.shape)

    m = miu + 2 * sigma1
    upper.append(m)
    n = miu - 2 * sigma1
    lower.append(n)

upper = np.array(upper).reshape((1, 200))
upper = upper[0]
lower = np.array(lower).reshape((1, 200))
lower = lower[0]

# 求均值和方差(有noise)
upper1 = []
lower1 = []
for i in range(200):
    get_Phi = Phi_test[i]
    miu1 = np.dot(M_n, np.transpose(get_Phi))
    sigma2 = np.sqrt(np.dot(get_Phi, np.dot(S_n, get_Phi.T)) + beta)

    m = miu1 + 2 * sigma2
    upper1.append(m)
    n = miu1 - 2 * sigma2
    lower1.append(n)

upper1 = np.array(upper1).reshape((1, 200))
upper1 = upper1[0]
lower1 = np.array(lower1).reshape((1, 200))
lower1 = lower1[0]

x_ = X_test.reshape((1, 200))

# shaded bars
plt.plot(x_[0], upper1, '--', label='upper-bound with the noise')
plt.plot(x_[0], lower1, '--', label='lower-bound without the noise')
plt.fill_between(x_[0], upper, lower, where=upper >= lower, alpha=0.5,
                 label='standard deviations error without noise',facecolor = 'grey')


# plt.plot(x_gd, y_gd, color='green', marker='v', linewidth=2, markersize=0, label='Gradient descent')
plt.xlabel('Test inputs')
plt.ylabel('Predictive mean')
plt.legend(loc='best',framealpha = 0.5)
plt.axis([-1, 1.5, -3, 6])
plt.show()
#print(theta)