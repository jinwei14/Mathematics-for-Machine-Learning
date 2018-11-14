import numpy as np
import matplotlib.pyplot as plt

# Training set
N = 25
X = np.reshape(np.linspace(0, 0.9, N), (N, 1))
Y = np.cos(10 * X ** 2) + 0.1 * np.sin(100 * X)


def lml(alpha, beta, Phi, Y):
    a = -0.5 * (Phi.shape[0]) * np.log(2 * np.pi)
    m = np.dot(np.dot(Phi * alpha, np.eye(Phi.shape[1])), np.transpose(Phi)) + beta * np.eye(Phi.shape[0])
    b = -0.5 * np.log(np.linalg.det(m))
    c = -0.5 * np.dot(np.dot(np.transpose(Y), np.linalg.inv(m)), Y)
    return float(a + b + c)


def grad_lml(alpha, beta, Phi, Y):
    # compute alpha*phi*phi_T+beta*I
    m = np.dot(alpha * Phi, np.transpose(Phi)) + beta * np.eye(Phi.shape[0])
    # compute y*y_T-m
    n = np.dot(Y, np.transpose(Y)) - m
    p = 0.5 * np.dot(np.dot(np.linalg.inv(m), n), np.linalg.inv(m))
    d_lml_d_alpha = np.trace(np.dot(np.transpose(p), np.dot(Phi, np.transpose(Phi))))
    d_lml_d_beta = np.trace(np.dot(np.transpose(p), np.eye(Phi.shape[0])))
    return np.array([d_lml_d_alpha, d_lml_d_beta])


def Phi1(X):
    M = len(X)
    Phi = np.zeros((M, 11))
    # means
    means = np.reshape(np.linspace(-0.5, 1, 10), (10, 1))
    for i in range(M):
        Phi[i][0] = 1
        for j in range(1, 11):
            Phi[i][j] = np.exp(-((X[i] - means[j - 1]) ** 2) / (2 * 0.1 * 0.1))
    return np.array(Phi)


# 求posterior的中值和方差
alpha = 1.0
beta = 0.1
Phi = Phi1(X)
print(Phi)

s_N = np.array(np.linalg.inv(np.dot(Phi.T, Phi) / beta + 1 / alpha * np.eye(Phi.shape[1])))
m_N = np.array(np.dot(s_N, np.dot(Phi.T, Y) * 1.0 / beta)).reshape((1, 11))
print(s_N.shape)
print(m_N.shape)
samples = np.random.multivariate_normal(m_N[0], s_N, 5)
print(samples.shape)

# test points
X_test = np.reshape(np.linspace(-1, 1.5, 200), (200, 1))

Phi_test = Phi1(X_test)
print(Phi_test.shape)
# print(Phi_test.shape)

for sample in samples:
    mean = np.dot(Phi_test, sample)
    # print(sample.shape)
    plt.plot(X_test, mean, label='sample')

# 求均值和方差(没有noise)
upper = []
lower = []
for i in range(200):
    get_Phi = Phi_test[i]
    miu = np.dot(m_N, np.transpose(get_Phi))
    sigma1 = np.sqrt(np.dot(get_Phi, np.dot(s_N, get_Phi.T)))
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
    miu1 = np.dot(m_N, np.transpose(get_Phi))
    sigma2 = np.sqrt(np.dot(get_Phi, np.dot(s_N, get_Phi.T)) + beta)

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
plt.fill_between(x_[0], upper, lower, where=upper >= lower, alpha=0.5,
                 label='2 standard deviation error bars without noise')
plt.plot(x_[0], upper1, '--', label='upper-bound including the noise')
plt.plot(x_[0], lower1, '--', label='lower-bound including the noise')

plt.title('Predictive mean and 2 standard deviation error')
plt.xlabel('Test inputs')
plt.ylabel('Predictive mean')
plt.legend(loc=2, bbox_to_anchor=(1.05, 1.0), borderaxespad=0.)
plt.show()
