# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt


def cost_gradient(W, X, Y, n):
    y_hat = np.dot(X, W)
    G = (1 / n) * np.dot(X.T, (y_hat - Y))  ###### Gradient
    j = (1 / (2 * n)) * np.sum((y_hat - Y) ** 2)  ###### cost with respect to current W

    return (j, G)


def gradientDescent(W, X, Y, lr, iterations):
    n = np.size(Y)
    J = np.zeros([iterations, 1])

    for i in range(iterations):
        (J[i], G) = cost_gradient(W, X, Y, n)
        W = W - lr * G  ###### Update W based on gradient

    return (W, J)


iterations = 2000  ###### Training loops
lr = 0.0006144  ###### Learning rate

data = np.loadtxt('LR.txt', delimiter=',')

n = np.size(data[:, 1])
W = np.zeros([2, 1])
X = np.c_[np.ones([n, 1]), data[:, 0]]
Y = data[:, 1].reshape([n, 1])

(W, J) = gradientDescent(W, X, Y, lr, iterations)

# Draw figure
plt.figure()
plt.plot(data[:, 0], data[:, 1], 'rx')
plt.plot(data[:, 0], np.dot(X, W))
# print(J)
plt.figure()
plt.plot(range(iterations), J)
plt.show()



