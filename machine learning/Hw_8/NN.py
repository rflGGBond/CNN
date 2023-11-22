# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def softmax(z):
    max_z = np.max(z, axis=1, keepdims=True)
    z = z - max_z
    e_z = np.exp(z)
    return e_z / np.sum(np.exp(z), axis=1, keepdims=True)


# Utilities
def onehotEncoder(Y, ny):
    return np.eye(ny)[Y]


# Xavier Initialization
def initWeights(M):
    l = len(M)
    W = []
    B = []

    for i in range(1, l):
        W.append(np.random.randn(M[i - 1], M[i]))
        B.append(np.zeros([1, M[i]]))

    return W, B


# Forward propagation
def networkForward(X, W, B):
    l = len(W)
    A = [None for i in range(l + 1)]
    A[0] = X
    for i in range(l - 1):
        Z = np.dot(A[i], W[i]) + B[i]
        A[i + 1] = sigmoid(Z)
    A[l] = np.dot(A[l - 1], W[l - 1]) + B[l - 1]
    A[l] = softmax(A[l])
    ##### Calculate the output of every layer A[i], where i = 0, 1, 2, ..., l
    return A


# --------------------------

# Backward propagation
def networkBackward(Y, A, W):
    l = len(W)
    dW = [None for i in range(l)]
    dB = [None for i in range(l)]

    dA = A[-1] - Y
    for i in range(l - 1, -1, -1):
          dZ = dA
          dW[i] = np.dot(A[i].T, dZ) / A[i].shape[0]
          dB[i] = np.sum(dZ, axis=0, keepdims=True) / A[i].shape[0]

          if i > 0:
                dA = np.dot(dZ, W[i].T)
                dA = dA * A[i] * (1 - A[i])
    ##### Calculate the partial derivatives of all w and b in each layer dW[i] and dB[i], where i = 1, 2, ..., l

    return dW, dB


# --------------------------

# Update weights by gradient descent
def updateWeights(W, B, dW, dB, lr):
    l = len(W)

    for i in range(l):
        W[i] = W[i] - lr * dW[i]
        B[i] = B[i] - lr * dB[i]

    return W, B


# Compute regularized cost function
def cost(A_l, Y, W):
    n = Y.shape[0]
    c = -np.sum(Y * np.log(A_l)) / n

    return c


def train(X, Y, M, lr=0.1, iterations=3000):
    costs = []
    W, B = initWeights(M)

    for i in range(iterations):
        A = networkForward(X, W, B)
        c = cost(A[-1], Y, W)
        dW, dB = networkBackward(Y, A, W)
        W, B = updateWeights(W, B, dW, dB, lr)

        if i % 200 == 0:
            print("Cost after iteration %i: %f" % (i, c))
            costs.append(c)

    return W, B, costs


def predict(X, W, B, Y):
    Y_out = np.zeros([X.shape[0], Y.shape[1]])

    A = networkForward(X, W, B)
    idx = np.argmax(A[-1], axis=1)
    Y_out[range(Y.shape[0]), idx] = 1

    return Y_out


def test(Y, X, W, B):
    Y_out = predict(X, W, B, Y)
    acc = np.sum(Y_out * Y) / Y.shape[0]
    print("Training accuracy is: %f" % (acc))

    return acc


iterations = 6000  ###### Training loops
lr = 0.3  ###### Learning rate

data = np.load("data.npy")

X = data[:, :-1]
Y = data[:, -1].astype(np.int32)
(n, m) = X.shape
Y = onehotEncoder(Y, 10)

M = [400, 250, 200, 150, 100, 10]
W, B, costs = train(X, Y, M, lr, iterations)

plt.figure()
plt.plot(range(len(costs)), costs)

test(Y, X, W, B)
plt.show()
