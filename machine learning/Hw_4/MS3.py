# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold


def read_data(addr):
    data = np.loadtxt(addr, delimiter=',')

    n = data.shape[0]

    ###### You may modify this section to change the model
    X = np.concatenate([np.ones([n, 1]), data[:, 0:6]], axis=1)
    X[:, 1] = X[:, 1] ** 4
    X[:, 2] = X[:, 2] ** 2
    X[:, 3] = X[:, 3] ** 3
    X[:, 4] = X[:, 4] ** 1
    X[:, 5] = X[:, 5] ** 5
    X[:, 6] = X[:, 6] ** 6
    ###### You may modify this section to change the model

    Y = None
    if "Train" in addr:
        Y = np.expand_dims(data[:, 6], axis=1)

    return (X, Y, n)


def cost_gradient(W, X, Y, n):
    z = -1 * (X @ W)
    y0 = 1 / (np.exp(z) + 1)  # y hat
    y1 = (1 / (np.exp(z) + 1) - Y) / n
    G = X.T @ y1
    j = ((-1 * Y).T @ np.log(y0) - (1 - Y).T @ np.log(1 - y0)) / n

    return (j, G)


def train(W, X, Y, lr, n, iterations):
    J = np.zeros([iterations, 1])
    E_trn = np.zeros([iterations, 1])
    E_val = np.zeros([iterations, 1])
    kf = KFold(n_splits=10)
    for train_index, val_index in kf.split(X):
        X_trn, X_val = X[train_index], X[val_index]
        Y_trn, Y_val = Y[train_index], Y[val_index]
        for i in range(iterations):
            (J[i], G) = cost_gradient(W, X_trn, Y_trn, n)
            W = W - lr * G
            E_trn[i] = error(W, X_trn, Y_trn)
            E_val[i] = error(W, X_val, Y_val)
        print(E_val[-1])  # Print the error of the last iteration for each fold

    return (W, J, E_trn, E_val)


def error(W, X, Y):
    Y_hat = 1 / (1 + np.exp(-X @ W))
    Y_hat[Y_hat < 0.5] = 0
    Y_hat[Y_hat > 0.5] = 1

    return (1 - np.mean(np.equal(Y_hat, Y)))


def predict(W):
    (X, _, _) = read_data("Test_Data.csv")

    Y_hat = 1 / (1 + np.exp(-X @ W))
    Y_hat[Y_hat < 0.57] = 0
    Y_hat[Y_hat > 0.57] = 1

    idx = np.expand_dims(np.arange(1, 201), axis=1)
    np.savetxt("predict.csv", np.concatenate([idx, Y_hat], axis=1), header="Index,ID", comments='', delimiter=',')


iterations = 1000
lr = 0.33

(X, Y, n) = read_data("Train.csv")
W = np.random.random([X.shape[1], 1])

(W, J, E_trn, E_val) = train(W, X, Y, lr, n, iterations)

###### You may modify this section to do 10-fold validation
plt.figure()
plt.plot(range(iterations), J)
plt.figure()
plt.ylim(0, 1)
plt.plot(range(iterations), E_trn, "b")
plt.plot(range(iterations), E_val, "r")
###### You may modify this section to do 10-fold validation

predict(W)
