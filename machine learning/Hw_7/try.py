# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


def read_data(addr,a):
    data = np.loadtxt(addr, delimiter=',')
    n = data.shape[0]

    # You may modify this section to change the model
    X = np.concatenate([np.ones([n, 1]),
                        np.expand_dims(np.power(data[:, 0], a[0]), axis=1),
                        np.expand_dims(np.power(data[:, 1], a[1]), axis=1),
                        np.expand_dims(np.power(data[:, 2], a[2]), axis=1),
                        np.expand_dims(np.power(data[:, 3], a[3]), axis=1),
                        np.expand_dims(np.power(data[:, 4], a[4]), axis=1),
                        np.expand_dims(np.power(data[:, 5], a[5]), axis=1),
                        np.expand_dims(np.power(data[:, 6], a[6]), axis=1),
                        np.expand_dims(np.power(data[:, 7], a[7]), axis=1)],
                       axis=1)

    Y = None
    if "train" in addr:
        Y = np.expand_dims(data[:, -1], axis=1)

    return (X, Y, n)


def cost_gradient(W, X, Y, n, lambd):
    Y_head = 1 / (1 + np.exp(-X @ W))
    G = (1 / n) * np.dot(X.T, (Y_head - Y)) + lambd * W  # Gradient
    j = (1 / (2 * n)) * np.sum((Y_head - Y) ** 2) + lambd * np.sum(W ** 2) / 2.0  # Cost with respect to current W

    return (j, G)


def train(W, X, Y, lr, n, iterations, lambd):
    J = np.zeros([iterations, 1])

    for i in range(iterations):
        (J[i], G) = cost_gradient(W, X, Y, n, lambd)
        W = W - lr * G
    err = error(W, X, Y)

    return (W, J, err)


def error(W, X, Y):
    Y_hat = 1 / (1 + np.exp(-X @ W))
    Y_hat[Y_hat < 0.5] = 0
    Y_hat[Y_hat > 0.5] = 1

    return (1 - np.mean(np.equal(Y_hat, Y)))


def predict(W,a):
    (X, _, _) = read_data("test_Data.csv",a)

    Y_hat = 1 / (1 + np.exp(-X @ W))
    Y_hat[Y_hat < 0.5] = 0
    Y_hat[Y_hat > 0.5] = 1

    idx = np.expand_dims(np.arange(1, 201), axis=1)
    np.savetxt("predict.csv", np.concatenate([idx, Y_hat], axis=1), header="Index,ID", comments='', delimiter=',')


def find_best_feature_combination(addr, iterations, lr, lambd):
    best_feature_combination = [2, 3, 2, 4, 1, 2, 3, 2]  # Default feature combination
    best_err = float('inf')

    for degree_0 in range(4, 5):
        for degree_1 in range(3, 5):
            for degree_2 in range(2, 5):
                for degree_3 in range(4, 5):
                    for degree_4 in range(1, 5):
                        for degree_5 in range(2, 5):
                            for degree_6 in range(3, 5):
                                for degree_7 in range(2, 5):
                                    feature_combination = [degree_0, degree_1, degree_2, degree_3, degree_4, degree_5,
                                                           degree_6, degree_7]

                                    (X, Y, n) = read_data(addr,feature_combination)

                                    W = np.random.random([X.shape[1], 1])
                                    (W, _, err) = train(W, X, Y, lr, n, iterations, lambd)
                                    print("Feature Combination:", feature_combination)
                                    print("Error:", best_err)
                                    if err < best_err:
                                        best_err = err
                                        best_feature_combination = feature_combination
                                    print("Best Feature Combination:", best_feature_combination)
                                    print("Best Error:", best_err)
                                    print("\n")

    return best_feature_combination, best_err


if __name__ == "__main__":
    # Specify the file path for the training data
    training_data_path = "train.csv"

    iteration = 500
    lr = 0.035
    lambd = 0.109
    # Find the best feature combination
    best_feature_combination, best_err = find_best_feature_combination(training_data_path, iteration, lr,
                                                                       lambd)
    print("Best Feature Combination:", best_feature_combination)
    print("Best Error:", best_err)

    # Now, you can use the best_feature_combination to read and train the data with the selected features
    (X, Y, n) = read_data(training_data_path,best_feature_combination)
    W = np.random.random([X.shape[1], 1])
    (W, J, err) = train(W, X, Y, lr, n, iteration, lambd)
    print("Final Error with Best Feature Combination:", err)

    # You can proceed with prediction and visualization as before
    plt.figure()
    plt.plot(range(iteration), J)

    predict(W,best_feature_combination)
    plt.show()
