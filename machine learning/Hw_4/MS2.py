# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt



def read_data(addr):
    data = np.loadtxt(addr, delimiter=',')

    n = data.shape[0]  # 1000

    ###### You may modify this section to change the model
    X = np.concatenate([np.ones([n, 1]), data[:, 0:6]], axis=1)
    ###### You may modify this section to change the model

    '''polynomial_degrees = [1, 2, 3, 4, 6, 6]
    X_poly = np.ones([n, 1])
    for i, degree in enumerate(polynomial_degrees):
        poly = PolynomialFeatures(degree=degree, include_bias=False)
        X_poly_i = poly.fit_transform(np.expand_dims(data[:, i], axis=1))
        X_poly = np.concatenate([X_poly, X_poly_i], axis=1)

    X = np.concatenate([X, X_poly], axis=1)'''

    # 定义多项式的次数列表，每一维对应的多项式次数
    polynomial_degrees = [1, 2, 3, 4, 6, 6]

    # 初始化一个空的多项式特征矩阵，每行都为1
    X_poly = np.ones((n, 1))

    # 循环遍历每一维的多项式次数
    for i, degree in enumerate(polynomial_degrees):
        # 获取当前维度的数据
        data_i = data[:, i]

        # 初始化一个空的特征矩阵，用于存储当前维度的多项式特征
        X_poly_i = np.ones((n, 1))

        # 循环遍历多项式次数，构建多项式特征
        for d in range(1, degree + 1):
            # 计算当前维度的多项式特征，并将其添加到特征矩阵中
            poly_feature = np.power(data_i, d)
            X_poly_i = np.concatenate((X_poly_i, poly_feature.reshape(-1, 1)), axis=1)

        # 将当前维度的多项式特征添加到总的多项式特征矩阵中
        X_poly = np.concatenate((X_poly, X_poly_i), axis=1)

    # 将多项式特征矩阵与原始特征矩阵连接，得到最终的特征矩阵
    X = np.concatenate((X, X_poly), axis=1)

    Y = None
    if "train" in addr:
        Y = np.expand_dims(data[:, 6], axis=1)

    return (X, Y, n)


def cost_gradient(W, X, Y, n):

    Z = np.dot(X, W)
    Y_head = 1.0 / (1.0 + np.exp(-Z))
    G = (1 / n) * np.dot(X.T, (Y_head - Y))

    j = (1 / n) * np.sum(-Y * np.log(Y_head) - (1 - Y) * np.log(1 - Y_head))

    return (j, G)


def train1(W, X, Y, lr, n, iterations):
    ###### You may modify this section to do 10-fold validation
    J = np.zeros([iterations, 1])
    E_trn = np.zeros([iterations, 1])
    E_val = np.zeros([iterations, 1])
    '''n = int(0.9*n)
    # X_trn = X[:n]
    # Y_trn = Y[:n]
    # X_val = X[n:]
    # Y_val = Y[n:]

    kf = KFold(n_splits=10)
    for i in range(iterations):
        # (J[i], G) = cost_gradient(W, X_trn, Y_trn, n)
        # W = W - lr*G
        # E_trn[i] = error(W, X_trn, Y_trn)
        # E_val[i] = error(W, X_val, Y_val)
        for train_index, val_index in kf.split(X):
            X_trn, X_val = X[train_index], X[val_index]
            Y_trn, Y_val = Y[train_index], Y[val_index]

            (J[i], G) = cost_gradient(W, X_trn, Y_trn, len(train_index))
            W = W - lr * G
            E_trn[i] = error(W, X_trn, Y_trn)
            E_val[i] = error(W, X_val, Y_val)'''
    # 定义折叠数
    n_splits = 10

    # 计算每一折的样本数
    fold_size = len(X) // n_splits

    # 初始化KFold的分割索引
    fold_indices = np.arange(len(X))

    # 迭代训练
    for i in range(iterations):
        # 打乱数据集并划分折
        np.random.shuffle(fold_indices)
        folds = np.array_split(fold_indices, n_splits)

        for fold in folds:
            # 选择验证集和训练集的索引
            val_indices = fold
            train_indices = np.concatenate(folds[np.arange(n_splits) != fold])

            # 根据索引获取验证集和训练集
            X_trn, X_val = X[train_indices], X[val_indices]
            Y_trn, Y_val = Y[train_indices], Y[val_indices]

            # 计算当前迭代的代价和梯度
            (J[i], G) = cost_gradient(W, X_trn, Y_trn, len(train_indices))

            # 更新模型参数W
            W = W - lr * G

            # 计算训练集和验证集的错误率并保存
            E_trn[i] = error(W, X_trn, Y_trn)
            E_val[i] = error(W, X_val, Y_val)

    print(E_val[-1])
    ###### You may modify this section to do 10-fold validation

    return (W, J, E_trn, E_val)


def train(W, X, Y, lr, n, iterations):
    # 初始化一个长度为iterations的数组来存储代价
    J = np.zeros([iterations, 1])

    # 初始化一个长度为iterations的数组来存储训练集错误
    E_trn = np.zeros([iterations, 1])

    # 初始化一个长度为iterations的数组来存储验证集错误
    E_val = np.zeros([iterations, 1])

    # 定义折叠数
    n_splits = 10

    # 计算每一折的样本数
    fold_size = n // n_splits

    # 创建KFold的分割索引
    kf_indices = np.arange(n)

    # 迭代训练
    for i in range(iterations):
        # 打乱KFold的分割索引
        np.random.shuffle(kf_indices)

        # 划分训练集为10折
        folds = np.array_split(kf_indices, n_splits)

        for fold in folds:
            # 选择验证集和训练集的索引
            val_indices = fold
            train_indices = np.concatenate([f for f in folds if not np.array_equal(f, fold)])

            # 根据索引获取验证集和训练集
            X_trn, X_val = X[train_indices], X[val_indices]
            Y_trn, Y_val = Y[train_indices], Y[val_indices]

            # 计算当前迭代的代价和梯度
            (J[i], G) = cost_gradient(W, X_trn, Y_trn, len(train_indices))

            # 更新模型参数W
            W = W - lr * G

            # 计算训练集和验证集的错误率并保存
            E_trn[i] = error(W, X_trn, Y_trn)
            E_val[i] = error(W, X_val, Y_val)

    print(E_val[-1])

    return (W, J, E_trn, E_val)


def error(W, X, Y):
    Y_hat = 1 / (1 + np.exp(-X @ W))
    Y_hat[Y_hat < 0.5] = 0
    Y_hat[Y_hat > 0.5] = 1

    return (1 - np.mean(np.equal(Y_hat, Y)))


def predict(W):
    (X, _, _) = read_data("test_data.csv")

    Y_hat = 1 / (1 + np.exp(-X @ W))
    Y_hat[Y_hat < 0.5] = 0
    Y_hat[Y_hat > 0.5] = 1

    idx = np.expand_dims(np.arange(1, 201), axis=1)
    np.savetxt("predict.csv", np.concatenate([idx, Y_hat], axis=1), header="Index,ID", comments='', delimiter=',')


iterations = 15000  ###### Training loops
lr = 0.0015  ###### Learning rate

(X, Y, n) = read_data("train.csv")
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
plt.show()