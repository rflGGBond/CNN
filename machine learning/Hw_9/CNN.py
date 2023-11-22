# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt


# Utilities
def onehotEncoder(Y, ny):
    return np.eye(ny)[Y]


# Compute the cost function
def cost(Y_hat, Y):
    n = Y.shape[0]
    c = -np.sum(Y * np.log(Y_hat)) / n

    return c


def ReLU(X):
    return np.maximum(0, X)


def softmax(z):
    max_z = np.max(z, axis=1, keepdims=True)
    z = z - max_z
    e_z = np.exp(z)
    return e_z / np.sum(np.exp(z), axis=1, keepdims=True)


def initWeights(n_kernels, kernel_size, n_channels):
    W = np.random.randn(n_kernels, kernel_size, kernel_size, n_channels)
    B = np.random.randn(n_kernels, kernel_size, kernel_size, n_channels)

    return W, B


# 卷积层前向传播
def ConvForward(X, W, B, stride, padding):
    # 输入的维度
    N_in, H_in, W_in = X.shape
    # 卷积核的维度
    n_kernels, H_k, W_k, C_k = W.shape
    # 输出的维度
    H_out = (H_in + 2 * padding - H_k) // stride + 1
    W_out = (W_in + 2 * padding - W_k) // stride + 1

    output = np.zeros((N_in, H_out, W_out, n_kernels))
    padding_in = np.pad(X, ((0, 0), (padding, padding), (padding, padding)),
                        'constant', constant_values=0)
    # 开始卷积
    for n in range(N_in):
        for i in range(n_kernels):
            for y in range(H_in):
                for x in range(W_in):
                    y_start = y * stride  # 竖向起点
                    y_end = y_start + H_k  # 竖向终点
                    x_start = x * stride  # 横向起点
                    x_end = x_start + W_k  # 横向终点

                    temp = padding_in[y_start:y_end, x_start:x_end]
                    # 卷积操作
                    conv = np.sum(temp * W[i, :, :, :]) + B[i]
                    output[n, y, x, i] = conv
                    # output = ReLU(output)
    cache = (X, W, B)

    return output, cache


# 卷积层反向传播
def ConvBackward(dA, cache, stride, padding):
    (X, W, B) = cache
    (N_in, H_in_prev, W_in_prev, n_C_prev) = X.shape
    (n_kernels, H_k, W_k, C_k) = W.shape
    (N_in, H_in, W_in, n_C) = dA.shape

    dA_prev = np.zeros((N_in, H_in, W_in, n_C))
    dW = np.zeros((n_kernels, H_k, W_k, C_k))
    dB = np.zeros((1, 1, 1, n_C))

    X_pad = np.pad(X, ((0, 0), (padding, padding), (padding, padding)),
                        'constant', constant_values=0)
    dA_prev_pad = np.pad(dA_prev, ((0, 0), (padding, padding), (padding, padding)),
                        'constant', constant_values=0)

    for n in range(N_in):
        for i in range(n_kernels):
            for y in range(H_in):
                for x in range(W_in):
                    y_start = y * stride  # 竖向起点
                    y_end = y_start + H_k  # 竖向终点
                    x_start = x * stride  # 横向起点
                    x_end = x_start + W_k  # 横向终点

# 池化层前向传播
def MaxPoolingForward(X, W, pool_size, stride, padding):
    # 输入的维度
    N_in, H_in, W_in, n_C = X.shape
    # 卷积核的维度
    n_kernels, H_k, W_k, C_k = W.shape
    # 输出的维度
    H_out = (H_in + 2 * padding - H_k) // stride + 1
    W_out = (W_in + 2 * padding - W_k) // stride + 1

    A = np.zeros((N_in, H_out, W_out, n_C))

    for n in range(N_in):
        for i in range(n_kernels):
            for y in range(H_in):
                for x in range(W_in):
                    y_start = y * stride  # 竖向起点
                    y_end = y_start + H_k  # 竖向终点
                    x_start = x * stride  # 横向起点
                    x_end = x_start + W_k  # 横向终点

                    temp = X[i, y_start:y_end, x_start:x_end, x]
                    A[n, y, x, i] = np.max(temp)

    return A


# 全连接层前向传播
def FullyConnectedForward(X, W, B):
    Z = np.dot(X, W) + B
    Z = softmax(Z)

    return Z


def test(Y_hat, Y):
    Y_out = np.zeros_like(Y)

    idx = np.argmax(Y_hat[-1], axis=1)
    Y_out[range(Y.shape[0]), idx] = 1
    acc = np.sum(Y_out * Y) / Y.shape[0]
    print("Training accuracy is: %f" % (acc))

    return acc


lin = 20
###### Training loops
iterations = 2000
###### Learning rate
lr = 0.3
###### The number of layers
n_layers = 3
###### The number of convolutional kernels in each layer
n_kernels = 1
###### The size of convolutional kernels
kernel_size = 2
###### The size of pooling kernels
pool_size = 2
n_channels = 1

data = np.load("data.npy")

X = data[:, :-1].reshape(data.shape[0], 20, 20).transpose(0, 2, 1)
Y = data[:, -1].astype(np.int32)
(n, L, _) = X.shape
Y = onehotEncoder(Y, 10)

test(Y_hat, Y)
