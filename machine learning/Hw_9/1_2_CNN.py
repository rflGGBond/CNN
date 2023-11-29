# -*- coding: utf-8 -*-
from asyncio import sleep
import numpy as np
import matplotlib.pyplot as plt
import time

#迭代次数
iterations = 10

##学习率
lr =  0.1

###### The number of layers层数
layers = 1

#每层卷积的通道数
num_channels = 1

#每层卷积核的数量
kernels_layer_number = 1

#卷积核的大小,长宽都为3
kernels_size = 3

#池化核的大小,目前设长宽为4
pool_size = 4

#Utilities
def onehotEncoder(Y, ny):
    return np.eye(ny)[Y]

#Relu函数
def relu_forward(X):
    """
    ReLU激活函数

    返回：
    - 返回激活后的矩阵
    """
    return np.maximum(0, X)

#ReLU激活函数的反向传播
def relu_backward(dpool, conv_output):
    """
    ReLU激活函数的反向传播。

    参数：
    - conv_output: 卷积后的矩阵
    - dpool: 池化反向传播后的结果
    """
    X = conv_output
    dX = np.array(dpool, copy=True)  # 复制dout到dX
    dX[X <= 0] = 0  # 当X中的数值小于等于0时，梯度为0
    return dX

#初始化W,B
def initWeights(kernels_layer_number,kernels_size,num_channels):
    """
    初始化W,B

    参数：
    - layers:卷积层数
    - kernels_layer_number:每层卷积核的数量
    - kernels_size:卷积核的大小,长宽都一样
    - num_channels: 每层卷积的通道数
    返回：
    - 参数W,B
    """
    W = np.random.rand(kernels_layer_number, kernels_size,
                    kernels_size,num_channels)
    B = np.random.rand(kernels_layer_number)
    return W, B

# 卷积层的前向传播
def conv_forward(X, W, b, stride, padding):
    """
    执行一个2D卷积。

    参数:
    - X: 形状为 (N,H, W, C) N是数据集大小,H 是高度,W 是宽度,C 是通道数,通道数为1。
    - W: 形状为 (n_kernels, H_k, W_k,C) 的3维数组,n_kernels 是卷积核的数量,
      H_k 是卷积核的高度,W_k 是卷积核的宽度, C 是通道数,通道数为1。
    - bias: 形状为 (n_kernels,) 的1维数组,包含每个卷积核的偏置。
    - stride: 在每一步卷积中卷积核移动的像素数。
    - padding: 为输入添加到边界的填充的像素数。

    返回:
    - output: 卷积后的图像,形状为 (H_out, W_out, n_kernels)。
    - conv_cache: padding_input填充后的X数据,stride为步长
    """
    # 输入的维度
    N_in,H_in, W_in= X.shape

    # 卷积核的维度
    n_kernels, H_k, W_k,C_k = W.shape

    #输出的维度
    H_out = (H_in + 2 * padding - H_k) // stride + 1
    W_out = (W_in + 2 * padding - W_k) // stride + 1

    #用零初始化输出
    output = np.zeros((N_in,H_out,W_out,n_kernels))

    #对输入层进行边界像素填充
    padding_input = np.pad(X, ((0, 0), (padding, padding), (padding, padding)), 
                           mode='constant', constant_values=0)

    # 执行卷积
    for n in range(N_in):
        for i in range(n_kernels):
            for y in range(H_in):
                for x in range(W_in):
                    # 确定输入的区域
                    y_start = y * stride
                    y_end =  y_start + H_k
                    x_start = x * stride
                    x_end = x_start + W_k

                    # 从填充输入中提取区域
                    region = padding_input[n,y_start:y_end,x_start:x_end]

                    #执行卷积操作并添加偏置
                    convolved = np.sum(region * W[i,:,:,:]) + b[i]

                    #将卷积后的值赋给输出像素
                    output[n,y,x,i] = convolved

    conv_cache = (padding_input, stride, W, b)
    return output,conv_cache

#卷积层的反向传播
def conv_backward(dout, conv_cache):
    """
    卷积层的反向传播。

    参数:
    - dout: 损失相对于卷积层输出的梯度，形状为(N, F, H', W')
    - conv_cache: 卷积层前向传播时的缓存数据,包含padding_input,stride,W,b

    返回:
    - dW: 损失相对于卷积层权重的梯度,形状与W相同
    - db: 损失相对于卷积层偏置的梯度,形状与b相同
    """
    N,H_in,W_in,C_in = dout.shape
    padding_input,stride,W,B = conv_cache
    layer_number,H_out,W_out,C_out = W.shape
    dW = np.zeros((layer_number,H_out,W_out,C_out))
    dB = B
    for n in range(layer_number):
        for i in range(C_out):
            for y in range(H_out):
                for x in range(W_out):
                    y_start = y * stride
                    y_end = y_start + H_in
                    x_start = x * stride
                    x_end = x_start + W_in 
                    dW[n,y,x,i] = np.sum(padding_input[:,y_start:y_end,x_start:x_end] * dout[:,:,:,i]) / N
                    dB[n] = 1
    
    return dW,dB


#池化
def max_pool_forward(X, pool_size, stride,padding):
    """
    执行最大池化操作的前向传播。
    
    参数:
    - X: 输入数据,形状为 (H,W,n_kernels), 其中H 是高度,W 宽度,n_kernels是通道数
    - pool_size: 池化窗口的大小
    - stride: 池化操作的步幅
    - padding: 为输入添加到边界的填充的像素数。

    返回:
    - output: 池化后的输出数据。
    - cache: 缓存一些值,用于反向传播时计算梯度。
    """
    #获取输入数据的维度
    N_in,H_in,W_in,n_kernels = X.shape

    #计算输出数据的高度和宽度
    H_out = (H_in + padding - pool_size) // stride + 1
    W_out = (W_in + padding - pool_size) // stride + 1

    #用零初始化数据
    output = np.zeros((N_in,H_out,W_out,n_kernels))
    max_array = np.zeros((N_in,H_in,W_in,n_kernels))
    #进行池化操作
    for n in range(N_in):
        for i in range(n_kernels):
            for y in range(H_out):
                for x in range(W_out):
                    y_start = y * stride
                    y_end = y_start + pool_size
                    x_start = x * stride
                    x_end = x_start + pool_size

                    #选择当前池化窗口的区域
                    region = X[n,y_start:y_end+1,x_start:x_end+1,i]
                    #获取最大值索引
                    max_index = np.unravel_index(region.argmax(), region.shape)
                    max_array[n,y_start+max_index[0],x_start+max_index[1],i] = 1
                    #对当前窗口取最大值
                    output[n,y,x,i] = np.max(region)

    #缓存值,反向传播时使用
    cache = (max_array, pool_size, stride)

    return output,cache

#展平
def flatten(X):
    """
    展平操作将输入数据的多维特征图转换为一维特征向量。

    参数:
    - X: 输入数据,形状为 (N,H, W, C),其中
      N 是批量大小,
      C 是通道数,
      H 是特征图的高度,
      W 是特征图的宽度。

    返回:
    - out: 展平后的输出数据,形状为 (N, C*H*W)。
    """
    N, H, W ,C = X.shape  # 获取输入数据的维度
    out = X.reshape(N, -1)  # 使用reshape函数将数据展平,-1表示自动计算该维度的大小
    return out

#恢复展平
def d_flatten(pool_output,X):
    """
    恢复展平后的结果

    参数:
    - pool_output: 池化后的数据,形状为 (N,H, W, C),其中
      N 是批量大小,
      H 是特征图的高度,
      W 是特征图的宽度。
      C 是通道数,

    - X: 展平后的数据形状为 (N, C*H*W)

    返回:
    - out: 恢复展平后的输出数据,形状为  (N,H, W, C)。
    """
    N, H, W ,C = pool_output.shape  # 获取输入数据的维度
    out = X.reshape(N,H, W, C)  # 使用reshape函数将数据展平,-1表示自动计算该维度的大小
    return out

#前向传播到全连接层
def fully_connected_forward(X, W, b):
    """
    全连接层的前向传播。
    
    参数:
    - X: 输入数据,(N,C),N是数据量,C是列数
    - W: 权重矩阵,(C,R),C与X的列数对应,R是输出结果的数量,十分类就是10
    - b: 偏置向量,有十个输出,所以是(1,10)
    
    返回:
    - out: 全连接层的输出
    """
    fully_out = np.dot(X, W) + b  # 输入数据和权重矩阵的点积后加上偏置
    return fully_out

#池化层的反向传播
def max_pool_backward(d_in,cache):
    """
    最大池化层的反向传播。
    """
    max_array, pool_size, stride = cache
    N_out,H_out,W_out,n_kernels = max_array.shape
    #d_out = np.zeros((N_out,H_out,W_out,n_kernels))
    N_in,H_in,W_in,n_kernels = d_in.shape
    for n in range(N_out):
        for i in range(n_kernels):
            for y in range(H_in):
                for x in range(W_in):
                    y_start = y * stride
                    y_end = y_start + pool_size
                    x_start = x * stride
                    x_end = x_start + pool_size
                    max_array[n,y_start:y_end+1,x_start:x_end+1,i] *= d_in[n,y,x,i]
    return max_array

#softmax函数
def softmax(x):
    """
    计算输入x的softmax。

    参数:
    - x: 一个形状为 (n, m) 的numpy数组,n是样本数,m是特征数。

    返回:
    - softmax_x: 输入x的softmax值,同样是一个形状为 (n, m) 的numpy数组。
    """
    # 防止在指数计算时溢出,每个样本减去其最大值
    x_exp_shifted = np.exp(x - np.max(x, axis=1, keepdims=True))
    # 计算softmax
    softmax_x = x_exp_shifted / np.sum(x_exp_shifted, axis=1, keepdims=True)
    return softmax_x

#计算损失函数
def cost(Y_hat, Y):
    n = Y.shape[0]
    c = -np.sum(Y*np.log(Y_hat)) / n
    
    return c

#测试
def test(Y_hat, Y):
    Y_out = np.zeros_like(Y)
    
    #idx = np.argmax(Y_hat[-1], axis=1)
    idx = np.argmax(Y_hat, axis=1)
    Y_out[range(Y.shape[0]),idx] = 1
    acc = np.sum(Y_out * Y) / Y.shape[0]
    print("Training accuracy is: %f" %(acc))
    return acc

#开始训练
def train(X, Y, lr = 0.1, iterations = 3):
    """
    开始训练

    参数：
    - X : 输入数据
    - Y : 数据结果
    - lr : 学习率
    - iterations : 迭代次数
    """
    costs = []
    W = []
    B = []
    W_temp,B_temp = initWeights(kernels_layer_number,kernels_size,num_channels)
    W.append(W_temp)
    B.append(B_temp)
    W.append(np.random.rand(25,10))
    B.append(np.random.rand(1,10))
    Y_hat = np.zeros_like(Y)
    for i in range(iterations):
        #第一层卷积
        conv_output,conv_cache = conv_forward(X,W[0],B[0],stride = 1,padding = 1)
        #ReLU激活函数
        relu_output = relu_forward(conv_output)
        #池化
        pool_output,cache = max_pool_forward(relu_output, pool_size, stride=4, padding=0)
        #展平
        flatten_out = flatten(pool_output)
        #前向传播到全连接层
        fully_out = fully_connected_forward(flatten_out, W[1],B[1])
        #softmax激活函数
        softmax_out = softmax(fully_out)
        Y_hat = softmax_out
        #计算损失
        #softmax激活函数反向传播
        dX = (softmax_out - Y) @ W[1].T
        dw = (flatten_out.T @ (softmax_out - Y)) / 5000
        db = np.mean((softmax_out - Y),axis=0)
        W[1] = W[1] - lr * dw
        B[1] = B[1] - lr * db
        dZ = d_flatten(pool_output, dX)
        #池化的反向传播
        dX = max_pool_backward(dZ,cache)
        #relu函数的反向传播
        dX = relu_backward(conv_output,dX)
        #卷积层的反向传播
        dW,dB = conv_backward(dX,conv_cache)
        W[0] = W[0] - lr * dW
        B[0] = B[0] - lr * dB
        c = cost(softmax_out, Y)
        costs.append(c)
        print(c)
        # c = cost(A[-1], Y, W)
        # dW, dB = networkBackward(Y, A, W)
        # W, B = updateWeights(W, B, dW, dB, lr)
    acc = test(Y_hat,Y)
    return W, B, costs

  
#处理数据
data = np.load("data.npy")
X = data[:,:-1].reshape(data.shape[0], 20, 20).transpose(0,2,1)
Y = data[:,-1].astype(np.int32)
(n, L, _) = X.shape
Y = onehotEncoder(Y, 10)
W, B, costs = train(X, Y, lr, iterations)
plt.figure()
plt.plot(range(len(costs)), costs)
#test(Y_hat, Y)