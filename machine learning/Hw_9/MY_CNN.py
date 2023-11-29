# -*- coding: utf-8 -*-
from asyncio import sleep
import numpy as np
import matplotlib.pyplot as plt
import time
from numba import jit

"""
这是一层卷积，
卷积核大小为3,学习率为0.01最好
"""

#迭代次数
iterations = 300
##学习率
lr =  0.05
###### The number of layers层数
layers = 1

#每层卷积核的数量
kernels_layer_number = [1,1]
#卷积核的大小
kernels_size = [3,3]
#每层卷积的通道数
num_channels = [1,1]
#卷积步长
conv_stride = [1,1]
#卷积填充大小
conv_padding = [1,1]
#池化核的大小
pool_size = [2,2]
#池化步长
pool_stride = [1,1]
#池化填充大小
pool_padding = [1,1]
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

#卷积层初始化W,B
def conv_initWeights(X,W,B):
    """
    卷积层初始化W,B

    参数：
    - layers:卷积层数
    - kernels_layer_number:每层卷积核的数量
    - kernels_size:卷积核的大小,长宽都一样
    - num_channels: 每层卷积的通道数
    """
    _,C_in,H_in, W_in= X.shape
    for n in range(layers):
        W.append(np.random.rand(kernels_layer_number[n],num_channels[n],kernels_size[n],
                        kernels_size[n]))
        H_out = (H_in + 2 * conv_padding[n] - kernels_size[n]) // conv_stride[n] + 1
        W_out = (W_in + 2 * conv_padding[n] - kernels_size[n]) // conv_stride[n] + 1
        B.append(np.random.rand(kernels_layer_number[n],H_out,W_out))
        H_in = (H_out + pool_padding[n] - pool_size[n]) // pool_stride[n] + 1
        W_in = (W_out + pool_padding[n] - pool_size[n]) // pool_stride[n] + 1

#全连接层初始化W,B
def fully_initWeights(W,B,X,out):
    """
    全连接层初始化W,B
    参数：
    - W: 存储W的列表
    - B: 存储B的列表
    - X: 展平后的数据(N,L),N是数据量,L是展平后的长度
    - out: 这个是分类,比如这次是十分类就是10
    """
    W.append(np.random.rand(X.shape[1],out))
    B.append(np.random.rand(1,out))

# 卷积层的前向传播的加速
@jit(nopython=True)
def conv_forward_accelerate(N_in,n_kernels,H_out,W_out,C_in,stride,
                            H_k,W_k,padding_input,W,conv_output,B):
    """
    卷积层的前向传播的加速
    """
    for n in range(N_in):                #样本数量
        for i in range(n_kernels):       #卷积核数量 
            for y in range(H_out):        #卷积核高度
                for x in range(W_out):    #卷积核宽度
                    convolved = 0
                    for c in range(C_in): #层数
                        # 确定输入的区域
                        y_start = y * stride
                        y_end =  y_start + H_k
                        x_start = x * stride
                        x_end = x_start + W_k

                        # 从填充输入中提取区域
                        region = padding_input[n,c,y_start:y_end,x_start:x_end]
                        #执行卷积操作并添加偏置
                        convolved += np.sum(region * W[i,c,:,:])

                    #将卷积后的值赋给输出像素
                    conv_output[n,i,y,x] = convolved + B[i,y,x]

# 卷积层的前向传播
def conv_forward(X, W, B, stride, padding):
    """
    执行一个2D卷积。

    参数:
    - X: 形状为 (N,C,H, W) N是数据集大小,H 是高度,W 是宽度,C是层数。
    - W: 形状为 (n_kernels, H_k, W_k,C) 的4维数组,n_kernels 是卷积核的数量,
      H_k 是卷积核的高度,W_k 是卷积核的宽度, C 是通道数,通道数为1。
    - B: 形状为 (n_kernels,) 的1维数组,包含每个卷积核的偏置。
    - stride: 在每一步卷积中卷积核移动的像素数。
    - padding: 为输入添加到边界的填充的像素数。

    返回:
    - conv_output: 卷积后的图像,形状为 (N_out, n_kernels, H_out, W_out)。
    - conv_cache: padding_input填充后的X数据,stride为步长
    """
    # 输入的维度
    N_in, C_in, H_in, W_in= X.shape
    #对输入层进行边界像素填充
    padding_input = np.pad(X, ((0, 0), (0,0),(padding, padding), (padding, padding)), 
                    mode='constant', constant_values=0)
    # 获取卷积核的参数
    n_kernels, C_k, H_k, W_k = W.shape
    #输出的维度
    H_out = (H_in + 2 * padding - H_k) // stride + 1
    W_out = (W_in + 2 * padding - W_k) // stride + 1

    #用零初始化输出
    conv_output = np.zeros((N_in, n_kernels, H_out, W_out))

    conv_forward_accelerate(N_in,n_kernels,H_out,W_out,C_in,stride,
                            H_k,W_k,padding_input,W,conv_output,B)
    # 执行卷积
    # for n in range(N_in):                #样本数量
    #     for i in range(n_kernels):       #卷积核数量 
    #         for y in range(H_out):        #卷积核高度
    #             for x in range(W_out):    #卷积核宽度
    #                 convolved = 0
    #                 for c in range(C_in): #层数
    #                     # 确定输入的区域
    #                     y_start = y * stride
    #                     y_end =  y_start + H_k
    #                     x_start = x * stride
    #                     x_end = x_start + W_k

    #                     # 从填充输入中提取区域
    ##                     region = padding_input[n,c,y_start:y_end,x_start:x_end]
    #                     #执行卷积操作并添加偏置
    #                     convolved += np.sum(region * W[i,c,:,:])

    #                 #将卷积后的值赋给输出像素
    #                 conv_output[n,i,y,x] = convolved + B[i,y,x]
    #记录相关变量方便反向传播
    conv_cache = (padding_input, stride, W, B)
    return conv_output,conv_cache

#卷积层的反向传播的加速
@jit(nopython=True)
def conv_backward_accelerate(layer_number,C_out,H_out,W_out,stride,
                            H_in,W_in,dW,padding_input,dout,dB):
    for n in range(layer_number):
        for i in range(C_out):
            #dB[n] = np.mean(dout[:,n,:,:],axis=0)
            for y in range(H_out):
                for x in range(W_out):
                    y_start = y * stride
                    y_end = y_start + H_in
                    x_start = x * stride
                    x_end = x_start + W_in 
                    dW[n,i,y,x] = np.sum(padding_input[:,i,y_start:y_end,x_start:x_end] * dout[:,n,:,:]) / 5000
                    dB[n,y,x] = np.sum(dout[:,n,y,x]) / 5000

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
    padding_output = np.pad(dout, ((0, 0), (0,0),(2, 2), (2, 2)), 
                    mode='constant', constant_values=0)
    _n, _c, _h, _w = padding_output.shape
    dX = np.zeros((_n, _c, _h, _w))
    N,C_in,H_in,W_in = dout.shape
    padding_input,stride,W,B = conv_cache
    layer_number,C_out,H_out,W_out = W.shape
    dW = np.zeros((layer_number,C_out,H_out,W_out))
    dB = B
    conv_backward_accelerate(layer_number,C_out,H_out,W_out,stride,
                            H_in,W_in,dW,padding_input,dout,dB)
    # for n in range(layer_number):
    #     for i in range(C_out):
    #         #dB[n] = np.mean(dout[:,n,:,:],axis=0)
    #         for y in range(H_out):
    #             for x in range(W_out):
    #                 y_start = y * stride
    #                 y_end = y_start + H_in
    #                 x_start = x * stride
    #                 x_end = x_start + W_in 
    #                 dW[n,i,y,x] = np.sum(padding_input[:,i,y_start:y_end,x_start:x_end] * dout[:,n,:,:]) / N
    #                 dB[n,y,x] = np.sum(dout[:,n,y,x]) / N
    #翻转W
    # flipped_W = np.flip(np.flip(W, 0),1)
    # for i in range(C_out):
    #     for n in range(layer_number):
    #         for y in range(_h-2):
    #             for x in range(_w-2):
    #                 y_start = y * stride
    #                 y_end = y_start + H_out
    #                 x_start = x * stride
    #                 x_end = x_start + W_out
    #                 dX[:,i,y,x] += np.sum(padding_output[:,n,y_start:y_end,x_start:x_end] * flipped_W[n,i,:,:])
    return dX,dW,dB
#池化的加速
@jit(nopython=True)
def max_pool_forward_accelerate(N_in,n_kernels,H_out,W_out,pool_stride,pool_size,
                                X,max_array,pool_output):
    for n in range(N_in):
        for i in range(n_kernels):
            for y in range(H_out):
                for x in range(W_out):
                    y_start = y * pool_stride
                    y_end = y_start + pool_size
                    x_start = x * pool_stride
                    x_end = x_start + pool_size
                    my_max = -1
                    #选择当前池化窗口的区域
                    region = X[n,i,y_start:y_end,x_start:x_end]
                    #获取最大值索引
                    max_index = [0,0]
                    for t in range(region.shape[0]):
                        for h in range(region.shape[1]):
                            if region[t,h] > my_max:
                                my_max = region[t,h]
                                max_index[0] = t
                                max_index[1] = h
                    #max_index = np.unravel_index(region.argmax(), region.shape)
                    max_array[n,i,y_start+max_index[0],x_start+max_index[1]] = 1
                    #对当前窗口取最大值
                    pool_output[n,i,y,x] = np.max(region)
#池化
def max_pool_forward(X, pool_size, pool_stride,pool_padding):
    """
    执行最大池化操作的前向传播。
    
    参数:
    - X: 输入数据,形状为 (N,n_kernels,H,W), 其中H 是高度,W 宽度,n_kernels是通道数,N是样品数量
    - pool_size: 池化窗口的大小
    - pool_stride: 池化操作的步幅
    - pool_padding: 为输入添加到边界的填充的像素数。

    返回:
    - pool_output: 池化后的输出数据。
    - cache: 缓存一些值,用于反向传播时计算梯度(max_array, pool_size, pool_stride)。
    max_array记录最大值索引位置,pool_size是池化大小,pool_stride是移动步长。
    """
    #获取输入数据的维度
    N_in,n_kernels,H_in,W_in = X.shape
    #计算输出数据的高度和宽度
    H_out = (H_in + pool_padding - pool_size) // pool_stride + 1
    W_out = (W_in + pool_padding - pool_size) // pool_stride + 1

    #用零初始化数据
    pool_output = np.zeros((N_in,n_kernels,H_out,W_out))
    max_array = np.zeros((N_in,n_kernels,H_in,W_in))
    max_pool_forward_accelerate(N_in,n_kernels,H_out,W_out,pool_stride,pool_size,
                                X,max_array,pool_output)
    #进行池化操作
    # for n in range(N_in):
    #     for i in range(n_kernels):
    #         for y in range(H_out):
    #             for x in range(W_out):
    #                 y_start = y * pool_stride
    #                 y_end = y_start + pool_size
    #                 x_start = x * pool_stride
    #                 x_end = x_start + pool_size

    #                 #选择当前池化窗口的区域
    # #                region = X[n,i,y_start:y_end,x_start:x_end]
    #                 #获取最大值索引
    #                 max_index = np.unravel_index(region.argmax(), region.shape)
    #                 max_array[n,i,y_start+max_index[0],x_start+max_index[1]] = 1
    #                 #对当前窗口取最大值
    #                 pool_output[n,i,y,x] = np.max(region)

    #缓存值,反向传播时使用
    cache = (max_array, pool_size, pool_stride)

    return pool_output,cache

#展平
def flatten(X):
    """
    展平操作将输入数据的多维特征图转换为一维特征向量。

    参数:
    - X: 输入数据,形状为 (N,C,H, W),其中
      N 是批量大小,
      C 是通道数,
      H 是特征图的高度,
      W 是特征图的宽度。

    返回:
    - out: 展平后的输出数据,形状为 (N, C*H*W)。
    """
    N,C, H, W = X.shape  # 获取输入数据的维度
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
    N, C, H, W = pool_output.shape  # 获取输入数据的维度
    out = X.reshape(N, C, H, W)  # 使用reshape函数将数据展平,-1表示自动计算该维度的大小
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
    - fully_out: 全连接层的输出
    """
    fully_out = np.dot(X, W) + b  # 输入数据和权重矩阵的点积后加上偏置
    return fully_out

#池化层的反向传播的加速
@jit(nopython=True)
def max_pool_backward_accelerate(N_out,n_kernels_in,H_in,W_in,pool_stride,pool_size,
                                 max_array,dX):
    for n in range(N_out):
        for i in range(n_kernels_in):
            for y in range(H_in):
                for x in range(W_in):
                    y_start = y * pool_stride
                    y_end = y_start + pool_size
                    x_start = x * pool_stride
                    x_end = x_start + pool_size
                    max_array[n, i, y_start:y_end, x_start:x_end] *= dX[n, i, y, x]
#池化层的反向传播
def max_pool_backward(dX,cache):
    """
    最大池化层的反向传播。
    """
    max_array, pool_size, pool_stride = cache
    N_out,n_kernels,H_out,W_out = max_array.shape
    #d_out = np.zeros((N_out,H_out,W_out,n_kernels))
    N_in,n_kernels_in,H_in,W_in = dX.shape
    max_pool_backward_accelerate(N_out,n_kernels_in,H_in,W_in,pool_stride,pool_size,
                                 max_array,dX)
    # for n in range(N_out):
    #     for i in range(n_kernels_in):
    #         for y in range(H_in):
    #             for x in range(W_in):
    #                 y_start = y * pool_stride
    #                 y_end = y_start + pool_size
    #                 x_start = x * pool_stride
    #                 x_end = x_start + pool_size
    #                 max_array[n, i, y_start:y_end, x_start:x_end] *= dX[n, i, y, x]
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
    my_shape = [] #记录层数
    conv_initWeights(X,W,B)
    Y_hat = np.zeros_like(Y)
    start_time = time.time()
    for i in range(iterations):
        if i > 1000:
            lr = 0.001
        #第一层卷积
        conv_output_first,conv_cache_first = conv_forward(X,W[0],B[0],conv_stride[0],conv_padding[0])
        #ReLU激活函数
        relu_output_first = relu_forward(conv_output_first)
        my_shape.append({"第一层卷积激活后的维度：":relu_output_first.shape})
        #池化
        pool_output_first,pool_cache_first = max_pool_forward(relu_output_first, pool_size[0], pool_stride[0], pool_padding[0])

        #展平
        flatten_out = flatten(pool_output_first)
        my_shape.append({"展平后的维度：":flatten_out.shape})

        #开始反向传播

        #全连接层的反向传播
        #初始化全连接层的参数W，B
        if i == 0:
            fully_initWeights(W, B, flatten_out, out = 10)
        #前向传播到全连接层
        fully_out = fully_connected_forward(flatten_out, W[-1],B[-1])
        my_shape.append({"全连接后的维度：":flatten_out.shape})
        #softmax激活函数
        Y_hat = softmax(fully_out)
        #softmax激活函数反向传播
        dX = (Y_hat - Y) @ W[-1].T
        my_shape.append({"全连接反向传播后的维度：":dX.shape})
        #对全连接层的W，B求偏导
        dw = (flatten_out.T @ (Y_hat - Y)) / 5000
        db = np.mean((Y_hat - Y),axis=0)
        #更新全连接层的W，B
        W[-1] = W[-1] - lr * dw
        B[-1] = B[-1] - lr * db

        #恢复展平，准备反向传播
        dX = d_flatten(pool_output_first, dX)
        my_shape.append({"恢复展平后的维度：":dX.shape})
        
        #池化的反向传播
        dZ = max_pool_backward(dX,pool_cache_first)

        #relu函数的反向传播
        dX = relu_backward(dZ,conv_output_first)
        my_shape.append({"第二层relu函数的反向传播后的维度: ":dX.shape})
        #卷积层的反向传播
        dX,dW,dB = conv_backward(dX,conv_cache_first)
        W[0] = W[0] - lr * dW
        B[0] = B[0] - lr * dB

        c = cost(Y_hat, Y)
        costs.append(c)
        end_time = time.time()
        my_time = end_time - start_time
        minutes = (int)(my_time // 60)
        seconds = (int)(my_time % 60)
        print(f"第 {i} 次迭代   花费时间:{minutes}m {seconds}s   损失值为:{c},")
        # c = cost(A[-1], Y, W)
        # dW, dB = networkBackward(Y, A, W)
        # W, B = updateWeights(W, B, dW, dB, lr)
    acc = test(Y_hat,Y)
    return W, B, costs

  
#处理数据
data = np.load("data.npy")
X = data[:,:-1].reshape(data.shape[0], 20, 20).transpose(0,2,1)
X = X.reshape(X.shape[0],1,20,20)
Y = data[:,-1].astype(np.int32)
Y = onehotEncoder(Y, 10)
W, B, costs = train(X, Y, lr, iterations)
plt.figure()
plt.plot(range(len(costs)), costs)
#test(Y_hat, Y)