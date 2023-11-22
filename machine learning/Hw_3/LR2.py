# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

def cost_gradient(W, X, Y, n):
      z = np.dot(X, W)
      y_hat = 1.0 / (1.0 + np.exp(-z))
      G = (1 / n) * np.dot(X.T, (y_hat - Y))###### Gradient
      j = (1/n)*np.sum(-Y * np.log(y_hat) - (1 - Y) * np.log(1 - y_hat))###### cost with respect to current W
      #j = (1 / (2*n)) * np.sum((y_hat-Y)**2) + lam * np.sum(W**2)
      return (j, G)

def gradientDescent(W, X, Y, n, lr, iterations):
      #n = np.size(Y)
      X[:, 1] = X[:, 1] * X[:, 1]
      J = np.zeros([iterations, 1])
      
      for i in range(iterations):
          (J[i], G) = cost_gradient(W, X, Y, n)
          W = W - lr * G ###### Update W based on gradient

      return (W,J)

iterations = 2000###### Training loops
lr = 0.03###### Learning rate
#lam = 20

data = np.loadtxt('LR2.txt', delimiter=',')

n = data.shape[0]
W = np.random.random([3, 1])
X = np.concatenate([np.ones([n, 1]), data[:,0:2]], axis=1)
Y = np.expand_dims(data[:, 2], axis=1)

(W,J) = gradientDescent(W, X, Y, n, lr, iterations)

#Draw figure
idx0 = (data[:, 2]==0)
idx1 = (data[:, 2]==1)

plt.figure()
plt.ylim(-12,12)
plt.plot(data[idx0,0], data[idx0,1],'go')
plt.plot(data[idx1,0], data[idx1,1],'rx')

x1 = np.arange(-10,10,0.2)
y1 = (W[0] + W[1]*x1*x1) / -W[2]
plt.plot(x1, y1)

plt.figure()
plt.plot(range(iterations), J)
plt.show()