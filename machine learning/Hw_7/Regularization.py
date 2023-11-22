# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

def read_data(addr):
    data = np.loadtxt(addr, delimiter=',')

    n = data.shape[0]
    
    ###### You may modify this section to change the model
    X = np.concatenate([np.ones([n, 1]),
                        np.expand_dims(np.power(data[:,0],4), axis=1),
                        np.expand_dims(np.power(data[:,1],3), axis=1),
                        np.expand_dims(np.power(data[:,2],2), axis=1),
                        np.expand_dims(np.power(data[:,3],4), axis=1),
                        np.expand_dims(np.power(data[:,4],1), axis=1),
                        np.expand_dims(np.power(data[:,5],2), axis=1),
                        np.expand_dims(np.power(data[:,6],3), axis=1),
                        np.expand_dims(np.power(data[:,7],2), axis=1)],
                        axis=1)
    ###### You may modify this section to change the model

    Y = None
    if "train" in addr:
        Y = np.expand_dims(data[:, -1], axis=1)
    
    return (X,Y,n)

def cost_gradient(W, X, Y, n, lambd):
    z = -1 * (X @ W)
    y0 = 1 / (np.exp(z) + 1)  # y hat
    y1 = (y0 - Y) / n
    G = X.T @ y1 + lambd * W###### Gradient
    j = (1 / (2 * n)) * np.sum((y0 - Y)**2) + (lambd // 2) * np.sum(W ** 2)###### cost with respect to current W
    
    return (j, G)

def train(W, X, Y, lr, n, iterations, lambd):
    J = np.zeros([iterations, 1])

    for i in range(iterations):
        (J[i], G) = cost_gradient(W, X, Y, n, lambd)
        W = W - lr*G
    err = error(W, X, Y)
    
    return (W,J,err)

def error(W, X, Y):
    Y_hat = 1 / (1 + np.exp(-X@W))
    Y_hat[Y_hat<0.5] = 0
    Y_hat[Y_hat>0.5] = 1
    
    return (1-np.mean(np.equal(Y_hat, Y)))

def predict(W):
    (X, _, _) = read_data("test_data.csv")
    
    Y_hat = 1 / (1 + np.exp(-X@W))
    Y_hat[Y_hat<0.75] = 0
    Y_hat[Y_hat>0.75] = 1
    
    idx = np.expand_dims(np.arange(1,201), axis=1)
    np.savetxt("predict.csv", np.concatenate([idx, Y_hat], axis=1), header = "Index,ID", comments='', delimiter=',')
    
iterations = 200000###### Training loops
lr = 0.038###### Learning rate
lambd = 0.009##### Lambda to control the weight of Regularization part

(X, Y, n) = read_data("train.csv")
W = np.random.random([X.shape[1], 1])

(W,J,err) = train(W, X, Y, lr, n, iterations, lambd)
print(err)

plt.figure()
plt.plot(range(iterations), J)
plt.show()
predict(W)