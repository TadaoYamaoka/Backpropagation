# -*- coding:utf-8 -*-
import numpy as np

# ニューラルネットワークのパラメータ
W1 = np.array([[0.1, 0.3], [0.2, 0.4]])
W2 = np.array([[0.1, 0.3], [0.2, 0.4]])

learning_rate = 0.005

def sigmoid(u):
    return 1 / (1 + np.exp(-u))

def softmax(u):
    e = np.exp(u)
    return e / np.sum(e)

# 順伝播
def forward(x):
    global W1
    global W2
    u1 = x.dot(W1)
    z1 = sigmoid(u1)
    u2 = z1.dot(W2)
    y = softmax(u2)
    return y, z1

# 逆伝播
def back_propagation(x, z1, y, d):
    global W1
    global W2
    delta2 = y - d
    grad_W2 = z1.T.dot(delta2)

    sigmoid_dash = z1 * (1 - z1)
    delta1 = delta2.dot(W2.T) * sigmoid_dash
    grad_W1 = x.T.dot(delta1)

    W2 -= learning_rate * grad_W2
    W1 -= learning_rate * grad_W1

# main
if __name__ == '__main__':
    # 順伝播
    x = np.array([[1, 0.5]])
    y, z1 = forward(x)

    print "z1"
    print z1
    print "y"
    print y

    # 誤差逆伝播
    d = np.array([[1, 0]])
    back_propagation(x, z1, y, d)

    print "W1"
    print W1
    print "W2"
    print W2
