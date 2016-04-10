# -*- coding:utf-8 -*-

import numpy as np
import chainer
from chainer import Function, Variable, optimizers
from chainer import Link, Chain
import chainer.functions as F
import chainer.links as L

# main
if __name__ == '__main__':
    x_data = np.array([[1.0, 0.5]], dtype=np.float32)
    t_data = np.array([0])

    model = Chain(layer1=L.Linear(2, 2, initialW=np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)),
            layer2=L.Linear(2, 2, initialW=np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)))

    optimizer = optimizers.SGD()
    optimizer.setup(model)

    x = Variable(x_data)
    t = Variable(t_data)
    u1 = model.layer1(x)
    z1 = F.sigmoid(u1)
    u2 = model.layer2(z1)
    y = F.softmax(u2)

    print "x=\n" + str(x.data)
    print "w1=\n" + str(model.layer1.W.data)
    print "b1=\n" + str(model.layer1.b.data)
    print "u1=\n" + str(u1.data)
    print "z1=\n" + str(z1.data)
    print "w2=\n" + str(model.layer2.W.data)
    print "b2=\n" + str(model.layer2.b.data)
    print "u2=\n" + str(u2.data)
    print "y=\n" + str(y.data)


    loss = F.softmax_cross_entropy(u2, t)

    print "loss=\n" + str(loss.data)

    optimizer.zero_grads()

    loss.backward()

    optimizer.weight_decay(0.05)
    optimizer.update()

    print "w1=\n" + str(model.layer1.W.data)
    print "b1=\n" + str(model.layer1.b.data)
    print "w2=\n" + str(model.layer2.W.data)
    print "b2=\n" + str(model.layer2.b.data)
