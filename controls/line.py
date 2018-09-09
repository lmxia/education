# -*- coding:utf-8 -*-
from __future__ import print_function
import logging
from utils.response import JsonResponse
from rest_framework import status
import tensorflow as tf
import numpy

LOG = logging.getLogger(__name__)

class LineControl(object):
    learning_rate = 0.02
    training_epochs = 500
    display_step = 50
    back = {"w":"0", "b":"0"}
    rng = numpy.random
    x_list = []
    y_list = []
    sess = tf.Session()
    # tf Graph Input，tf图输入
    X = tf.placeholder("float")
    Y = tf.placeholder("float")

    # Set model weights，初始化网络模型的权重
    W = tf.Variable(rng.randn(), name="weight")
    b = tf.Variable(rng.randn(), name="bias")

    # Construct a linear model，构造线性模型
    pred = tf.add(tf.multiply(X, W), b)
    n_samples = 10
    # Mean squared error，损失函数：均方差
    cost = tf.reduce_sum(tf.pow(pred-Y, 2))/(2*n_samples)
    # Gradient descent， 优化方式：梯度下降
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
    # Initialize the variables (i.e. assign their default value)，初始化所有图节点参数
    init = tf.global_variables_initializer()
    sess.run(init)
    print("initialed...")
    @classmethod
    def regression(cls, data):
        for item in data:
            cls.x_list.append(eval(item.get("x")))
            cls.y_list.append(eval(item.get("y")))
        
        train_X = numpy.array(cls.x_list)
        train_Y = numpy.array(cls.y_list)
        cls.n_samples = train_X.shape[0]

        # Fit all training data
        for epoch in range(cls.training_epochs):
            for (x, y) in zip(train_X, train_Y):
                cls.sess.run(cls.optimizer, feed_dict={cls.X: x, cls.Y: y})

            #Display logs per epoch step
            # if (epoch+1) % cls.display_step == 0:
            #     c = cls.sess.run(cls.cost, feed_dict={cls.X: train_X, cls.Y:train_Y})
            #     print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(c),"W=", cls.sess.run(cls.W), "b=", sess.run(b))
                    
        print("Optimization Finished!")
        training_cost = cls.sess.run(cls.cost, feed_dict={cls.X: train_X, cls.Y: train_Y})
        print("Training cost=", training_cost, "W=", cls.sess.run(cls.W), "b=", cls.sess.run(cls.b), '\n')
        back["w"] = cls.sess.run(cls.W)
        back["b"] = cls.sess.run(cls.b)
        return JsonResponse(data=back, code=status.HTTP_200_OK, desc='get house success') 