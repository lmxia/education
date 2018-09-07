# -*- coding:utf-8 -*-
from __future__ import print_function
import logging
from utils.response import JsonResponse
from rest_framework import status
import tensorflow as tf
import numpy as np

LOG = logging.getLogger(__name__)

class ArcControl(object):

    hiddenDim = 256
    num_input = 1
    
    sess = tf.InteractiveSession()
    X = tf.placeholder(tf.float32, [None, num_input])
    Y = tf.placeholder(tf.float32, [None, num_input])
    
    W = tf.Variable(tf.truncated_normal([num_input, hiddenDim], stddev = 0.1))
    b = tf.Variable(tf.constant(0.1, shape = [1,hiddenDim]))

    W2 = tf.Variable(tf.truncated_normal([hiddenDim,1], stddev = 0.1))
    b2 = tf.Variable(tf.constant(0.1, shape = [1]))

    hidden = tf.nn.sigmoid(tf.matmul(X,W) + b)
    y = tf.matmul(hidden, W2) + b2

    step = tf.Variable(0,trainable=False)
    rate = tf.train.exponential_decay(0.15, step, 1, 0.9999)

    optimizer = tf.train.AdamOptimizer(rate)
    train = optimizer.minimize(loss,global_step = step)
    init = tf.global_variables_initializer()
    loss = tf.reduce_mean(tf.square(y - Y))#最小均方误差
    sess.run(init)
    @classmethod
    def regression(cls, data):
        x_list = []
        y_list = []
        for item in data:
            print(item.get("x"))
            print(item.get("y"))
            x_list.append(eval(item.get("x")))
            y_list.append(eval(item.get("y")))
        train_X = np.array(x_list)[:,np.newaxis]
        train_Y = np.array(y_list)[:,np.newaxis]
        print(train_X.shape)

        for time in range(0,10001):
            cls.train.run({cls.X:train_X, cls.Y:train_Y},cls.sess)
            if time % 1000 == 0:
                print('train time:', time, 'loss is ', cls.loss.eval({cls.X:train_X, cls.Y:train_Y},cls.sess))
        back = cls.y.eval({cls.X:train_X, cls.Y:train_Y},cls.sess)[:,0]
        return JsonResponse(data=back, code=status.HTTP_200_OK, desc='get success') 