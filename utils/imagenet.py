# -*- coding:utf-8 -*-
from data.imagenet_classes import *
import numpy as np
import skimage.transform
from skimage import io
import os, time
import tensorflow as tf
from tensorlayer.layers import *
VGG_MEAN = [103.939, 116.779, 123.68]
#get the properbilities
def print_prob(prob):
    synset = class_names
    # print prob
    pred = np.argsort(prob)[::-1]
    # Get top1 label
    top1 = synset[pred[0]]
    print("Top1: ", top1, prob[pred[0]])
    # Get top5 label
    top5 = [(synset[pred[i]], prob[pred[i]]) for i in range(5)]
    print("Top5: ", top5)
    return top1, top5

# preprocess the images
def load_image(path):
    print(path)
    # load image
    img = io.imread(path) / 255.0
    assert (0 <= img).all() and (img <= 1.0).all()
    print "Original Image Shape: ", img.shape
    # we crop image from center
    short_edge = min(img.shape[:2])
    yy = int((img.shape[0] - short_edge) / 2)
    xx = int((img.shape[1] - short_edge) / 2) 
    print "Change Image Shape:{} {} ".format(yy, xx)
    crop_img = img[yy: yy + short_edge, xx: xx + short_edge]
    # resize to 224, 224
    resized_img = skimage.transform.resize(crop_img, (224, 224))
    return resized_img

# Vgg19的api
def Vgg19_simple_api(rgb):
    """
    Build the VGG 19 Model
    Parameters
    -----------
    rgb : rgb image placeholder [batch, height, width, 3] values scaled [0, 1]
    """
    start_time = time.time()
    print("build model started")
    rgb_scaled = rgb * 255.0
    # Convert RGB to BGR
    if tf.__version__ <= '0.11':
        red, green, blue = tf.split(3, 3, rgb_scaled)
    else: # TF 1.0
        print(rgb_scaled)
        red, green, blue = tf.split(rgb_scaled, 3, 3)
    assert red.get_shape().as_list()[1:] == [224, 224, 1]
    assert green.get_shape().as_list()[1:] == [224, 224, 1]
    assert blue.get_shape().as_list()[1:] == [224, 224, 1]
    if tf.__version__ <= '0.11':
        bgr = tf.concat(3, [
            blue - VGG_MEAN[0],
            green - VGG_MEAN[1],
            red - VGG_MEAN[2],
        ])
    else:
        bgr = tf.concat([
            blue - VGG_MEAN[0],
            green - VGG_MEAN[1],
            red - VGG_MEAN[2],
        ], axis=3)
    assert bgr.get_shape().as_list()[1:] == [224, 224, 3]

    """ input layer """
    net_in = InputLayer(bgr, name='input')
    """ conv1 """
    network = Conv2d(net_in, n_filter=64, filter_size=(3, 3),
                strides=(1, 1), act=tf.nn.relu,padding='SAME', name='conv1_1')
    network = Conv2d(network, n_filter=64, filter_size=(3, 3),
                strides=(1, 1), act=tf.nn.relu,padding='SAME', name='conv1_2')
    network = MaxPool2d(network, filter_size=(2, 2), strides=(2, 2),
                padding='SAME', name='pool1')
    """ conv2 """
    network = Conv2d(network, n_filter=128, filter_size=(3, 3),
                strides=(1, 1), act=tf.nn.relu,padding='SAME', name='conv2_1')
    network = Conv2d(network, n_filter=128, filter_size=(3, 3),
                strides=(1, 1), act=tf.nn.relu,padding='SAME', name='conv2_2')
    network = MaxPool2d(network, filter_size=(2, 2), strides=(2, 2),
                padding='SAME', name='pool2')
    """ conv3 """
    network = Conv2d(network, n_filter=256, filter_size=(3, 3),
                strides=(1, 1), act=tf.nn.relu,padding='SAME', name='conv3_1')
    network = Conv2d(network, n_filter=256, filter_size=(3, 3),
                strides=(1, 1), act=tf.nn.relu,padding='SAME', name='conv3_2')
    network = Conv2d(network, n_filter=256, filter_size=(3, 3),
                strides=(1, 1), act=tf.nn.relu,padding='SAME', name='conv3_3')
    network = Conv2d(network, n_filter=256, filter_size=(3, 3),
                strides=(1, 1), act=tf.nn.relu,padding='SAME', name='conv3_4')
    network = MaxPool2d(network, filter_size=(2, 2), strides=(2, 2),
                padding='SAME', name='pool3')
    """ conv4 """
    network = Conv2d(network, n_filter=512, filter_size=(3, 3),
                strides=(1, 1), act=tf.nn.relu,padding='SAME', name='conv4_1')
    network = Conv2d(network, n_filter=512, filter_size=(3, 3),
                strides=(1, 1), act=tf.nn.relu,padding='SAME', name='conv4_2')
    network = Conv2d(network, n_filter=512, filter_size=(3, 3),
                strides=(1, 1), act=tf.nn.relu,padding='SAME', name='conv4_3')
    network = Conv2d(network, n_filter=512, filter_size=(3, 3),
                strides=(1, 1), act=tf.nn.relu,padding='SAME', name='conv4_4')
    network = MaxPool2d(network, filter_size=(2, 2), strides=(2, 2),
                padding='SAME', name='pool4')
    """ conv5 """
    network = Conv2d(network, n_filter=512, filter_size=(3, 3),
                strides=(1, 1), act=tf.nn.relu,padding='SAME', name='conv5_1')
    network = Conv2d(network, n_filter=512, filter_size=(3, 3),
                strides=(1, 1), act=tf.nn.relu,padding='SAME', name='conv5_2')
    network = Conv2d(network, n_filter=512, filter_size=(3, 3),
                strides=(1, 1), act=tf.nn.relu,padding='SAME', name='conv5_3')
    network = Conv2d(network, n_filter=512, filter_size=(3, 3),
                strides=(1, 1), act=tf.nn.relu,padding='SAME', name='conv5_4')
    network = MaxPool2d(network, filter_size=(2, 2), strides=(2, 2),
                padding='SAME', name='pool5')
    """ fc 6~8 """
    network = FlattenLayer(network, name='flatten')
    network = DenseLayer(network, n_units=4096, act=tf.nn.relu, name='fc6')
    network = DenseLayer(network, n_units=4096, act=tf.nn.relu, name='fc7')
    network = DenseLayer(network, n_units=1000, act=tf.identity, name='fc8')
    print("build model finished: %fs" % (time.time() - start_time))
    return network