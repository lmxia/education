# -*- coding:utf-8 -*-
import os, time, inspect, skimage, logging
import numpy as np
from utils.response import JsonResponse
from utils.imagenet import *
from rest_framework import status
VGG_MEAN = [103.939, 116.779, 123.68]
LOG = logging.getLogger(__name__)

class ImageControl(object):
    params = []
    @classmethod
    def pred_image(cls, path):
        print(path)
        #1. 处理图片，调整格式
        processed_images = load_image(path)
        img1 = processed_images.reshape((1, 224, 224, 3))
        sess = tf.InteractiveSession()
        x = tf.placeholder("float", [None, 224, 224, 3])
        # network = Vgg19(x)
        network = Vgg19_simple_api(x)
        y = network.outputs
        probs = tf.nn.softmax(y, name="prob")
        tl.layers.initialize_global_variables(sess)
        print("Restoring model from npz file")
        tl.files.assign_params(sess, cls.params, network)
        start_time = time.time()
        prob = sess.run(probs, feed_dict= {x : img1})
        print("End time : %.5ss" % (time.time() - start_time))

        print_prob(prob[0])
        return JsonResponse(data=probs, code=status.HTTP_200_OK, desc='get identify success')

    @classmethod
    def pre_load(cls, npy_path="../vgg19.npy"):
        # You need to download the pre-trained model - VGG19 NPZ
        if not os.path.isfile(npy_path):
            print('''请从 https://github.com/machrisaa/tensorflow-vgg 下载 vgg19.npz 模型文件
                     make sure vgg19.npz located in root path
                    ''')
            return JsonResponse(data={}, code=status.HTTP_417_EXPECTATION_FAILED, desc="can't find vgg19.npy")
        npz = np.load(npy_path, encoding='latin1').item()

        for val in sorted( npz.items() ):
            W = np.asarray(val[1][0])
            b = np.asarray(val[1][1])
            print("  Loading %s: %s, %s" % (val[0], W.shape, b.shape))
            cls.params.extend([W, b])

# TODO make it env variables
ImageControl.pre_load("../vgg19.npy")

    
