# -*- coding:utf-8 -*-
import os, time, skimage, logging
import numpy as np
import tensorlayer as tl
from utils.response import JsonResponse
from utils.imagenet import *
from rest_framework import status

LOG = logging.getLogger(__name__)

class ImageControl(object):
    params = []
    x = tf.placeholder("float", [None, 224, 224, 3])
    network = Vgg19_simple_api(x)
    probs = tf.nn.softmax(network.outputs, name="prob")

    @classmethod
    def pred_image(cls, path):
        print(path)
        #1. 处理图片，调整格式
        processed_images = load_image(path)
        img1 = processed_images.reshape((1, 224, 224, 3))
        sess = tf.InteractiveSession()
        tl.layers.initialize_global_variables(sess)
        print("Restoring model from npz file")
        tl.files.assign_params(sess, cls.params, cls.network)
        start_time = time.time()
        prob = sess.run(cls.probs, feed_dict= {cls.x : img1})
        print("End time : %.5ss" % (time.time() - start_time))
        pred_result_top1, pred_result_top5 = print_prob(prob[0])
        return JsonResponse(data=pred_result_top5, code=status.HTTP_200_OK, desc='get identify success')

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
        # network = Vgg19(x)
# TODO make it env variables
ImageControl.pre_load("/notebooks/education/vgg19.npy")

    
