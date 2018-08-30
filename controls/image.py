# -*- coding:utf-8 -*-
import os, time, inspect, skimage, logging
import numpy as np
import skimage.io
import skimage.transform
from skimage import io,data
from utils.response import JsonResponse
VGG_MEAN = [103.939, 116.779, 123.68]
LOG = logging.getLogger(__name__)

class ImageControl(object):

    @classmethod
    def load_image(cls, path):
        print(path)
        # load image
        img = skimage.io.imread(path)
        img = img / 255.0
        assert (0 <= img).all() and (img <= 1.0).all()
        # print "Original Image Shape: ", img.shape
        # we crop image from center
        short_edge = min(img.shape[:2])
        yy = int((img.shape[0] - short_edge) / 2)
        xx = int((img.shape[1] - short_edge) / 2)
        crop_img = img[yy: yy + short_edge, xx: xx + short_edge]
        # resize to 224, 224
        resized_img = skimage.transform.resize(crop_img, (224, 224))
        return JsonResponse(data=img, code=status.HTTP_200_OK, desc='get house success')
