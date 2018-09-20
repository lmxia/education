# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from datetime import datetime
import os
import numpy as np
import tensorflow as tf
from scipy.misc import imread, imsave
from models import Decoder, Encoder
import utils

alpha = 1
output_path = '../output/'

class StyleTransferService(object):
    def __init__(self, encode_path, decode_path):
        self.sess = None
        self.content_input = None
        self.style_input = None
        self.generated_img = None
        self.gragh = tf.Graph()
        self.init_session_handler()

    def init_session_handler(self):
        self.gragh.as_default()
        self.sess = tf.Session()

        encoder = Encoder()
        decoder = Decoder()

        self.content_input = tf.placeholder(tf.float32, shape=(1, None, None, 3), name='content_input')
        self.style_input = tf.placeholder(tf.float32, shape=(1, None, None, 3), name='style_input')

        # switch RGB to BGR
        content = tf.reverse(self.content_input, axis=[-1])
        style = tf.reverse(style_input, axis=[-1])
        # preprocess image
        content = encoder.preprocess(content)
        style = encoder.preprocess(style)

        # encode image
        # we should initial global variables before restore model
        enc_c_net = encoder.encode(content, 'content/')
        enc_s_net = encoder.encode(style, 'style/')

        # pass the encoded images to AdaIN
        target_features = utils.AdaIN(enc_c_net.outputs, enc_s_net.outputs, alpha=alpha)

        # decode target features back to image
        dec_net = decoder.decode(target_features, prefix="decoder/")

        generated_img = dec_net.outputs

        # deprocess image
        generated_img = encoder.deprocess(generated_img)

        # switch BGR back to RGB
        generated_img = tf.reverse(generated_img, axis=[-1])

        # clip to 0..255
        generated_img = tf.clip_by_value(generated_img, 0.0, 255.0)

        # sess.run(tf.global_variables_initializer())

        encoder.restore_model(self.sess, ENCODER_PATH, enc_c_net)
        encoder.restore_model(self.sess, ENCODER_PATH, enc_s_net)
        decoder.restore_model(self.sess, DECODER_PATH, dec_net)



    def transfer(self, content_file, style_file):
        start_time = datetime.now()
        content_image = imread(content_file, mode='RGB')
        style_image = imread(style_file, mode='RGB')
        content_tensor = np.expand_dims(content_image, axis=0)
        style_tensor = np.expand_dims(style_image, axis=0)
        result = self.sess.run(self.generated_img, feed_dict={self.content_input: content_tensor, self.style_input: style_tensor})
        result_name = os.path.join(output_path, s.split('.')[0] + '_' + c.split('.')[0] + '.jpg')
        print(result_name, ' is generated')
        imsave(result_name, result[0])
        elapsed_time = datetime.now() - start_time
        return result

ENCODER_PATH = '../pretrained_vgg19_encoder_model.npz'
DECODER_PATH = '../pretrained_vgg19_decoder_model.npz'

transfer_service = StyleTransferService(ENCODER_PATH, DECODER_PATH)


def index(request):
    return HttpResponse(
        "You should POST /style/transfer/ .")


# Disable CSRF, refer to https://docs.djangoproject.com/en/dev/ref/csrf/#edge-cases
@csrf_exempt
def transfer(request):
    if request.method == 'POST':
        parser_classes = (FileUploadParser,)
        content_file = request.data['content_file']
        style_file = request.data['style_file']
        # The post body should be json, such as {"key": [1.0, 2.0], "features": [[10,10,10,8,6,1,8,9,1], [6,2,1,1,1,1,7,1,1]]}

        result = transfer_service.transfer(content_file, style_file)
        return HttpResponse("Success to predict cancer, result: {}".format(
            result))
    else:
        return HttpResponse("Please use POST to request with data")