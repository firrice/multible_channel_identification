# -*- coding: utf-8 -*-
#前一层的输出为[1 , 3 , 230 , 80]，将每张图片的特征图(大小为[1 , 32 , 77 , 27])分成5个通道；
import sys

caffe_root = '/home/zf/caffe/'
sys.path.insert(0, caffe_root + 'python/')
import caffe
import numpy as np
import yaml
import cv2


class MC3_DevideLayer(caffe.Layer):
    def setup(self, bottom, top):
        pass

    def reshape(self, bottom, top):
        top[0].reshape(1 , 32 , 20 , 27)
        top[1].reshape(1 , 32 , 20 , 27)
        top[2].reshape(1 , 32 , 20 , 27)
        top[3].reshape(1 , 32 , 20 , 27)
        top[4].reshape(1 , 32 , 77 , 27)

    def forward(self, bottom, top):
        #print "The shape of the global_body is:{}.".format(*bottom[0].shape)
        body = np.zeros((1 , 32 , 77 , 27))
        body = bottom[0].data[...]
        top[0].data[...] = body[: , : , :20 , :]
        top[1].data[...] = body[: , : , 20:40 , :]
        top[2].data[...] = body[: , : , 40:60 , :]
        top[3].data[...] = body[: , : , 57:77 , :]
        top[4].data[...] = body

    def backward(self, top, propagate_down, bottom):
        pass