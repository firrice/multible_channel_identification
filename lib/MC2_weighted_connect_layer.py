# -*- coding: utf-8 -*-
#这一层的作用主要是对经过5个通道后的特征图进行加权连接,每个通道结构均为[1 , 100]；
import sys

caffe_root = '/home/zf/caffe/'
sys.path.insert(0, caffe_root + 'python/')
import caffe
import numpy as np
import yaml
import cv2


class MC2_WeightedconnectLayer(caffe.Layer):
    def setup(self, bottom, top):
        self.final_feature = np.zeros((1 , 500))

    def reshape(self, bottom, top):
        top[0].reshape(1 , 500)

    def forward(self, bottom, top):  #权值连接方式只是简单的相加；
        self.final_feature[: , :100] = bottom[0].data[...]
        self.final_feature[: , 100:200] = bottom[1].data[...]
        self.final_feature[: , 200:300] = bottom[2].data[...]
        self.final_feature[: , 300:400] = bottom[3].data[...]
        self.final_feature[: , 400:500] = bottom[4].data[...]
        top[0].data[...] = self.final_feature


    def backward(self, top, propagate_down, bottom):
        pass
