# -*- coding: utf-8 -*-
#这一层的作用主要是计算triplet_loss,来自三个通道的结构均为[1 , 800]
import sys

caffe_root = '/home/zf/caffe/'
sys.path.insert(0, caffe_root + 'python/')
import caffe
import numpy as np
import yaml
import cv2


class TripletLossLayer(caffe.Layer):
    def setup(self, bottom, top):
        self.w = np.zeros((800 , 1))
        self.N = 3  #样本个数????
        self.a = -1  #tao1值
        self.b = 0.01  #tao2值
        self.c = 0.002  #beita值



    def reshape(self, bottom, top):
        top[0].reshape(1 , 1)


    def forward(self, bottom, top):  #计算triplet_loss
        self.I_0 = np.dot(bottom[0].data[...] , self.w)
        self.I_p = np.dot(bottom[1].data[...] , self.w)
        self.I_n = np.dot(bottom[2].data[...] , self.w)
        self.d_p = (self.I_0 - self.I_p) ** 2
        self.d_n = (self.I_0 - self.I_p) ** 2 - (self.I_0 - self.I_n) ** 2
        #print "The value of bottom is {}.".format(bottom[0].data[...].shape)
        #print "The value of I_0 is {}.".format(self.I_0.shape)
        #print "The value of w is {}.".format(self.w.shape)
        triplet_loss = (max(self.d_n , self.a) + self.c * max(self.d_p , self.b)) / self.N
        top[0].data[...] = triplet_loss
        print "The finnal triplet loss is {} .".format(triplet_loss)


    def backward(self, top, propagate_down, bottom):
        propagate_down[0] = true
        propagate_down[1] = true
        propagate_down[2] = true
        grad_I_0 = bottom[0].data[...]
        grad_I_p = bottom[1].data[...]
        grad_I_n = bottom[2].data[...]
        grad_d_p = 2 * (self.I_0 - self.I_p) * (grad_I_0 - grad_I_p)
        grad_d_n = 2 * (self.I_0 - self.I_p) * (grad_I_0 - grad_I_p) - 2 * (self.I_0 - self.I_n) * (grad_I_0 - grad_I_n)

        if(self.d_n > self.a):
            h1 = grad_d_n
        else:
            h1 = 0
        if(self.d_p > self.b):
            h2 = grad_d_p
        else:
            h2 = 0

        grad_L = h1 / self.N + h2 / self.N * self.c
        self.w = self.w - grad_L
        bottom[0].diff[...] = grad_L

