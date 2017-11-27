# -*- coding: utf-8 -*-
#data input,three  images,two positive samples and one negative samples
import json
import time
import pickle
import scipy.misc
import skimage.io
import caffe

import numpy as np
import os.path as osp

from xml.dom import minidom
from random import shuffle
import random
from threading import Thread
from PIL import Image

from tools import SimpleTransformer


class MCDataLayer(caffe.Layer):

    """
    This is a simple synchronous datalayer for training a multilabel model on
    PASCAL.
    """

    def setup(self, bottom, top):

        self.top_names = ['image_1', 'image_2' , 'image_3']

        # === Read input parameters ===

        # 加载python层中的params_str参数
        params = eval(self.param_str)

        # 检查参数有效性
        check_params(params)

        # 储存batchsize大小，这里为3，即一次性输入三张图片；
        self.batch_size = params['batch_size']

        # 使用BatchLoader类加载图片,batch_loader是一个对象；
        self.batch_loader = BatchLoader(params, None)

        # === reshape tops ===
        # since we use a fixed input image size, we can shape the data layer
        # once. Else, we'd have to do it in the reshape call.
        top[0].reshape(
            1, 3, params['im_shape'][0] , params['im_shape'][1])  #[1,3,230,80]
        top[1].reshape(
            1, 3, params['im_shape'][0] , params['im_shape'][1])
        top[2].reshape(
            1, 3, params['im_shape'][0] , params['im_shape'][1])

        #print "The shape of the top[0] is:{}.".format(top[0].shape)
        print_info("MCDataLayer", params)

    def forward(self, bottom, top):
        """
        Load data.
        """
        for itt in range(self.batch_size):
            # Use the batch loader to load the next triplet_images.
            im = self.batch_loader.load_next_three_images()

            # Add directly to the caffe data layer,分别输出三张经过预处理的图片
            top[0].data[...] = im[0]
            top[1].data[...] = im[1]
            top[2].data[...] = im[2]

    def reshape(self, bottom, top):
        """
        There is no need to reshape the data, since the input is of fixed size
        (rows and columns)
        """
        pass

    def backward(self, top, propagate_down, bottom):
        """
        These layers does not back propagate
        """
        pass


class BatchLoader(object):

    """
    This class abstracts away the loading of images.
    Images can either be loaded singly, or in a batch. The latter is used for
    the asyncronous data layer to preload batches while other processing is
    performed.
    """

    def __init__(self, params, result):
        self.result = result
        self.batch_size = params['batch_size']
        self.viper_root = params['viper_root'] #'/home/zf/caffe/multible_channel_identification/data/VIPeR'
        self.im_shape_h = params['im_shape'][0]
        self.im_shape_w = params['im_shape'][1]
        # get list of image indexes.
        list_file_a = params['split'][0] + '.txt'  #split:cam_a或者cam_b
        list_file_b = params['split'][1] + '.txt'
        self.indexlist_a = [line.rstrip('\n\r') for line in open(osp.join(params['viper_root'][0] , list_file_a))] #0000到0631
        self.indexlist_b = [line.rstrip('\n\r') for line in open(osp.join(params['viper_root'][1] , list_file_b))]
        self.i_a = 0  # current image
        # this class does some simple data-manipulations
        self.transformer = SimpleTransformer()

        print "BatchLoader initialized with {} images".format(
            len(self.indexlist_a)) #632

    def load_next_three_images(self):
        """
        Load the next image in a batch.
        """
        # Did we finish an epoch?
        if self.i_a == len(self.indexlist_a):
            self.i_a = 0
            shuffle(self.indexlist_a)

        # Load three images
        index_a = self.indexlist_a[self.i_a]  # Get the image index of cam_a
        i_b = 0
        while(self.indexlist_b[i_b] != index_a):
            i_b = i_b + 1
            if(i_b == len(self.indexlist_a)):
                print "there is some error in the data"
                return
        index_b = self.indexlist_b[i_b]  # Get the image index of cam_b
        i_c = random.randint(0, 631)
        if (i_c == self.i_a):
            i_c = random.randint(0, 631)
        index_c = self.indexlist_a[i_c]  #Get the positive
        image_file_a_name = index_a + '.bmp'
        image_file_b_name = index_b + '.bmp'
        image_file_c_name = index_c + '.bmp'

        im = np.zeros((3 , 230 , 80 , 3))
        im[0] = np.asarray(Image.open(osp.join(self.viper_root[0] , image_file_a_name)))
        im[1] = np.asarray(Image.open(osp.join(self.viper_root[1] , image_file_b_name)))
        im[2] = np.asarray(Image.open(osp.join(self.viper_root[0] , image_file_c_name)))

        #print "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"

        pic = np.zeros((3 , 3 , 230 , 80))
        pic[0] = self.transformer.preprocess(im[0])
        pic[1] = self.transformer.preprocess(im[1])
        pic[2] = self.transformer.preprocess(im[2])
        return pic



def check_params(params):
    """
    A utility function to check the parameters for the data layers.
    """
    assert 'split' in params.keys(
    ), 'Params must include split (train, val, or test).'

    required = ['batch_size', 'viper_root', 'im_shape']
    for r in required:
        assert r in params.keys(), 'Params must include {}'.format(r)


def print_info(name, params):
    """
    Output some info regarding the class
    """
    print "{} initialized for split: {}, with bs: {}, im_shape_w: {}, im_shape_h: {}.".format(
        name,
        params['split'],
        params['batch_size'],
        params['im_shape'][0],
        params['im_shape'][1])