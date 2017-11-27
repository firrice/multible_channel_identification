# -*- coding: utf-8 -*-
#将所有图片均调整为230*80
import os
from PIL import Image


def resize(old_path , new_path , new_h , new_w):
    filelist = os.listdir(old_path)  # 获取文件下的所有文件名
    print len(filelist)
    for files in filelist:
        filetype = os.path.splitext(files)[1]  # 后缀名,是一个列表
        if(filetype != '.bmp'):
            continue
        old_pic_dir = old_path + files  # 原来的文件路径
        print old_pic_dir
        new_pic_dir = new_path + files  #将要保存的路径
        im = Image.open(old_pic_dir)
        out = im.resize((new_h , new_w) , Image.ANTIALIAS)
        out.save(new_pic_dir)

resize('/home/zf/caffe/multible_channel_identification/data/VIPeR/cam_b/' , '/home/zf/caffe/multible_channel_identification/data/VIPeR/cam_b/' , 80 , 230)

