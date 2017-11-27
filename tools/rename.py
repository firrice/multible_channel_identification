# -*- coding: utf-8 -*-
import os


def rename(path, newname):
    filelist = os.listdir(path)  # 获取文件下的所有文件名
    print len(filelist)
    m = 0
    for files in filelist:
        Olddir = path + files  # 原来的文件路径
        filename = os.path.splitext(files)[0]  # 文件名
        # print filename
        filetype = os.path.splitext(files)[1]  # 后缀名,是一个列表
        Newdir = os.path.join(path, newname + filetype) % m  # 这里由于filetype是一个列表，因此不能用Newdir=path+'face%05d'+filetype!
        m += 1
        os.rename(Olddir, Newdir)
    print m
    print filetype

rename(path = '/home/zf/VIPeR/cam_a/' ,newname = '%6d')  #最后的文件名形如0001的格式