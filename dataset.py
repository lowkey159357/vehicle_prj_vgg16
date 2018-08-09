#!/usr/bin/env python3
#coding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import sys
import time

import tensorflow as tf
slim = tf.contrib.slim

import numpy as np
import random
import re

# 图片信息
NUM_CLASSES = 764

#根据文件名生成一个队列
_R_MEAN = 123.68
_G_MEAN = 116.78
_B_MEAN = 103.94

def _mean_image_subtraction(image, means):
    if image.get_shape().ndims != 3:
        raise ValueError('Input must be of size [height, width, C>0]')
    num_channels = image.get_shape().as_list()[-1]
    if len(means) != num_channels:
        raise ValueError('len(means) must match the number of channels')

    channels = tf.split(axis=2, num_or_size_splits=num_channels, value=image)
    for i in range(num_channels):
        channels[i] -= means[i]
    return tf.concat(axis=2, values=channels)


def read_and_decode(filename,batch_size=2, shuffle=1, img_H=224, img_W=224):
    
    filename_queue = tf.train.string_input_producer(filename)  # filename要是列表形式
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)   #返回文件名和文件
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'image/encoded': tf.FixedLenFeature([], tf.string),
                                           'image/format': tf.FixedLenFeature([], tf.string),
                                           'image/class/label' : tf.FixedLenFeature([], tf.int64),
                                           'image/height': tf.FixedLenFeature([], tf.int64),
                                           'image/width': tf.FixedLenFeature([], tf.int64),
                                       })

    img_format = tf.cast(features['image/format'], tf.string)
    img_height = tf.cast(features['image/height'], tf.int64)
    img_width = tf.cast(features['image/width'], tf.int64)
    img_label = tf.cast(features['image/class/label'], tf.int64)
    
    #image = tf.decode_raw(features['image/encoded'], tf.uint8)  # 结果是一维变量
    #image = tf.reshape(image, [img_height, img_width, 3] )
    image = tf.image.decode_jpeg(features['image/encoded'], channels=3)  # 结果是三维变量
    #image = tf.image.resize_images(image,[img_H, img_W])   # 图片尺度必须统一,for则batch会报错
    image_float = tf.to_float(image, name='ToFloat')
    mean_centered_image = _mean_image_subtraction(image_float, [_R_MEAN, _G_MEAN, _B_MEAN])  # 随机截取，数据增广
    image = tf.image.resize_image_with_crop_or_pad(mean_centered_image,img_H,img_W) 
#    image = tf.image.per_image_standardization(image)   #resize
    
    if shuffle==1:
        img_batch, label_batch, height_batch, width_batch,format_batch = tf.train.shuffle_batch([image,img_label,img_height,img_width, img_format],
                                                        batch_size= batch_size,
                                                        num_threads=4,
                                                        capacity=50*batch_size,
                                                        min_after_dequeue=5*batch_size)
    else:
        img_batch, label_batch, height_batch, width_batch,format_batch = tf.train.batch([image,img_label,img_height,img_width, img_format],
                                                        batch_size= batch_size,
                                                        num_threads=4,
                                                        capacity=50*batch_size)    
    return img_batch, label_batch


def gen_data_batch(dataset_dir=None, batch_size=2, Train=True):
    assert os.path.exists(dataset_dir), '[{0}] not exist!!!'.format(dataset_dir)
    
    if  Train==True:
        filenames=os.listdir(dataset_dir)
        TFrecored_name=[]
        for filename in filenames:
            if re.search('train',filename):
                TFrecored_name.append(os.path.join(dataset_dir, filename))
        img_train_batch, label_train_batch = read_and_decode(TFrecored_name, batch_size)
        return img_train_batch,label_train_batch
    
    else:
        filenames=os.listdir(dataset_dir)
        TFrecored_name=[]
        for filename in filenames:
            if re.search('validation',filename):
                TFrecored_name.append(os.path.join(dataset_dir, filename))
        img_val_batch, label_val_batch = read_and_decode(TFrecored_name, batch_size)        
        return img_val_batch,label_val_batch   

