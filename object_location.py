#!/usr/bin/env python3
#coding: utf-8

#%matplotlib inline

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import sys
import time
import tensorflow as tf
import argparse

from object_util import run_test


label_text_dir = "C:\\jupyter_work\lastweek\\vgg_16_adverse_small_dataset"
checkpoint_dir = "C:\\jupyter_work\lastweek\\vgg_16_adverse_small_dataset\\out_log"
test_img_dir='C:\\jupyter_work\\lastweek\\vgg_16_adverse_small_dataset\\test_img_files'


def parse_args(check=True):
    parser = argparse.ArgumentParser()
    # train
    parser.add_argument('--label_text_dir', type=str,default=label_text_dir)
    parser.add_argument('--test_img_dir', type=str, default=test_img_dir)
    parser.add_argument('--checkpoint_dir', type=str, default=checkpoint_dir)   
    parser.add_argument('--loop_max', type=int,default=1)
    parser.add_argument('--number_of_classes', type=int,default=5)
    parser.add_argument('--alpha', type=float, default=0.3)
    parser.add_argument('--sigma', type=float, default=0.6)
    parser.add_argument('--area_th', type=float, default=0.05)
    parser.add_argument('--prob_th', type=float, default=0.5)
    parser.add_argument('--pred_th', type=float, default=0.3)    
    parser.add_argument('--plt_on', type=bool, default=True)

    FLAGS, unparsed = parser.parse_known_args()
    return FLAGS, unparsed

if __name__ == '__main__':
    FLAGS, unparsed = parse_args()
    img_with_bbox_batch,img_crf_bbox_batch=run_test(loop_max = FLAGS.loop_max,\
                 number_of_classes = FLAGS.number_of_classes,\
                 sigma = FLAGS.sigma,\
                 alpha = FLAGS.alpha,\
                 area_th = FLAGS.area_th,\
                 prob_th = FLAGS.prob_th,\
                 pred_th = FLAGS.pred_th,\
                 plt_on = FLAGS.plt_on,\
                 label_text_dir = FLAGS.label_text_dir,\
                 checkpoint_dir = FLAGS.checkpoint_dir,\
                 test_img_dir = FLAGS.test_img_dir)


