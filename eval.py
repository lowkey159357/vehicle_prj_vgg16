#!/usr/bin/env python3
#coding: utf-8


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import sys
import time
import tensorflow as tf
import argparse

from tran_eval_util import run_eval


def parse_args(check=True):
    parser = argparse.ArgumentParser()
    # train
    parser.add_argument('--dataset_dir', type=str,default=' ')
    parser.add_argument('--checkpoint_dir', type=str, default=' ')    
    parser.add_argument('--number_of_classes', type=int,default= 764)
    parser.add_argument('--sigma', type=float, default= 0.6)
    parser.add_argument('--val_loop', type=int,default= 256)
    parser.add_argument('--batch_size', type=int, default= 16)

    FLAGS, unparsed = parser.parse_known_args()
    return FLAGS, unparsed

if __name__ == '__main__':
    FLAGS, unparsed = parse_args()
    run_eval(val_loop = FLAGS.val_loop,\
             number_of_classes = FLAGS.number_of_classes,\
             batch_size = FLAGS.batch_size,\
             sigma = FLAGS.sigma,\
             checkpoint_dir = FLAGS.checkpoint_dir,\
             dataset_dir = FLAGS.dataset_dir)
    