
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


from tran_eval_util import run_training


def parse_args(check=True):
    parser = argparse.ArgumentParser()
    # train
    parser.add_argument('--dataset_dir', type=str,default='')
    parser.add_argument('--logs_train_dir', type=str, default='')
    parser.add_argument('--checkpoint_dir', type=str, default='')    
    parser.add_argument('--checkpoint_exclude_scopes', type=str, default='')
    parser.add_argument('--number_of_classes', type=int,default=764)
    parser.add_argument('--hide_prob', type=float, default=0.25)
    parser.add_argument('--sigma', type=float, default=0.5)
    parser.add_argument('--num_epoc', type=int,default=5)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--num_train_img', type=int, default=43971)
    parser.add_argument('--batch_size', type=int, default=16)

    FLAGS, unparsed = parser.parse_known_args()
    return FLAGS, unparsed

if __name__ == '__main__':
    FLAGS, unparsed = parse_args()
    run_training(num_epoc=FLAGS.num_epoc,\
                 number_of_classes = FLAGS.number_of_classes,\
                 sigma = FLAGS.sigma,\
                 hide_prob = FLAGS.hide_prob,\
                 batch_size=FLAGS.batch_size,\
                 learning_rate=FLAGS.learning_rate,\
                 num_train_img=FLAGS.num_train_img,\
                 dataset_dir=FLAGS.dataset_dir,\
                 logs_train_dir=FLAGS.logs_train_dir,\
                 checkpoint_dir=FLAGS.checkpoint_dir,\
                 checkpoint_exclude_scopes='vgg_16_advers')

    
    
