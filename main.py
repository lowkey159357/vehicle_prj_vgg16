#!/usr/bin/env python3
#coding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os

#dataset_dir = "C:\\jupyter_work\lastweek\\vgg_16_adverse_small_dataset\\image_dataset"
#logs_train_dir = "C:\\jupyter_work\lastweek\\vgg_16_adverse_small_dataset\\out_log"
#checkpoint_dir=os.path.join(logs_train_dir, 'model.ckpt-89560')

def parse_args(check=True):
    parser = argparse.ArgumentParser()
    # train
    parser.add_argument('--dataset_dir', type=str,default=' ')
    parser.add_argument('--logs_train_dir', type=str, default=' ')
    parser.add_argument('--checkpoint_dir', type=str, default=' ') 
    parser.add_argument('--checkpoint_exclude_scopes', type=str, default=' ')
    parser.add_argument('--max_epoc', type=int, default=(65//5))
    parser.add_argument('--num_epoc', type=int, default=5)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--num_train_img', type=int, default=43971)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--number_of_classes', type=int,default=764)
    parser.add_argument('--hide_prob', type=float, default = 0.25)
    parser.add_argument('--sigma', type=float, default = 0.5)
    # eval
    parser.add_argument('--val_loop', type=int,default=256)

    FLAGS, unparsed = parser.parse_known_args()
    return FLAGS, unparsed


train_cmd = 'python ./train.py   --dataset_dir={dataset_dir}  --logs_train_dir={logs_train_dir} \
              --checkpoint_dir={checkpoint_dir}  --checkpoint_exclude_scopes={checkpoint_exclude_scopes}  --num_epoc={num_epoc}  \
              --learning_rate={learning_rate}  --num_train_img={num_train_img} \
              --batch_size={batch_size} --number_of_classes={number_of_classes} --hide_prob={hide_prob} --sigma={sigma} '

eval_cmd = 'python ./eval.py   --dataset_dir={dataset_dir}  --checkpoint_dir={checkpoint_dir}  \
            --number_of_classes={number_of_classes}   --sigma={sigma}  --val_loop={val_loop} --batch_size={batch_size} '

if __name__ == '__main__':
    FLAGS, unparsed = parse_args()
    print('current working dir [{0}]'.format(os.getcwd()))
    w_d = os.path.dirname(os.path.abspath(__file__))
    print('change wording dir to [{0}]'.format(w_d))
    os.chdir(w_d)

    for i in range(FLAGS.max_epoc):
        print('***************************epock {}*********************************'.format(i*5))
        # train 1 epoch
        print('################    train    ################')
        p = os.popen(train_cmd.format(**{'dataset_dir': FLAGS.dataset_dir, 'logs_train_dir': FLAGS.logs_train_dir,
                                         'checkpoint_dir': FLAGS.checkpoint_dir,'checkpoint_exclude_scopes': FLAGS.checkpoint_exclude_scopes,
                                         'num_epoc': FLAGS.num_epoc,'learning_rate': FLAGS.learning_rate, 'num_train_img': FLAGS.num_train_img,
                                         'batch_size': FLAGS.batch_size, 'number_of_classes': FLAGS.number_of_classes,
                                         'hide_prob': FLAGS.hide_prob, 'sigma': FLAGS.sigma}))
        for l in p:
            print(l.strip())

        # eval
        print('################    eval    ################')
        p = os.popen(eval_cmd.format(**{'dataset_dir': FLAGS.dataset_dir, 'checkpoint_dir': FLAGS.logs_train_dir,
                                        'number_of_classes':FLAGS.number_of_classes, 'sigma': FLAGS.sigma,
                                        'val_loop': FLAGS.val_loop, 'batch_size': FLAGS.batch_size}))
        for l in p:
            print(l.strip())
