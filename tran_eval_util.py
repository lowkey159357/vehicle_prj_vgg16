#!/usr/bin/env python3
#coding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import sys
import time
import random
import re
import tensorflow as tf
slim = tf.contrib.slim

import cv2 
from matplotlib import pyplot as plt
import numpy as np

import vgg
from dataset import gen_data_batch

from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_nn_ops
@ops.RegisterGradient("GuidedRelu")
def _GuidedReluGrad(op, grad):
    return tf.where(0. < grad, gen_nn_ops._relu_grad(grad, op.outputs[0]), tf.zeros(grad.get_shape()))


def patch_epock_img(img= None,step=0, epock_step=3, hide_prob=0.3):
    global patch_num          # 静态变量，很关键
    global grid_sizes
    if step%epock_step==0:
        grid_sizes=random.sample([16,25,36,49],1)[0]
        patch_num=random.sample(list(range(0,grid_sizes)),max(int(grid_sizes*hide_prob),1) )
    
    img1=img.copy()
    mean_img=np.mean(img)
    grid_num=int(grid_sizes**0.5)
    gd_w = int(224/grid_num)
    gd_h = int(224/grid_num)
    for i,value in enumerate(patch_num):
        h=value//grid_num
        w=value % grid_num
        img1[:, h*gd_h:(h+1)*gd_h, w*gd_w:(w+1)*gd_w, :]=mean_img
  
    return img1


def inference(X_tensor,number_of_classes,is_training_placeholder):
    with slim.arg_scope(vgg.vgg_arg_scope()):
        logits, end_points = vgg.vgg_16(inputs = X_tensor,
                                        num_classes = number_of_classes,
                                        is_training = is_training_placeholder,
                                        fc_conv_padding = 'VALID')
    return logits, end_points


def forward_tran(x_img=None,label_index=None, number_of_classes=5,layer_name='vgg_16/pool5',Training=True):
    # 根据logits，label_index，计算目标函数对conv层maps的梯度
    logits, end_points = inference(x_img, number_of_classes, Training)
    prob=tf.nn.softmax(logits)
    if Training==True:
        prob_max_label = label_index
    else:
        prob_max_label = tf.argmax(prob,1)
    label_hot = tf.one_hot(prob_max_label, number_of_classes)
    cost = (-1) * tf.reduce_sum(tf.multiply( tf.log(prob), label_hot ), axis=1)
    y_c = tf.reduce_sum(tf.multiply( logits, label_hot), axis=1)  # Grad CAM 
    target_conv_layer=end_points[layer_name]
    #gb_grad = tf.gradients(cost, x_img)[0]                                  # guided Grad-CAM
    target_grad_ac = tf.gradients(y_c, target_conv_layer)[0]                # Grad CAM 
    target_grad_yc= tf.gradients( tf.exp(y_c), target_conv_layer)[0]        #Grad CAM ++
    return target_conv_layer,target_grad_ac,target_grad_yc,logits,end_points   

def forward_tran_advers(x_img=None, label_index=None, number_of_classes=5,layer_name=None,Training=True):
    # 根据logits，label_index，计算目标函数对conv层maps的梯度
    with slim.arg_scope(vgg.vgg_arg_scope()):
        logits, end_points = vgg.vgg_16_adversarial(inputs = x_img,\
                                        num_classes = number_of_classes,\
                                        is_training = Training,\
                                        fc_conv_padding = 'VALID')
    
    prob=tf.nn.softmax(logits)
    if Training==True:
        prob_max_label = label_index
    else:
        prob_max_label = tf.argmax(prob,1)
    label_hot = tf.one_hot(prob_max_label, number_of_classes)
    cost = (-1) * tf.reduce_sum(tf.multiply( tf.log(prob), label_hot ), axis=1)
    y_c = tf.reduce_sum(tf.multiply( logits, label_hot), axis=1)  # Grad CAM 
    #target_conv_layer=end_points[layer_name]
    target_conv_layer=x_img
    #gb_grad = tf.gradients(cost, x_img)[0]                                  # guided Grad-CAM
    target_grad_ac = tf.gradients(y_c, target_conv_layer)[0]                 # Grad CAM 
    target_grad_yc= tf.gradients( tf.exp(y_c), target_conv_layer)[0]         # Grad CAM ++
    return target_conv_layer,target_grad_ac,target_grad_yc,logits,end_points   


def cal_Grad_CAM_pp(target_conv_layer, target_grad_ac, target_grad_yc):
    
    conv_sum=tf.reduce_sum(target_conv_layer,axis=[0, 1])
    fenmu=tf.multiply(conv_sum,target_grad_ac)
    alpha_kc= 1.0/(2.0+fenmu)
    target_grad_yc_relu=tf.nn.relu(target_grad_yc)
    w_kc=tf.reduce_sum(alpha_kc * target_grad_yc_relu,axis=[0, 1])
    
    Grad_CAM_pp_gray =tf.reduce_sum(w_kc * target_conv_layer,axis=2)
    Grad_CAM_pp_gray=tf.nn.relu(Grad_CAM_pp_gray)
    Grad_CAM_pp_gray=Grad_CAM_pp_gray-tf.reduce_min(Grad_CAM_pp_gray)
    Grad_CAM_pp_gray = Grad_CAM_pp_gray / tf.reduce_max(Grad_CAM_pp_gray)       # scale 0 to 1.0
    
    one_img = tf.expand_dims(Grad_CAM_pp_gray, 0)
    one_img = tf.expand_dims(one_img, -1) #-1表示最后一维
    Grad_CAM_pp_gray_224= tf.image.resize_bilinear(one_img, [224, 224])
    Grad_CAM_pp_gray_224= tf.reshape(Grad_CAM_pp_gray_224,[224, 224])
    Grad_CAM_pp_gray_224 =  255*Grad_CAM_pp_gray_224
    
    return Grad_CAM_pp_gray, Grad_CAM_pp_gray_224
      
def mask_image(CAM_gray=None, img_input=None, sigma=0.7):
    
    gcam=CAM_gray
    gcam=gcam-tf.reduce_min(gcam)
    gcam = gcam / tf.reduce_max(gcam)       # scale 0 to 1.0
    mask=tf.cast(gcam<sigma,tf.float32)
    #mask_img = tf.stack([img_input[:,:,0]*mask,img_input[:,:,1]*mask,img_input[:,:,2]*mask], 2) 
    mask_img_1 = tf.expand_dims(mask, -1)
    mask_img = img_input*mask_img_1

    constan= 0.0*mask_img + tf.reduce_mean(img_input)
    mask_img=tf.where(tf.equal(mask_img,0),constan, mask_img)
    
    return mask_img

def batch_mask_images(batch_size=1,img_input=None,\
                      img_conv=None,target_conv_layer=None,\
                      target_grad_ac=None,target_grad_yc=None,\
                      sigma=0.5 ):
    # 对卷积层mask
    if img_conv!=None:
        for i in  range(batch_size):                 
            Grad_CAM_pp_gray, _=cal_Grad_CAM_pp(target_conv_layer[i], target_grad_ac[i], target_grad_yc[i])
            mask_img_1 = mask_image(CAM_gray=Grad_CAM_pp_gray, img_input=img_conv[i],sigma=sigma)
            #mask_img_1 = tf.reshape(mask_img_1,[1,224,224,3])
            mask_img_1 = tf.expand_dims(mask_img_1, 0)
            if i==0:
                mask_conv_imgs=mask_img_1
            else:
                mask_conv_imgs=tf.concat([mask_conv_imgs,mask_img_1],axis=0)
    else:
        mask_conv_imgs=0
    # 对神经网络输入层mask    
    if img_input!=None:
        for i in  range(batch_size):                 
            _, Grad_CAM_pp_gray_224=cal_Grad_CAM_pp(target_conv_layer[i], target_grad_ac[i], target_grad_yc[i])
            mask_img_1 = mask_image(CAM_gray=Grad_CAM_pp_gray_224, img_input=img_input[i],sigma=0.5)
            #mask_img_1 = tf.reshape(mask_img_1,[1,224,224,3])
            mask_img_1 = tf.expand_dims(mask_img_1, 0)
            if i==0:
                mask_input_imgs=mask_img_1
            else:
                mask_input_imgs=tf.concat([mask_input_imgs,mask_img_1],axis=0)            
    else:
        mask_input_imgs=0            
            
    return mask_conv_imgs,mask_input_imgs 


def losses(logits_cl,logits_adver,logits_amin,labels_hot,label_advers_hot,number_of_classes):
    with tf.variable_scope("loss") as scope:
        # logit_cl_loss
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits_cl,labels=labels_hot)
        logit_cl_loss = tf.reduce_mean(cross_entropy, name="logit_cl_loss")
        # logit_adver_loss
        cross_entropy2 = tf.nn.softmax_cross_entropy_with_logits(logits=logits_adver,labels=label_advers_hot)
        logit_adver_loss = tf.reduce_mean(cross_entropy2, name="logit_adver_loss")             
        # logit_amin_loss
        logits_amin=tf.nn.sigmoid(logits_amin)
        logit_amin_loss = 1*tf.reduce_mean(tf.multiply(logits_amin, labels_hot),name="logit_amin_loss")
        #logit_amin_loss = tf.reduce_sum(tf.multiply(logits_amin, labels_hot))    
        # regularization_loss
        regularization_loss = 1*tf.add_n(slim.losses.get_regularization_losses())  
        # total_loss
        total_loss = logit_cl_loss + logit_adver_loss + logit_amin_loss + regularization_loss
        
        tf.summary.scalar(scope.name + "/logit_cl_loss", logit_cl_loss)
        tf.summary.scalar(scope.name + "/logit_adver_loss", logit_adver_loss)
        tf.summary.scalar(scope.name + "/logit_amin_loss", logit_amin_loss)
        tf.summary.scalar(scope.name + "/regularization_loss", regularization_loss)
        tf.summary.scalar(scope.name + "/total_loss", total_loss)     
    return total_loss,logit_cl_loss,logit_adver_loss,logit_amin_loss,regularization_loss  

def trainning(loss, learning_rate):
    with tf.name_scope("optimizer"):
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        global_step = tf.Variable(0, name="global_step", trainable=False, dtype=tf.int64)
        train_op = optimizer.minimize(loss, global_step=global_step)
    return train_op, global_step 

def evaluation(logits, labels_hot):
    with tf.variable_scope("accuracy") as scope:
        correct_prediction = tf.equal(tf.argmax(logits,1), tf.argmax(labels_hot,1))  
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))    
        #correct = tf.nn.in_top_k(logits, labels, 1)
        #correct = tf.cast(correct, tf.float16)
        #accuracy = tf.reduce_mean(correct)
        tf.summary.scalar(scope.name + "accuracy", accuracy)
    return accuracy
    
    
def run_training(number_of_classes = 5,\
                 batch_size=10,\
                 learning_rate=0.00001,\
                 num_train_img=90,\
                 num_epoc=200,\
                 hide_prob=0.25,\
                 sigma = 0.6,\
                 dataset_dir=None,\
                 logs_train_dir=None,\
                 checkpoint_dir=None,\
                 checkpoint_exclude_scopes=None):

    if not os.path.exists(logs_train_dir):
        os.makedirs(logs_train_dir)    

    #定义输入变量
    tf.reset_default_graph()
    with tf.Graph().as_default() as g:
    
        # 生成训练数据
        img_train,label_train=gen_data_batch(dataset_dir=dataset_dir, batch_size=batch_size, Train=True)
        
        # 推理、训练、准确度
        #logits, end_points = inference(x_img, number_of_classes,True)
        x_img_in = tf.placeholder(tf.float32,[batch_size, 224, 224, 3]) 
        x_label = tf.placeholder(tf.int64,[batch_size]) 
        label_hot = tf.one_hot(x_label, number_of_classes)
        
        label_advers=number_of_classes-1-x_label
        label_advers_hot = tf.one_hot(label_advers, number_of_classes)
        layer_name1='vgg_16/pool5'
        layer_name2='vgg_16_advers/erash_pool'
        target_conv_layer,target_grad_ac,target_grad_yc,logits_cl,end_points = forward_tran(x_img=x_img_in,\
                                                                                            label_index=x_label,\
                                                                                            number_of_classes=number_of_classes,\
                                                                                            layer_name=layer_name1,\
                                                                                            Training=True)   
        conv_in=end_points[layer_name1]
        mask_conv_imgs,mask_input_imgs=batch_mask_images(batch_size=batch_size,\
                                                         img_input = x_img_in,\
                                                         img_conv = conv_in,\
                                                         target_conv_layer=target_conv_layer,\
                                                         target_grad_ac=target_grad_ac,\
                                                         target_grad_yc=target_grad_yc,\
                                                         sigma=sigma)
        
        _,_,_,logits_adver,_ = forward_tran_advers(x_img = mask_conv_imgs,\
                                                   label_index = label_advers,\
                                                   number_of_classes = number_of_classes,\
                                                   Training = True)
        
        _,_,_,logits_cl_amin,_ = forward_tran(x_img=mask_input_imgs,\
                                              label_index=x_label,\
                                              number_of_classes=number_of_classes,\
                                              layer_name=layer_name1,\
                                              Training=True)   
        
        total_loss,logit_cl_loss,logit_adver_loss,logit_amin_loss,L2_loss = losses(logits_cl,\
                                                                   logits_adver,\
                                                                   logits_cl_amin,\
                                                                   label_hot,\
                                                                   label_advers_hot,\
                                                                   number_of_classes)
        
        train_op, global_step = trainning(total_loss,learning_rate)
        train_accuracy = (evaluation(logits_cl,label_hot) + evaluation(logits_adver,label_advers_hot))/2.0
        # 
        config = tf.ConfigProto()                                    # 配置GPU参数 
        config.gpu_options.allow_growth=True                         # 动态分配GPU资源   
        #config.gpu_options.per_process_gpu_memory_fraction = 0.85   # 占用GPU90%的显存 
        sess = tf.Session(config=config)
        #  
        summary_op = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(logs_train_dir,sess.graph)
        start_time = time.time()
        with sess: #开始一个会话
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            coord=tf.train.Coordinator()
            threads= tf.train.start_queue_runners(coord=coord) 
            
            saver = tf.train.Saver(tf.all_variables())
            exclusions = []
            if checkpoint_exclude_scopes:
                exclusions = [scope.strip() for scope in checkpoint_exclude_scopes.split(',')]
            
            variables_to_restore = []
            for var in slim.get_model_variables():
                excluded = False
                for exclusion in exclusions:
                    if var.op.name.startswith(exclusion):
                        excluded = True
                        break
                if not excluded:
                    variables_to_restore.append(var)    
            
            ckpt = tf.train.get_checkpoint_state(logs_train_dir)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                print('checkpoint restored from [{0}]'.format(ckpt.model_checkpoint_path))
                print('model_checkpoint_path Loading success')
            elif checkpoint_dir:
                read_weights_restore = slim.assign_from_checkpoint_fn(checkpoint_dir, variables_to_restore)
                read_weights_restore(sess)
                print('checkpoint restored from [{0}]'.format(checkpoint_dir))
            else: 
                print('No checkpoint before training')
                pass
            
            one_epock_step=num_train_img//batch_size
            MAX_STEP=num_epoc*one_epock_step
            for step in range(num_epoc*num_train_img//batch_size):
            #for step in range(1):    
            
                _img_train,_label_train =sess.run([img_train,label_train])
                patch_img = patch_epock_img(img= _img_train,step=step,epock_step=one_epock_step,\
                                            hide_prob=hide_prob)
                                
                feed_vars={x_img_in:_img_train,x_label:_label_train }
                _global_step,_, _total_loss,_logit_cl_loss,_logit_adver_loss,_logit_amin_loss,\
                _L2_loss,_accuracy,summary_str = sess.run([global_step,train_op,total_loss,\
                                                           logit_cl_loss,logit_adver_loss,\
                                                           logit_amin_loss,L2_loss,\
                                                           train_accuracy,summary_op],\
                                                          feed_dict=feed_vars)
                #每迭代50次，打印出一次结果
                if step %  20 == 0:
                    print('epoc %d, Step %d, total_loss = %.5f, cl_loss==%.5f, adver_loss=%.5f, amin_loss=%.5f, L2_loss=%.5f, accuracy = %.3f，（%.3f sec/step）'\
                                      %((step*batch_size)//num_train_img, _global_step, _total_loss,_logit_cl_loss,_logit_adver_loss,\
                                        _logit_amin_loss,_L2_loss,_accuracy,(time.time()-start_time)/(step+1e-5) )) 
                    train_writer.add_summary(summary_str,_global_step)
                #每迭代200次，利用saver.save()保存一次模型文件，以便测试的时候使用
                if ((step % 450 ==0) and step>0) or (step +1) == MAX_STEP:
                    checkpoint_path = os.path.join(logs_train_dir,'model.ckpt')
                    saver.save(sess,checkpoint_path, global_step = _global_step)

            coord.request_stop()
            coord.join(threads)
        sess.close()
        
    print('time use is %d second'%(time.time()-start_time)) 

	
	
def run_eval(val_loop=2,number_of_classes = 5,batch_size=40,sigma=0.6,checkpoint_dir=None,dataset_dir=None):
   
    #定义输入变量
    tf.reset_default_graph()
    with tf.Graph().as_default() as g:
    
        # 生成训练数据
        img_val,label_val=gen_data_batch(dataset_dir=dataset_dir, batch_size=batch_size, Train=False)
        
        # 推理、训练、准确度
        x_img_in = tf.placeholder(tf.float32,[batch_size, 224, 224, 3]) 
        x_label = tf.placeholder(tf.int64,[batch_size]) 
        label_hot = tf.one_hot(x_label, number_of_classes)
        
        label_advers=number_of_classes-1-x_label
        label_advers_hot = tf.one_hot(label_advers, number_of_classes)
        layer_name1='vgg_16/pool5'
        layer_name2='vgg_16_advers/erash_pool'
        target_conv_layer,target_grad_ac,target_grad_yc,logits_cl,end_points = forward_tran(x_img=x_img_in,
                                                                                   number_of_classes=number_of_classes,
                                                                                   layer_name=layer_name1,
                                                                                   Training=False)   
        conv_in=end_points[layer_name1]
        mask_conv_imgs,_=batch_mask_images(batch_size=batch_size,img_conv = conv_in,target_conv_layer=target_conv_layer,\
                                    target_grad_ac=target_grad_ac,target_grad_yc=target_grad_yc,sigma=sigma)
        
        _,_,_,logits_adver,_ = forward_tran_advers(x_img=mask_conv_imgs,number_of_classes=number_of_classes,\
                                                    layer_name=layer_name2,Training=False)
        
        accuracy = evaluation(logits_cl,label_hot)
        accuracy_adver=evaluation(logits_adver,label_advers_hot)
        prob_max = tf.argmax(tf.nn.softmax(logits_cl),1)  
        prob_adver_max=tf.argmax(tf.nn.softmax(logits_adver),1)
        # 
        config = tf.ConfigProto()                                    # 配置GPU参数 
        config.gpu_options.allow_growth=True                         # 动态分配GPU资源   
        #config.gpu_options.per_process_gpu_memory_fraction = 0.85   # 占用GPU90%的显存 
        sess = tf.Session(config=config)
        #  
        start_time = time.time()
        with sess: #开始一个会话
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            coord=tf.train.Coordinator()
            threads= tf.train.start_queue_runners(coord=coord) 
            
            saver = tf.train.Saver(tf.all_variables()) 
            ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                print('model_checkpoint_path Loading success')
            else:
                print('No checkpoint before training')
                pass
            
            accuracy_list=[]
            for i in range(val_loop):
                _img_val,_label_val =sess.run([img_val,label_val])
                feed_vars={x_img_in:_img_val, x_label:_label_val }
                _, _accuracy,_accuracy_adver, _prob_max,_prob_adver_max,_label_val = sess.run([logits_cl,\
                                                                                               accuracy,\
                                                                                               accuracy_adver,\
                                                                                               prob_max,\
                                                                                               prob_adver_max,\
                                                                                               label_val],\
                                                                                              feed_dict=feed_vars)
                accuracy_list.append(_accuracy)
                accuracy_list.append(_accuracy_adver)
                print( 'setp:%d, test, (accuracy_A, accuracy_B) = (%.3f, %.3f)'%( i, _accuracy,_accuracy_adver) )
                #print('step:{}, test_real_label is:{}'.format(i,_label_val) )
                #print('step:{}, test_prob_label is:{}'.format(i, _prob_max) )
                
            coord.request_stop()
            coord.join(threads)
        sess.close()
        
        aver_accuracy=sum(accuracy_list)/len(accuracy_list)
        print('At last, test, aver_accuracy:%.4f'%(aver_accuracy) )
        print('time use is %d second'%(time.time()-start_time)) 
        return aver_accuracy	
