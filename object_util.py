#!/usr/bin/env python3
# -*- coding: utf-8 -*-


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
import skimage
import skimage.io
import skimage.transform
from skimage.transform import resize
from PIL import Image, ImageDraw, ImageFont

from matplotlib import pyplot as plt
import numpy as np


import sys
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import compute_unary, create_pairwise_bilateral, \
    create_pairwise_gaussian, softmax_to_unary,unary_from_softmax

import vgg
from tran_eval_util import forward_tran,batch_mask_images,forward_tran_advers,evaluation


_R_MEAN = 123.68
_G_MEAN = 116.78
_B_MEAN = 103.94

def get_batch_image(img_dir=None,label_on=True,img_num_max=50):
    filenames=os.listdir(img_dir)
    img_path_list=[]
    for filename in filenames:
        if re.search('.jpg',filename) or re.search('.png',filename):
            img_path_list.append(os.path.join(img_dir, filename))

    batch_size=len(img_path_list)
    label_batch=[]
    size_batch=[]
    
    for i, value in enumerate(img_path_list):
        img0 = cv2.imread(value)
        # 图片大小
        height,width=img0.shape[0],img0.shape[1]
        size_batch.append([height,width])
        # label
        if label_on==True:
            label_index = os.path.basename(value).split('_')[-1].split('.')[-2]   # 4_validation_5.jpg  ,5是标签编号
            label_index = int(label_index) 
            label_batch.append(label_index)
        else:
            label_list.append(0)
        # 图片堆叠
        img0 = cv2.resize(img0,(224,224))
        img0 = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB )            # 三通道彩色图像
        img0=np.reshape(img0,[1,224,224,3])
        if i==0:
            img_batch=img0
        else:
            img_batch=np.concatenate((img_batch,img0),axis=0) 
        # 最多读取50张
        if i>img_num_max:
            break
    return img_batch,label_batch,size_batch,batch_size

def convet_to_ori(img):
    r, g, b = tf.split(img, 3, 3)
    ori_images = tf.concat([ r +_R_MEAN , g +_G_MEAN ,b +_B_MEAN], 3)
    return ori_images

def read_label_file(dataset_dir=None, filename='label.txt'):
    labels_filename = os.path.join(dataset_dir, filename)
    with tf.gfile.Open(labels_filename, 'rb') as f:
        lines = f.read().decode()
    lines = lines.split('\n')
    lines = filter(None, lines)

    labels_to_class_names = {}
    for line in lines:
        index = line.index(':')
        labels_to_class_names[int(line[:index])] = line[index+1:]
    return labels_to_class_names

def np_Grad_CAM_pp(target_conv_layer, target_grad_ac, target_grad_yc, ori_images,plt_on=False):
    
    conv_sum=np.sum(target_conv_layer,axis=(0, 1))
    fenmu = conv_sum * target_grad_ac
    alpha_kc= 1.0/(2.0+fenmu)
    target_grad_yc_relu = np.maximum(target_grad_yc, 0)                  # Relu
    w_kc=np.sum(alpha_kc * target_grad_yc_relu, axis=(0, 1) )            # 乘加

    Grad_CAM_pp_gray = np.sum(w_kc * target_conv_layer, axis=2)          # 梯度和卷积层各通道图像相乘
    Grad_CAM_pp_gray = np.maximum(Grad_CAM_pp_gray, 0)                   # Relu
    Grad_CAM_pp_gray = Grad_CAM_pp_gray / np.max(Grad_CAM_pp_gray)       # scale 0 to 1.0
    #Grad_CAM_pp_gray = resize(Grad_CAM_pp_gray, (224,224), preserve_range=True)   # 单通道灰度图像
    Grad_CAM_pp_gray = cv2.resize(Grad_CAM_pp_gray, (224,224))
    Grad_CAM_pp_gray = np.uint8( 255*Grad_CAM_pp_gray )
    Grad_CAM_pp = cv2.applyColorMap( Grad_CAM_pp_gray, cv2.COLORMAP_JET )   # 三通道彩色图像
    Grad_CAM_pp = cv2.cvtColor( Grad_CAM_pp, cv2.COLOR_BGR2RGB )            # 三通道彩色图像

    img = ori_images.astype(float)
    img -= np.min(img)
    img /= img.max()    
    alpha = 0.0025
    new_img = img+alpha*Grad_CAM_pp
    new_img /= new_img.max()
    new_img = np.uint8(255*new_img)
    
    if plt_on==True:
        # 可视化分析
        f, (ax1,ax2,ax3) = plt.subplots(1, 3, sharey=True,figsize=(14,3))
        ax1.imshow(img)  # BGR,格式
        ax1.set_title('Input Image')
        ax2.imshow(Grad_CAM_pp)  # BGR,格式
        ax2.set_title('Grad_CAM + +')
        ax3.imshow(new_img)  # BGR,格式
        ax3.set_title('Input image + (Grad_CAM++) ')
    
    return Grad_CAM_pp_gray,Grad_CAM_pp,new_img
      
def label_into_img(img, string, position=(112,112), type_color=(0, 0, 255), type_big=15):
    # 图像从OpenCV格式转换成PIL格式 
    img -= np.min(img)
    img=np.uint8(255*(img/img.max()))             # 归一化，量化
    cv2img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # cv2和PIL中颜色的hex码的储存顺序不同
    pilimg = Image.fromarray(cv2img)
    # PIL图片上打印汉字
    draw = ImageDraw.Draw(pilimg) # 图片上打印
    font = ImageFont.truetype("simhei.ttf", type_big, encoding="utf-8") # 参数1：字体文件路径，参数2：字体大小
    #font = ImageFont.truetype("simsun.ttc", type_big, encoding="utf-8") # 参数1：字体文件路径，参数2：字体大小
    draw.text(position, string, type_color, font=font) # 参数1：打印坐标，参数2：文本，参数3：字体颜色，参数4：字体
    # PIL图片转cv2 图片
    cv2charimg = cv2.cvtColor(np.array(pilimg), cv2.COLOR_RGB2BGR)
    # cv2.imshow("图片", cv2charimg) # 汉字窗口标题显示乱码
    return cv2charimg

def cal_heatmap_threshold(alpha=0.25,gray_img=None):
    h,w=gray_img.shape
    heatmap_threshold = np.sort(np.reshape(gray_img,(1,-1)))[0][int(h*w*(1-alpha))]    # 0.75百分位数
    return heatmap_threshold

def plot_box(img_gray=None, img_ori=None, label_text='Jeep-指南者', alpha=0.7, area_th=0.03,plt_on=False):
    # 画出属于某个单一类下的所有 bbox
    img_gray -= np.min(img_gray)
    img_gray = np.uint8(255*(img_gray/img_gray.max()))   # 归一化，量化
    img_ori -= np.min(img_ori)
    img_ori = np.uint8(255*(img_ori/img_ori.max()))      # 归一化，量化
    
    #heatmap_th = cal_heatmap_threshold(alpha = alpha, gray_img=img_gray)                   # 根据阈值计算热图的分割门限
    heatmap_th=256*alpha
    ret,img_gray_th = cv2.threshold(img_gray, heatmap_th, 255, cv2.THRESH_TOZERO)        # 大于门限的保留，小于门限的置0
    #thresh11111=img_gray_th.copy()   # CRF 函数操作
    image11, contours, hier = cv2.findContours(img_gray_th, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    try:
        num_bbox=hier.shape[1]
    except:
        return img_gray_th, 0
    print('bbox个数:',num_bbox)
    img_cp=(img_ori).copy()             # 不复制操作的话，cv2.rectangle将修改img_ori
    num=0
    for i in range(num_bbox):
        cnt = contours[i]
        x,y,w,h = cv2.boundingRect(cnt)
        if ((w*h)/(224.0*224.0))>area_th:
            num+=1
            print('arex of bbox %.4f'%((w*h)/(224.0*224.0)) )
            cv2.rectangle(img_cp,(x,y),(x+w,y+h),(255,255,255),2)   # cv2.rectangle函数对image操作是指针操作，具有历史记忆特性
            p_x = x
            p_y = (y+10) if (y-20)<=0  else  (y-20) 
            img_cp = label_into_img(img=img_cp, string=label_text, 
                                          position=(p_x, p_y), type_color=(255, 255, 255), type_big=15)
        else:
            continue
            
    img_with_bbox=img_cp 
    
    if  num==0:
        return img_gray_th, 0
            
    if plt_on==True :
        # 可视化分析
        plt.figure(figsize=(14,3))
        plt.subplot(1,3,1),plt.imshow(img_gray),plt.title('Grad CAM + +')
        plt.subplot(1,3,2),plt.imshow(img_gray_th),plt.title('After threshold ')
        plt.subplot(1,3,3),plt.imshow(img_with_bbox),plt.title('Input image + bbox')
            
    return img_gray_th, img_with_bbox

def perform_crf(image=None, probabilities=None, number_class=2):

    image = image.squeeze()
    softmax = probabilities.squeeze().transpose((2, 0, 1))
    # The input should be the negative of the logarithm of probability values
    # Look up the definition of the softmax_to_unary for more information
    unary = unary_from_softmax(softmax)
    # The inputs should be C-continious -- we are using Cython wrapper
    unary = np.ascontiguousarray(unary)
    d = dcrf.DenseCRF(image.shape[0] * image.shape[1], number_class)
    d.setUnaryEnergy(unary)
    # This potential penalizes small pieces of segmentation that are
    # spatially isolated -- enforces more spatially consistent segmentations
    feats = create_pairwise_gaussian(sdims=(10, 10), shape=image.shape[:2])
    d.addPairwiseEnergy(feats, compat=3,
                        kernel=dcrf.DIAG_KERNEL,
                        normalization=dcrf.NORMALIZE_SYMMETRIC)
    # This creates the color-dependent features --
    # because the segmentation that we get from CNN are too coarse
    # and we can use local color features to refine them
    feats = create_pairwise_bilateral(sdims=(50, 50), schan=(20, 20, 20),
                                      img=image, chdim=2)
    d.addPairwiseEnergy(feats, compat=10,
                        kernel=dcrf.DIAG_KERNEL,
                        normalization=dcrf.NORMALIZE_SYMMETRIC)
    Q = d.inference(5)
    res = np.argmax(Q, axis=0).reshape((image.shape[0], image.shape[1]))
    return res

def plot_box_CRF(img_gray=None, img_ori=None, label_text='Jeep-指南者', alpha=0.7, area_th=0.03,plt_on=False):
    # 本函数能画出属于某个单一类下的所有 bbox
    
    img_gray -= np.min(img_gray)
    img_gray= np.uint8(255*(img_gray/img_gray.max()))   # 归一化，量化
    img_ori -= np.min(img_ori)
    img_ori= np.uint8(255*(img_ori/img_ori.max()))      # 归一化，量化
    
    #heatmap_th=cal_heatmap_threshold(alpha = alpha,gray_img=img_gray)                      # 根据阈值计算热图的分割门限
    heatmap_th =int(256*alpha)
    ret,img_gray_th = cv2.threshold(img_gray, heatmap_th, 255, cv2.THRESH_TOZERO)          # 大于门限的保留，小于门限的置0
    
    image11, contours, hier = cv2.findContours(img_gray_th, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    try:
        num_bbox=hier.shape[1]
    except:
        return 0, 0, 0
    
    max_temp=[]
    num=0
    for i in range(num_bbox):
        cnt = contours[i]
        x,y,w,h = cv2.boundingRect(cnt)
        if ((w*h)/(224.0*224.0))>area_th:
            num+=1
            max_temp.append([x,y,w,h])
     
    if num==0:
        return 0, 0, 0
    
    img_rect = np.zeros((224,224))
    for i in range(len(max_temp)):
        x=max_temp[i][0]
        y=max_temp[i][1]
        w=max_temp[i][2]
        h=max_temp[i][3]
        img_rect[y:y+h, x:x+w] =img_rect[y:y+h, x:x+w]+img_gray[y:y+h, x:x+w]
      
    #img_gray_th = 1*img_gray_th+(img_gray-img_gray_th)*0.05                         # 门限以外衰减
    img_gray_th = img_rect*0.6 + img_gray_th*0.2 +(img_gray-img_rect)*0.2            # 门限以外衰减
    #img_gray_th=img_gray_th
    #plt.figure(figsize=(18,4))
    #plt.imshow(img_gray_th),plt.title('img_temp')
    # CRF 函数操作
    img_gray_th -= np.min(img_gray_th)
    img_gray_th = (img_gray_th/img_gray_th.max()).astype(float)
    temp=np.reshape(img_gray_th, [1, 224, 224])
    prob_img_gray =np.stack((1-temp,temp), axis=3)
    img_CRF = perform_crf(img_ori, prob_img_gray, 2)
    img_CRF= np.uint8(255*(img_CRF/img_CRF.max()))   # 归一化，量化
    
    # 检测所有的闭合的轮廓
    image11, contours, hier = cv2.findContours(img_CRF, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    try:
        num_bbox=hier.shape[1]
    except:
        return img_CRF, 0, 0
    print('CRF 过滤前bbox个数:',num_bbox)
    img_ori_cp=(img_ori).copy()                                           # 不复制操作的话，cv2.rectangle将修改img_ori
    img_CRF_cp=(img_CRF).copy()
    num=0
    for i in range(num_bbox):
        cnt = contours[i]
        x,y,w,h = cv2.boundingRect(cnt)
        if ((w*h)/(224.0*224.0))>area_th:
            num+=1
            print('arex of bbox %.4f'%((w*h)/(224.0*224.0)) )
            cv2.rectangle(img_ori_cp,(x,y),(x+w,y+h),(255,255,255),2)   # cv2.rectangle函数对image操作是指针操作，具有历史记忆特性
            cv2.rectangle(img_CRF_cp,(x,y),(x+w,y+h),(255,255,255),2)   # cv2.rectangle函数对image操作是指针操作，具有历史记忆特性
            p_x= x
            p_y= (y+10) if (y-20)<=0  else  (y-20) 
            img_ori_cp=label_into_img(img=img_ori_cp, string=label_text, 
                                          position=(p_x, p_y), type_color=(255, 255, 255), type_big=15)
        else:
            continue
    img_crf_bbox=img_ori_cp
    
    if num==0:
        return img_CRF, 0, img_CRF_cp
    
    if plt_on==True:
        # 可视化分析
        plt.figure(figsize=(14,3))
        plt.subplot(1,3,1),plt.imshow(img_CRF),plt.title('img_CRF')
        plt.subplot(1,3,2),plt.imshow(img_CRF_cp),plt.title('img_CRF + bbox')
        plt.subplot(1,3,3),plt.imshow(img_crf_bbox),plt.title('Input image + bbox')
                  
    return img_CRF, img_crf_bbox, img_CRF_cp

def cal_fuse_Grad_CAM_pp(CAM_a=None,CAM_b=None,img_ori=None,alpha=0.3,plt_on=True):
    # 两个热图选择大的输出
    fuse_CAM_gray=np.maximum(CAM_a,CAM_b)
    #fuse_CAM_gray=(CAM_a*1.0+CAM_b*1.0)
    fuse_CAM_gray = cv2.blur(fuse_CAM_gray, (5, 5))   # 平滑滤波
    fuse_CAM_gray -= np.min(fuse_CAM_gray)
    fuse_CAM_gray= np.uint8(255*(fuse_CAM_gray/fuse_CAM_gray.max()))   # 归一化，量化
    
    #heatmap_th=cal_heatmap_threshold(alpha = alpha,gray_img=img_gray)                        # 根据阈值计算热图的分割门限
    heatmap_th =int(256*alpha)
    ret,fuse_CAM_gray_th = cv2.threshold(fuse_CAM_gray, heatmap_th, 255, cv2.THRESH_TOZERO)   # 大于门限的保留，小于门限的置0
    fuse_CAM = cv2.applyColorMap( fuse_CAM_gray, cv2.COLORMAP_JET )   # 三通道彩色图像
    fuse_CAM = cv2.cvtColor( fuse_CAM, cv2.COLOR_BGR2RGB )            # 三通道彩色图像
    
    img = img_ori.astype(float)
    img -= np.min(img)
    img /= img.max()    
    alpha = 0.0025
    new_img = img+alpha*fuse_CAM
    new_img /= new_img.max()
    new_img = np.uint8(255*new_img)
    
    if plt_on==True:
        # 可视化分析
        f, (ax1,ax2,ax3) = plt.subplots(1, 3, sharey=True,figsize=(14,3))
        ax1.imshow(fuse_CAM_gray)  # BGR,格式
        ax1.set_title('fuse_Grad_CAM_gray')
        ax2.imshow(fuse_CAM)  # BGR,格式
        ax2.set_title('fuse_Grad_CAM ++')
        ax3.imshow(new_img)  # BGR,格式
        ax3.set_title('Input image + (fuse_Grad_CAM ++) ')    
    
    return fuse_CAM_gray,fuse_CAM_gray_th,new_img


def np_mask_image(CAM_gray=None, img_input=None,sigma=0.5):

    gcam=CAM_gray
    gcam -= np.min(gcam)
    gcam = gcam/gcam.max()
    mask=(gcam<sigma).astype(float)
    mask_img = np.dstack((img_input[:,:,0]*mask,
                     img_input[:,:,1]*mask,
                     img_input[:,:,2]*mask, )) 
 
    mask_img=np.where(mask_img==0, img_input.mean(), mask_img)
    
    plt.figure(figsize=(14,3))
    plt.subplot(1,4,1),plt.imshow(CAM_gray)
    plt.subplot(1,4,2),plt.imshow(mask)
    plt.subplot(1,4,3),plt.imshow(img_input)
    plt.subplot(1,4,4),plt.imshow(mask_img)   

    return mask_img

def sleclect_label(prob=[0], pred=[0],thresh_prob=0.5001, batch_size=1, number_of_classes=5 ):
    # prob.shape :(batch_size,num_class)
    
    label_slec={}
    label_slec_prob={}
    prob_sort=np.argsort(-prob, axis=1)
    for i in range(batch_size):
        label_slec[i]=[]
        label_slec_prob[i]=[]
        for j in range( min(int(1/thresh_prob), number_of_classes) ):
            label_slec[i].append(prob_sort[i,j])
            label_slec_prob[i].append(prob[i,prob_sort[i,j]])
    return label_slec, label_slec_prob


def run_test(loop_max = 1,\
             number_of_classes = 5,\
             sigma = 0.6,\
             alpha = 0.3,\
             area_th = 0.05,\
             prob_th = 0.5,\
             pred_th = 0.2,\
             plt_on = True,\
             label_text_dir = None,\
             checkpoint_dir = None,\
             test_img_dir = None):
    
    labels_to_class_names=read_label_file(label_text_dir, filename='labels.txt')
    #定义输入变量
    tf.reset_default_graph()
    with tf.Graph().as_default() as tesg:

        # 图片预处理
        #image_arr,label_index=get_one_image(img_dir)
        img_batch,label_batch,size_batch,batch_size = get_batch_image(img_dir=test_img_dir,label_on=True,img_num_max=50)
        image = tf.cast(img_batch, tf.float32)
        r, g, b = tf.split(image, 3, 3)
        img_val = tf.concat([ r -_R_MEAN , g -_G_MEAN ,b -_B_MEAN], 3)    
        ori_images=convet_to_ori(img_val)

        # 推理、训练、准确度
        x_img_in = tf.placeholder(tf.float32,[batch_size, 224, 224, 3]) 
        
        layer_name1='vgg_16/pool5'
        target_conv_layer,target_grad_ac,target_grad_yc,logits_cl,end_points = forward_tran(x_img=x_img_in,\
                                                                                            number_of_classes=number_of_classes,\
                                                                                            layer_name=layer_name1,\
                                                                                            Training=False)   
        conv_in=end_points[layer_name1]
        mask_conv_imgs,_=batch_mask_images(batch_size=batch_size,\
                                                         img_conv = conv_in,\
                                                         target_conv_layer=target_conv_layer,\
                                                         target_grad_ac=target_grad_ac,\
                                                         target_grad_yc=target_grad_yc,\
                                                         sigma=sigma)

        target_conv_layer_a,target_grad_ac_a,target_grad_yc_a,logits_adver,_ = forward_tran_advers(x_img=mask_conv_imgs,\
                                                                                      number_of_classes=number_of_classes,\
                                                                                      Training=False)
        
        label_hot = tf.one_hot(label_batch, number_of_classes)
        label_advers = list(number_of_classes-1-np.array(label_batch))
        label_advers_hot = tf.one_hot(label_advers, number_of_classes)
        accuracy = (evaluation(logits_cl,label_hot) + evaluation(logits_adver,label_advers_hot))/2.0
        prob = tf.nn.softmax(logits_cl)
        pred = tf.nn.sigmoid(logits_cl)
        prob_max = tf.argmax(prob,1)  

        prob_adver=tf.nn.softmax(logits_adver)
        pred_adver=tf.nn.sigmoid(logits_adver)
        prob_max_adver = tf.argmax(prob_adver,1)
        # 
        config = tf.ConfigProto()                                    # 配置GPU参数 
        config.gpu_options.per_process_gpu_memory_fraction = 0.85   # 占用GPU90%的显存 
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


            for step in range(loop_max):
                print('************%d**************'%(step))
                # 准备输入数据
                if step==0:
                    _img_val,_ori_images =sess.run([img_val,ori_images])
                    feed_vars={x_img_in:_img_val }
                else:
                    feed_vars={x_img_in:mask_img }
                # 计算输出结果    
                _accuracy,_prob,_pred,_prob_max,_prob_adver,_pred_adver,_prob_max_adver = sess.run([accuracy,prob,pred,prob_max,\
                                                                      prob_adver,pred_adver,prob_max_adver],\
                                                                     feed_dict=feed_vars)
                _target_conv_layer,_target_grad_ac,_target_grad_yc,_target_conv_layer_a,_target_grad_ac_a,_target_grad_yc_a = sess.run(\
                                                                [target_conv_layer,target_grad_ac,target_grad_yc,\
                                                                target_conv_layer_a,target_grad_ac_a,target_grad_yc_a],\
                                                                feed_dict=feed_vars)
                
                print('test, accuracy = %.3f'%( _accuracy) )
                print('test, real_label is {}, real_adver_label is {} '.format(label_batch,label_advers ) )
                print('test, prob_label is {} ,prob_adver_label is {}'.format(_prob_max, _prob_max_adver) )
                #print('test, prob is {}, prob_adver is {} '.format(_prob, _prob_adver))
                #print('test, pred is {}, pred_adver is {}'.format(_pred, _pred_adver))

                for i in  range(batch_size): 
                    print()
                    # 计算 Grad_CAM 图
                    Grad_CAM_pp_gray, Grad_CAM_pp,new_img = np_Grad_CAM_pp(_target_conv_layer[i], _target_grad_ac[i],\
                                                                _target_grad_yc[i], _ori_images[i],plt_on=plt_on )
                    Grad_CAM_pp_gray_a, Grad_CAM_pp_a,new_img_a = np_Grad_CAM_pp(_target_conv_layer_a[i], _target_grad_ac_a[i],\
                                                                _target_grad_yc_a[i], _ori_images[i],plt_on=plt_on )   

                    fuse_CAM_gray,fuse_CAM_gray_th,new_img_fuse = cal_fuse_Grad_CAM_pp(CAM_a=Grad_CAM_pp_gray,CAM_b=Grad_CAM_pp_gray_a,\
                                         img_ori=_ori_images[i],alpha=alpha,plt_on=plt_on)
                    # 根据热图和CRF处理画出box
                    label_str=labels_to_class_names[_prob_max[i]]
                    img_gray_th, img_with_bbox=plot_box(img_gray=fuse_CAM_gray, img_ori= _ori_images[i], 
                                                        label_text=label_str, alpha=alpha,area_th=area_th,plt_on=plt_on )

                    img_CRF, img_crf_bbox, img_CRF_cp=plot_box_CRF(img_gray=fuse_CAM_gray, img_ori=_ori_images[i], 
                                                                   label_text=label_str, alpha=alpha, area_th=area_th,plt_on=plt_on)
                    
                    # 生成下一轮计算的输入图片 
                    mask_img_1 = np_mask_image(CAM_gray=fuse_CAM_gray, img_input=feed_vars[x_img_in][i],sigma=alpha)
                    mask_img_1=np.reshape(mask_img_1,[1,224,224,3])
                    if i ==0:
                        mask_img=mask_img_1
                    else:
                        mask_img=np.concatenate((mask_img, mask_img_1),axis=0)
                    
                    # 生成 return 要输出的变量
                    img_with_bbox=np.reshape(img_with_bbox,[1,224,224,3])
                    if np.sum(img_crf_bbox)!=0:
                        img_crf_bbox=np.reshape(img_crf_bbox,[1,224,224,3])
                    else:
                        img_crf_bbox=np.reshape(img_with_bbox,[1,224,224,3])
                    if i==0:
                        img_with_bbox_batch = img_with_bbox
                        img_crf_bbox_batch = img_crf_bbox
                    else:
                        img_with_bbox_batch=np.concatenate((img_with_bbox_batch, img_with_bbox),axis=0) 
                        img_crf_bbox_batch=np.concatenate((img_crf_bbox_batch, img_crf_bbox),axis=0) 
                        
                    plt.show()

            coord.request_stop()
            coord.join(threads)
        sess.close()
        print('time use is %s second'%(time.time()-start_time)) 
        return img_with_bbox_batch,img_crf_bbox_batch 
