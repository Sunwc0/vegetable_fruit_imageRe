# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import base64
from StringIO import StringIO
import re
import scipy.misc as misc

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#随机处理选择循序处理 color_ordering 随机生成，几种处理方法
def distort_color(image,color_ordering = 0):
    if color_ordering == 0:
        # image = tf.image.random_brightness(image,max_delta=32. / 255.)#用随机因子调整图像亮度
        image = tf.image.random_saturation(image,lower=0.5,upper=1.5)#随机因子调整RGB图像的饱和度 饱和度的上下界
        # image = tf.image.random_hue(image,max_delta=0.2)#随机因子调整RGB图像的色调
        image = tf.image.random_contrast(image,lower=0.5,upper=1.5)#调整图像对比度
    return tf.clip_by_value(image,0.0,1.0)#调整图片的数值到0.0，1.0之间

#对输入的图像进行预处理
#bbox-标注框,方便截取图像转化为需要关注的部分
def pre_detail_image(image,height,witdh,bbox):
    if bbox is None:
        bbox = tf.constant([0.0,0.0,1.0,1.0],dtype=tf.float32,shape=[1,1,4])
    if image.dtype != tf.float32:
        image = tf.image.convert_image_dtype(image,dtype=tf.float32)
    #随机截取图片
    bbox_begin,bbox_size,_ = tf.image.sample_distorted_bounding_box(tf.shape(image),bounding_boxes=bbox)
    deal_image = tf.slice(image,bbox_begin,bbox_size)
    # deal_image = tf.image.resize_images(deal_image,[height,witdh],method=np.random.randint(4))#随机设置图片的大小
    deal_image = tf.image.resize_images(deal_image, [height, witdh])
    deal_image = tf.image.random_flip_left_right(deal_image)#随机左右翻转图片

    deal_image = distort_color(deal_image,0)
    return deal_image

def resize_image():
    file_path = "/Users/sunwc/Desktop/蔬菜水果图片数据集"
    test_img_path = "/Users/sunwc/Desktop/test_image_data"
    for file in os.listdir(file_path):
        data_path = file_path + "/" + file
        g = 0
        for file_ in os.listdir(data_path):
            image_path = data_path + "/" + file_
            image = Image.open(image_path).convert('RGB')
            img = image.resize((299, 299), Image.ANTIALIAS)
            print test_img_path+"/"+file+"/"+file_[:-5]+g.__str__()+".jpeg"
            img.save(test_img_path+"/"+file+"/"+file_[:-5]+g.__str__()+".jpeg",'jpeg')
            g = g+1
        print g

def main():

    file_path = "/Users/sunwc/Desktop/蔬菜水果图片数据集"
    for file in os.listdir(file_path):
        data_path = file_path+"/"+file
        g = 0
        for file_ in os.listdir(data_path):
            image_path = data_path+"/"+ file_
            image = tf.gfile.FastGFile(image_path).read()
            with tf.Session() as sess:
                input_image = tf.image.decode_jpeg(image)
                #处理成299X299的图片
                for i in range(3):
                    dealed_image = pre_detail_image(input_image,299,299,bbox=None)
                    #将所处理的图片保存至当前文件夹
                    img_data = tf.image.convert_image_dtype(dealed_image, dtype=tf.uint8)
                    encoded_image = tf.image.encode_jpeg(img_data)
                    #todo file_path = "/Users/sunwc/Desktop/image/tomato" 这个文件夹下存在不知名隐藏文件会报错 存在.DS.store
                    with tf.gfile.GFile("/Users/sunwc/Desktop/image_data/"+file+"/"+file_[:-5]+g.__str__()+i.__str__()+".jpeg", 'wb') as f:
                        f.write(encoded_image.eval())
            # data_path = "/Users/sunwc/Desktop/蔬菜水果图片数据集/"+file
            g=g+1
    print g

if __name__ == '__main__':
    # main()
    resize_image()