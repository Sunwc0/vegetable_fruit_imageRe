# -*- coding: utf-8 -*-

#验证模型的效果 输入一张图片并resize成299x299输入

from PIL import Image
import tensorflow as tf
import scipy.misc as misc

import numpy as np
from PIL import Image
import os
import inception_resnet_v2


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

MODEL_SAVE_PATH = '/Users/xxx/Desktop/model/'
slim = tf.contrib.slim

labels = ['土豆','圣女果','芒果','韭菜','大葱','大白菜','香蕉','胡萝卜','梨','黄瓜','西红柿','苹果']

# 训练时各个tensor的name
# logit:InceptionResnetV2/Logits/Logits/BiasAdd
# Predictions:InceptionResnetV2/Logits/Predictions




def validate_img():
    with tf.Graph().as_default():
        # img = Image.open("/Users/xxx/Desktop/image/tomato/timg-5.jpg")
        img = Image.open("/Users/xxx/Desktop/test_image_data/大白菜/timg (287.jpeg")
        # img = Image.open("/Users/xxx/Desktop/test_image_data/梨/tim9.jpeg")
        # img = Image.open("/Users/xxx/Desktop/test_image_data/胡萝卜/timg (1029.jpeg")
        # img = Image.open("/Users/xxx/Desktop/test_image_data/芒果/timg (136.jpeg")
        # img = Image.open("/Users/xxx/Desktop/test_image_data/苹果/0 (236.jpeg")
        # img = Image.open("/Users/xxx/Desktop/test_image_data/香蕉/timg (53.jpeg")
        img = misc.imresize(img, [299, 299])
        img = np.array(img)
        image = tf.cast(img, tf.float32)/255.-.5
        image = tf.image.per_image_standardization(image)
        image = tf.reshape(image, [1, 299, 299, 3])
        p,_ = inception_resnet_v2.inception_resnet_v2(image,is_training=False,num_classes=12) #模型输出结果
        logits = tf.nn.softmax(p)

        x = tf.placeholder(tf.float32, shape=[None,299, 299, 3])
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        
        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(init_op)
            ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
            # 这里的exclusions是不需要读取预训练模型中的Logits,因为默认的类别数目是1000，当你的类别数目不是1000的时候，如果还要读取的话，就会报错
            exclusions = ['InceptionResnetV2/Logits',
                          'InceptionResnetV2/AuxLogits']
            # 创建一个列表，包含除了exclusions之外所有需要读取的变量
            inception_except_logits = slim.get_variables_to_restore(exclude=exclusions)
            # 建立一个从预训练模型checkpoint中读取上述列表中的相应变量的参数的函数
            init_fn = slim.assign_from_checkpoint_fn("/Users/sunwc/Desktop/model/model.ckpt-1600", inception_except_logits,
                                                     ignore_missing_vars=True)
            # 运行该函数
            init_fn(sess)
            if ckpt and ckpt.model_checkpoint_path:
                # 调用saver.restore()函数，加载训练好的网络模型
                saver.restore(sess, "/Users/sunwc/Desktop/model/model.ckpt-1600")
                print('Loading success')
            else:
                print('No checkpoint')

            prediction = sess.run(logits, feed_dict={x: [img]})
            max_index = np.argmax(prediction)
            sprediction = prediction[0].copy()
            sprediction[max_index] = 0
            smax_index = np.argmax(sprediction)
            sprediction[smax_index] = 0
            tmax_index = np.argmax(sprediction)
            print('预测的标签可能依次为：')
            print labels[max_index], labels[smax_index], labels[tmax_index]
            print('置信度分别是：')
            print prediction[0][max_index], prediction[0][smax_index], prediction[0][tmax_index]

if __name__ == '__main__':
    validate_img
