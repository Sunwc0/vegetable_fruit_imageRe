# # -*- coding: utf-8 -*-
# #map 生成新的序列
# li = [11, 22, 33]
# sl = [1, 2, 3]
# new_list = map(lambda a, b: a + b, li, sl)
# for i in new_list:
#     print i
#
# #lambda代替简单的def
# my_lambda = lambda a:a+100
# print my_lambda(100)
#
# #filter第一个参数为空，将获取原来序列
# new_list = filter(lambda arg: arg > 22, li)
# print new_list[0]
#
# #reduce 对序列元素进行累计操作
# new_li = range(101)
# result = reduce(lambda arg1, arg2: arg1 + arg2, new_li)
# print result
#
# from tensorflow.contrib.slim.nets import resnet_v2
#
# print resnet_v2.resnet_v2([299,299,0,1],)
import numpy as np
import tensorflow as tf
import os

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# lales = ['potato','cherryTomatoes','tomato']
# a = np.array([3, 1, 2, 4, 6, 1])
# print(np.argmax(a))
#
# for i in range(10001):
#     print i
#     if i>5000 and i% 20:
#         print "----",i,"-----"
#
# import tensorflow as tf
# import numpy as np
#
# A = [[1, 3, 4, 5, 6]]
# B = [[1, 3, 4, 3, 2]]
#
# with tf.Session() as sess:
#     print(sess.run(tf.equal(A, B)))
#     correct_pre = tf.equal(A,B)
#     print(sess.run(tf.cast(correct_pre,tf.float32))) #todo there is a big question
#     pre_ = tf.cast(correct_pre,tf.float32)
#     print sess.run(tf.reduce_mean(pre_))
#
# a=[1,2,3]
# b=[4,5,6]
# a=a+b
# print a