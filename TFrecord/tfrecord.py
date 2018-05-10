# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
import scipy.misc as misc
import sys
import glob
from random import shuffle

sys.path.append(os.getcwd() + '/model')


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
labels = []
batch_size=3

def _int64_feature(value):
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def _byte_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

# 将数据转化为tf.train.Example格式
def _make_example(label, image):
    example = tf.train.Example(features=tf.train.Features(feature={
        'label': _int64_feature(label),
        'image_raw': _byte_feature(image)
    }))
    return example

def encode_to_tfrecord(file_path):
    vagetable_class = []
    addrs = [] #路径列表
    labels = [] #标签列表
    writer = tf.python_io.TFRecordWriter("test_train.tfrecords")
    i=0
    #生成路径对应标签水平的两个列表 addrs[] labels[]
    for file in os.listdir(file_path):
        vagetable_class.append(file)
        data_path = glob.glob(file_path+"/"+file+"/*.jpeg")
        addrs = addrs+data_path
        labels = labels + [i for j in range(data_path.__len__())]
        i = i+1

    if True:
        c = list(zip(addrs, labels))  # 将两列元素进行组合
        shuffle(c)  # random包的shuffle函数进行打乱处理
        new_addrs, new_labels = zip(*c)  # 将组合后的元素再进行拆分
    print "-----------------------------------------------------------"
    new_addrs = list(new_addrs)
    new_labels = list(new_labels)

    if new_labels.__len__() == new_addrs.__len__():
        for i in range(new_addrs.__len__()):
            image = Image.open(new_addrs[i])
            image = misc.imresize(image, [299, 299, 3])
            image_raw = image.tobytes()
            example = _make_example(image=image_raw, label=[new_labels[i]])
            writer.write(example.SerializeToString())
    return vagetable_class

# 读取TFRecord 输出image
def feature(file_name):
    filename_queue = generate_filenamequeue(file_name)
    image,label = decode_from_tfrecord(filename_queue)
    with tf.Session() as sess:
        with tf.device("/cpu:0"):
            # init_op = tf.initialize_all_variables()
            sess.run(tf.local_variables_initializer())
            # sess.run(init_op)
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)
            for i in range(100):
                example, l = sess.run([image, label])  # take out image and label
                img = Image.fromarray(example,"RGB")
                img.save(os.getcwd() + str(i) + '_''Label_' + str(l) + '.jpg')  # save image

            coord.request_stop()
            coord.join(threads)

def decode_from_tfrecord(filequeuelist, rows=299, cols=299):
    with tf.name_scope('decode_from_tfrecord'):
        reader = tf.TFRecordReader()  #文件读取
        _,example = reader.read(filequeuelist)
        features = tf.parse_single_example(example, features={'image_raw':tf.FixedLenFeature([], tf.string),
                                                            'label': tf.FixedLenFeature([], tf.int64)})
        image = tf.decode_raw(features['image_raw'], tf.uint8)
        image = tf.reshape(image,[rows,cols,3])
        image = tf.cast(image, tf.float32)
        label = tf.cast(features['label'], tf.int64)
        return image, label


def get_batch(filename_queue):
    with tf.name_scope('get_batch'):
        [image, label] = decode_from_tfrecord(filename_queue)
        images, labels = tf.train.shuffle_batch([image, label], batch_size=30, num_threads=2,
                                                capacity=180, min_after_dequeue=60)
        return images, labels


def generate_filenamequeue(filequeuelist):
    filename_queue = tf.train.string_input_producer([filequeuelist], num_epochs=3)
    return filename_queue

#查看tfrecord中的图片
def decode_batch(filename, batch_size):
    filename_queue = generate_filenamequeue(filename)
    [images, labels] = get_batch(filename_queue)

    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())
    sess = tf.InteractiveSession()
    sess.run(init_op)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    i = 0
    try:
        while not coord.should_stop():
            image, label = sess.run([images, labels])
            i = i + 1
            if i<3:
                for j in range(batch_size):  # 之前tfrecord编码的时候，数据范围变成[-0.5,0.5],现在相当于逆操作，把数据变成图片像素值
                    # image[j] = (image[j] + 0.5) * 255
                    ar = np.asarray(image[j], np.uint8)
                    # image[j]=tf.cast(image[j],tf.uint8)
                    img = Image.frombytes("RGB", (299, 299), ar.tostring())  # 函数参数中宽度高度要注意。构建299X299的图片 作为Inception_resenet_v2模型的输入
                    img.save(os.getcwd()+"reverse_%d.jpeg" % (j), "jpeg")  # 保存部分图片查看
            '''if(i>710):
                print("step %d"%(i))
                print image
                print label'''
    except tf.errors.OutOfRangeError:
        print('Done training -- epoch limit reached')
    finally:
        # When done, ask the threads to stop.
        coord.request_stop()

    # Wait for threads to finish.
    coord.join(threads)
    sess.close()

def main():
    file_path="/Users/sunwc/Desktop/test_image_data"
    lables = encode_to_tfrecord(file_path)
    str = "["
    for i in lables:
        str = str +"'"+ i + "',"
    print str
    # name = "train.tfrecords"
    # decode_batch(name,32)
    # get_batch(generate_filenamequeue(name),batch_size)

if __name__ == '__main__':
    main()

