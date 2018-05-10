# -*- coding: utf-8 -*-
import tensorflow as tf
import inception_resnet_v2 as net
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' #装的cpu版tensorflow试训练 避免报错

def decode_from_tfrecord(filequeuelist, rows=299, cols=299):
    with tf.name_scope('decode_from_tfrecord'):
        reader = tf.TFRecordReader()  #文件读取
        _,example = reader.read(filequeuelist)
        features = tf.parse_single_example(example, features={'image_raw':tf.FixedLenFeature([], tf.string),
                                                            'label': tf.FixedLenFeature([], tf.int64)})
        image = tf.decode_raw(features['image_raw'], tf.uint8)
        image = tf.reshape(image,[rows,cols,3])
        image = tf.cast(image, tf.float32)/255. - .5
        label = tf.cast(features['label'], tf.int64)
        return image, label


def get_batch(filename_queue,batch_size):
    with tf.name_scope('get_batch'):
        image, label = decode_from_tfrecord(filename_queue)
        # shuffle_batch 读样本时随机效果不好
        images, labels = tf.train.shuffle_batch([image, label], batch_size=batch_size, num_threads=2,
                                                capacity=20001, min_after_dequeue=600)
        return images, labels


def generate_filenamequeue(filequeuelist):
    filename_queue = tf.train.string_input_producer([filequeuelist], num_epochs=50)
    return filename_queue

batch_size = 30
lr = tf.Variable(0.0005, dtype=tf.float32)  # 下降梯度/学习率
x_train = tf.placeholder(tf.float32, [batch_size, 299, 299, 3],name="data-input")
y_train = tf.placeholder(tf.int64, [batch_size,],name="label-input")  # 训练集 num_class 你所要训练的分类类别
is_training = tf.placeholder(tf.bool,name="is_training")
step = 0

logits,endpoint = net.inception_resnet_v2(x_train,num_classes=12,is_training=is_training) # todo inception_resnet_2全连接层的权重值调整
#创建损失函数
loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf.one_hot(y_train,12),logits=logits)
loss0 = tf.reduce_mean(loss)
#定义一个梯度下降法来训练的优化器 并优化最下函数 梯度根据经验定义
optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss0)

#计算模型的准确度
correct_pred = tf.equal(tf.argmax(logits,1),tf.argmax(tf.one_hot(y_train,12),1))
acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32),name="acc")

try:
    images, labels = get_batch(generate_filenamequeue("../TFrecord/train.tfrecords"),batch_size)
    test_images, test_labels = get_batch(generate_filenamequeue("../Tfrecord/test_train.tfrecords"),batch_size)
    # 初始化
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    saver = tf.train.Saver()
    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state('./model_output')
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            sess.run(init_op)

        summary_writer = tf.summary.FileWriter('./logs', sess.graph)  # todo 阿里云训练时记录日志 tensorboard可视化训练过程

        # 创建一个协调器，管理线程
        coord = tf.train.Coordinator()
        # 启动队列
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        # 队列读取未停止，迭代次数小于10000
        while not coord.should_stop() and step <= 10000:
            # for i in range(101):
            image, label = sess.run([images, labels])
            sess.run(optimizer, feed_dict={x_train: image, y_train: label, is_training: True})

            # 每迭代20次，计算一次loss和准确率
            if step >= 20 and step % 20 == 0:
                test_image, test_label = sess.run([test_images, test_labels])
                acc0, loss_ = sess.run([acc, loss0],
                                       feed_dict={x_train: test_image, y_train: test_label,
                                                  is_training: False})  # 此处应输入验证集来输出模型的精度

                learning_rate = sess.run(lr)
                print("Iter:{}, Loss:{}, Accuracy:{}, Learning_rate:{}".format(step, loss_, acc0, learning_rate))
                # 保存模型
                if acc0 >= 0.85:
                    # saver.save(sess, 'save_path',global_step=i) 保存到你自己的save_path
                    sess.save(sess,os.path.join(os.getcwd(), 'inception_resnet_v3_model.ckpt'),global_step = i)#保存在本地目录
                    print "Model done"
                    break
                # 每迭代2000次，降低一次学习率
                if step % 20 == 0:
                    sess.run(tf.assign(lr, lr / 2))

            step = step + 1

except tf.errors.OutOfRangeError:
    print "the image data queue is over"
finally:
    # 通知其他线程关闭
    coord.request_stop()
    # 其他所有线程关闭之后,这一函数才能返回
    coord.join(threads)