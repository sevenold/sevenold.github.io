---
layout: post
title: "TensorFlow-keras之CNN实现手写数字识别"
date: 2018-08-27
description: "TensorFlow，keras，实现卷积神经网络"
tag: TensorFlow
---

### TensorFlow四种写法之四：keras

#### 代码：

```
#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/8/23 19:14
# @Author  : Seven
# @Site    : 
# @File    : CNN-keras.py
# @Software: PyCharm

# 0.导入环境
import os
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
from keras.layers.core import Dense, Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.objectives import categorical_crossentropy
from keras import backend as K
K.image_data_format()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 1.数据准备

# 使用tensorflow自带的工具加载MNIST手写数字集合
mnist = input_data.read_data_sets('data', one_hot=True)
# 查看数据的维度和target的维度
print(mnist.train.images.shape)
print(mnist.train.labels.shape)

# 2.准备好palceholder
x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])
learnRate = tf.placeholder(tf.float32)

# 3.构建网络计算图结构

# 把输入数据reshape--28x28=784, 单通道， -1表示None
with tf.name_scope('reshape'):
    x_image = tf.reshape(x, [-1, 28, 28, 1])

# 构建第一层卷积计算层--将一个灰度图像映射到32个feature maps, 卷积核为5x5
net = Conv2D(32, kernel_size=[5, 5], strides=[1, 1],
             activation='relu', padding='same',
             input_shape=[28, 28, 1])(x_image)

# 构建池化层--采用最大池化
net = MaxPooling2D(pool_size=[2, 2])(net)

# 构建第二层卷积计算层--maps 32 feature maps to 64.
net = Conv2D(64, kernel_size=[5, 5], strides=[1, 1],
             activation='relu', padding='same')(net)

# 构建第二层池化层--采用最大池化
net = MaxPooling2D(pool_size=[2, 2])(net)


# 构建全连接层--经过的两层的下采样（池化），28x28x1的图像-->7x7x64，然后映射到1024个特征
net = Flatten()(net)
net = Dense(1024, activation='relu')(net)

# 构建第二层全连接层--将1024个特性映射到10个类，每个类对应一个数字
net = Dense(10, activation='softmax')(net)

# 4.计算损失值并初始化optimizer
cross_entropy = tf.reduce_mean(categorical_crossentropy(y, net))

l2_loss = tf.add_n([tf.nn.l2_loss(w) for w in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)])

total_loss = cross_entropy + 7e-5*l2_loss

train_step = tf.train.AdamOptimizer(learnRate).minimize(total_loss)
# 5.初始化变量
init = tf.global_variables_initializer()

print("FUNCTION READY!!")

# 6.在会话中执行网络定义的运算
with tf.Session() as sess:
    sess.run(init)

    for step in range(3000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        lr = 0.01

        _, loss, l2_loss_value, total_loss_value = sess.run(
            [train_step, cross_entropy, l2_loss, total_loss],
            feed_dict={x: batch_xs, y: batch_ys, learnRate: lr})

        if (step + 1) % 100 == 0:
            print("step %d, entropy loss: %f, l2_loss: %f, total loss: %f" %
                  (step + 1, loss, l2_loss_value, total_loss_value))

            # 验证训练的模型
            correct_prediction = tf.equal(tf.argmax(net, 1), tf.argmax(y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            print("Train accuracy:", sess.run(accuracy, feed_dict={x: batch_xs, y: batch_ys}))

        if (step + 1) % 1000 == 0:
            print("Text accuracy:", sess.run(accuracy, feed_dict={x: batch_xs, y: batch_ys}))


```





