---
layout: post
title: "TensorFlow-slim之CNN实现手写数字识别"
date: 2018-08-27
description: "TensorFlow，slim，实现卷积神经网络"
tag: TensorFlow
---

### TensorFlow四种写法之三：Slim

#### 代码：

```
#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/8/23 19:40
# @Author  : Seven
# @Site    : 
# @File    : CNN-slim.py
# @Software: PyCharm

# 0.导入环境

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import os
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
    print(x_image.shape)

# 构建第一层卷积计算层--将一个灰度图像映射到32个feature maps, 卷积核为5x5
with tf.name_scope('conv1'):
    h_conv1 = tf.contrib.slim.conv2d(x_image, 32, [5, 5],
                                     padding='SAME',
                                     activation_fn=tf.nn.relu)
    print(h_conv1.shape)
# 构建池化层--采用最大池化
with tf.name_scope('pool1'):
    h_pool1 = tf.contrib.slim.max_pool2d(h_conv1, [2, 2], stride=2, padding='VALID')
    print(h_pool1.shape)
# 构建第二层卷积计算层--maps 32 feature maps to 64.
with tf.name_scope('conv2'):
    h_conv2 = tf.contrib.slim.conv2d(h_pool1, 64, [5, 5],
                                     padding='SAME',
                                     activation_fn=tf.nn.relu)
    print(h_conv2.shape)
# 构建第二个池化层
with tf.name_scope('pool2'):
    h_pool2 = tf.contrib.slim.max_pool2d(h_conv2, [2, 2], stride=[2, 2], padding='VALID')
    print(h_pool2.shape)
# 构建全连接层--经过的两层的下采样（池化），28x28x1的图像-->7x7x64，然后映射到1024个特征
with tf.name_scope('fc1'):
    h_pool2_flat = tf.contrib.slim.avg_pool2d(h_pool2, h_pool2.shape[1:3],
                                              stride=[1, 1], padding='VALID')
    h_fc1 = tf.contrib.slim.conv2d(h_pool2_flat, 1024, [1, 1],
                                   activation_fn=tf.nn.relu)
    print(h_fc1.shape)
# Dropout--防止过拟合
with tf.name_scope('dropout'):
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob=keep_prob)


# 构建全连接层--将1024个特性映射到10个类，每个类对应一个数字
with tf.name_scope('fc2'):
    out = tf.squeeze(tf.contrib.slim.conv2d(h_fc1_drop, 10, [1, 1], activation_fn=None))
    print(out.shape)

# 4.计算损失值并初始化optimizer
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=out))

l2_loss = tf.add_n([tf.nn.l2_loss(w) for w in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)])

total_loss = cross_entropy + 7e-5*l2_loss

train_step = tf.train.AdamOptimizer(learnRate).minimize(total_loss)
# 5.初始化变量
init = tf.global_variables_initializer()

# 6.在会话中执行网络定义的运算
with tf.Session() as sess:
    sess.run(init)

    for step in range(3000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        lr = 0.01

        _, loss, l2_loss_value, total_loss_value = sess.run(
            [train_step, cross_entropy, l2_loss, total_loss],
            feed_dict={x: batch_xs, y: batch_ys, learnRate: lr, keep_prob: 0.5})

        if (step+1) % 100 == 0:
            print("step %d, entropy loss: %f, l2_loss: %f, total loss: %f" %
                  (step+1, loss, l2_loss_value, total_loss_value))

            # 验证训练的模型
            correct_prediction = tf.equal(tf.argmax(out, 1), tf.argmax(y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            print("Train accuracy:", sess.run(accuracy, feed_dict={x: batch_xs, y: batch_ys, keep_prob:0.5}))

        if (step + 1) % 1000 == 0:
            print("Text accuracy:", sess.run(accuracy, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 0.5}))

```





