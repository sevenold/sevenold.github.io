---
layout: post
title: "卷积神经网络之AlexNet"
date: 2018-08-25
description: "卷积神经网络、AlexNet"
tag: 深度学习
---



### AlexNet背景

AlexNet是在2012年被发表的一个经典之作，并在当年取得了ImageNet最好成绩，也是在那年之后，更多的更深的神经网路被提出，比如优秀的vgg,GoogleLeNet.

其官方提供的数据模型，准确率达到57.1%,top 1-5 达到80.2%. 这项对于传统的机器学习分类算法而言，已经相当的出色。

### 框架介绍:

AlexNet的结构模型如下：

![images](/images/dl/88.png)



### 示例代码：

#### `AlexNet实现Minist手写数字识别`

```
#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/8/24 22:11
# @Author  : Seven
# @Site    : 
# @File    : AlexNet.py
# @Software: PyCharm

# TODO: 0.导入环境

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# TODO: 1.数据准备

# 使用tensorflow自带的工具加载MNIST手写数字集合
mnist = input_data.read_data_sets('data', one_hot=True)
# 查看数据的维度和target的维度
print(mnist.train.images.shape)
print(mnist.train.labels.shape)

# TODO: 2.初始化变量

# 定义网络超参数
learning_rate = 0.001
epochs = 200000
batch_size = 128
display_step = 5

# 定义网络参数
n_input = 784  # 输入的维度
n_classes = 10  # 标签的维度
dropout = 0.8  # Dropout 的概率

# TODO: 3.准备好placeholder

# 占位符输入
x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.float32, [None, n_classes])
keep_prob = tf.placeholder(tf.float32)

# TODO: 4.构建网络计算图结构


# 卷积操作
def conv2d(name, l_input, w, b):
    return tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(l_input, w, strides=[1, 1, 1, 1], padding='SAME'), b), name=name)


# 最大下采样操作
def max_pool(name, l_input, k):
    return tf.nn.max_pool(l_input, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME', name=name)


# 归一化操作
def norm(name, l_input, lsize=4):
    return tf.nn.lrn(l_input, lsize, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name=name)


# 存储所有的网络参数
weights = {
    'wc1': tf.Variable(tf.random_normal(shape=[11, 11, 1, 96])),
    'wc2': tf.Variable(tf.random_normal(shape=[5, 5, 96, 256])),
    'wc3': tf.Variable(tf.random_normal(shape=[3, 3, 256, 384])),
    'wc4': tf.Variable(tf.random_normal(shape=[3, 3, 384, 384])),
    'wc5': tf.Variable(tf.random_normal(shape=[3, 3, 384, 256])),
    'wd1': tf.Variable(tf.random_normal(shape=[4*4*256, 4096])),
    'wd2': tf.Variable(tf.random_normal(shape=[4096, 1024])),
    'out': tf.Variable(tf.random_normal(shape=[1024, n_classes]))
}
biases = {
    'bc1': tf.Variable(tf.random_normal([96])),
    'bc2': tf.Variable(tf.random_normal([256])),
    'bc3': tf.Variable(tf.random_normal([384])),
    'bc4': tf.Variable(tf.random_normal([384])),
    'bc5': tf.Variable(tf.random_normal([256])),
    'bd1': tf.Variable(tf.random_normal([4096])),
    'bd2': tf.Variable(tf.random_normal([1024])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}


# 定义整个网络
def alex_net(_X, _weights, _biases, _dropout):
    # 向量转为矩阵
    _X = tf.reshape(_X, shape=[-1, 28, 28, 1])
    print(_X.shape)
    # TODO: 第一层卷积：
    conv1 = conv2d('conv1', _X, _weights['wc1'], _biases['bc1'])
    # 下采样层
    pool1 = max_pool('pool1', conv1, k=2)
    # 归一化层
    norm1 = norm('norm1', pool1, lsize=4)
    print(norm1.shape)
    # TODO: 第二层卷积：
    conv2 = conv2d('conv2', norm1, _weights['wc2'], _biases['bc2'])
    # 下采样
    pool2 = max_pool('pool2', conv2, k=2)
    # 归一化
    norm2 = norm('norm2', pool2, lsize=4)
    print(norm2.shape)
    # TODO: 第三层卷积：
    conv3 = conv2d('conv3', norm2, _weights['wc3'], _biases['bc3'])
    # 归一化
    norm3 = norm('norm3', conv3, lsize=4)
    print(norm3.shape)
    # TODO: 第四层卷积
    # 卷积
    conv4 = conv2d('conv4', norm3, _weights['wc4'], _biases['bc4'])
    # 归一化
    norm4 = norm('norm4', conv4, lsize=4)
    print(norm4.shape)
    # TODO: 第五层卷积
    # 卷积
    conv5 = conv2d('conv5', norm4, _weights['wc5'], _biases['bc5'])
    # 下采样
    pool5 = max_pool('pool5', conv5, k=2)
    # 归一化
    norm5 = norm('norm5', pool5, lsize=4)
    print(norm5.shape)
    # TODO: 第六层全连接层
    # 先把特征图转为向量
    dense1 = tf.reshape(norm5, [-1, _weights['wd1'].get_shape().as_list()[0]])
    dense1 = tf.nn.relu(tf.matmul(dense1, _weights['wd1']) + _biases['bd1'], name='fc1')
    dense1 = tf.nn.dropout(dense1, _dropout)
    print(dense1.shape)
    # TODO: 第七层全连接层：
    dense2 = tf.nn.relu(tf.matmul(dense1, _weights['wd2']) + _biases['bd2'], name='fc2')  # Relu activation
    dense2 = tf.nn.dropout(dense2, _dropout)
    print(dense2.shape)
    # TODO: 第八层全连接层：
    # 网络输出层
    out = tf.matmul(dense2, _weights['out']) + _biases['out']
    print(out.shape)
    return out


# TODO: 5.计算损失值并初始化optimizer

# 构建模型
logits = alex_net(x, weights, biases, keep_prob)
# 定义损失函数和学习步骤
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
training_operation = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

print("FUNCTION READY!!")

# TODO: 6.模型评估
# 测试网络
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# TODO: 7.初始化变量
# 初始化所有的共享变量
init = tf.global_variables_initializer()

# TODO: 8.训练模型
# 开启一个训练
print("Training....")
with tf.Session() as sess:
    sess.run(init)
    # Keep training until reach max iterations
    for n in range(epochs):
        total_epochs = int(mnist.train.num_examples/batch_size)
        # 遍历所有epoch
        for epoch in range(total_epochs):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            # 获取批数据
            sess.run(training_operation, feed_dict={x: batch_xs, y: batch_ys, keep_prob: dropout})
        if n % display_step == 0:
            # 计算精度和损失值
            loss, acc = sess.run([cost, accuracy_operation], feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.})
            print("epoch :", (n+1))
            print("Minibatch Loss= {:.6f}".format(loss))
            print("Training Accuracy= {:.5f}".format(acc))

    print("Optimization Finished!")
    # 计算测试精度
    print("Testing Accuracy:",
          sess.run(accuracy_operation, feed_dict={x: mnist.test.images[:256],
                                                  y: mnist.test.labels[:256],
                                                  keep_prob: 1.}))


```







