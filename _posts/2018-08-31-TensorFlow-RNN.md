---
layout: post
title: "TensorFlow实现简单的递归神经网络-RNN"
date: 2018-08-31
description: "递归神经网络，长短时记忆，LSTM，手写数字识别"
tag: TensorFlow
---

### 递归神经网络实现MNIST手写数字识别

#### `代码`

```
#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/8/31 13:00
# @Author  : Seven
# @Site    : 
# @File    : RNN.py
# @Software: PyCharm

# TODO: 0.导入环境

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
print("Packages imported")

# TODO: 1.数据准备
mnist = input_data.read_data_sets("data/", one_hot=True)

# TODO: 2.数据处理
x_train, y_train = mnist.train.images, mnist.train.labels
x_test, y_test = mnist.test.images, mnist.test.labels

train_number = x_train.shape[0]
test_number = x_test.shape[0]
dim = y_train.shape[1]
classes_number = y_test.shape[1]
print("MNIST LOADED")

# TODO: 初始化权重参数
input_dim = 28
hidden_dim = 128
output_dim = classes_number
steps = 28

weights = {
    'hidden': tf.Variable(tf.random_normal([input_dim, hidden_dim])),
    'out': tf.Variable(tf.random_normal([hidden_dim, output_dim]))
}

biases = {
    'hidden': tf.Variable(tf.random_normal([hidden_dim])),
    'out': tf.Variable(tf.random_normal([output_dim]))
}

# TODO: 构建网络计算图


def _RNN(_x, _w, _b, _step, _name):
    # 1.把输入进行转换
    # [batchsize, steps, input_dim]==>[steps, batchsize, input_dim]
    _x = tf.transpose(_x, [1, 0, 2])
    # [steps, batchsize, input_dim]==>[steps*batchsize, input_dim]
    _x = tf.reshape(_x, [-1, input_dim])

    # 2. input layer ==> hidden layer
    hidden_layer = tf.add(tf.matmul(_x, _w['hidden']), _b['hidden'])
    # 把数据进行分割
    hidden_layer_data = tf.split(hidden_layer, _step, 0)
    # LSTM
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_dim, forget_bias=0.8)
    lstm_out, lstm_s = tf.nn.static_rnn(lstm_cell, hidden_layer_data, dtype=tf.float32)

    # 3. output
    output = tf.add(tf.matmul(lstm_out[-1], _w['out']), _b['out'])

    return_data = {
        'x': _x, 'hidden': hidden_layer, 'hidden_data': hidden_layer_data,
        'lstm_out': lstm_out, 'lstm_s': lstm_s, 'output': output
    }
    return return_data

# TODO: 5.计算损失值并初始化optimizer


learn_rate = 0.01
x = tf.placeholder(tf.float32, [None, steps, input_dim])
y = tf.placeholder(tf.float32, [None, output_dim])
MyRNN = _RNN(x, weights, biases, steps, 'basic')
# 预测值
prediction = MyRNN['output']
# 计算损失值
cross = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
# 初始化优化器
optimizer = tf.train.AdamOptimizer(learning_rate=learn_rate).minimize(cross)
# 模型评估
accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1)), tf.float32))

print("NETWORK READY!!")

# TODO: 6.初始化变量
init = tf.global_variables_initializer()


# TODO: 7.模型训练
print("Start optimization")
with tf.Session() as sess:
    sess.run(init)
    training_epochs = 50
    batch_size = 128
    display_step = 10
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(mnist.train.num_examples/batch_size)
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            batch_xs = batch_xs.reshape((batch_size, steps, input_dim))

            feeds = {x: batch_xs, y: batch_ys}
            sess.run(optimizer, feed_dict=feeds)

            avg_cost += sess.run(cross, feed_dict=feeds) / total_batch

        if epoch % display_step == 0:
            print("Epoch: %03d/%03d cost: %.9f" % (epoch+1, training_epochs, avg_cost))
            feeds = {x: batch_xs, y: batch_ys}
            train_acc = sess.run(accuracy, feed_dict=feeds)
            print(" Training accuracy: %.3f" % train_acc)
            batch_x_test = mnist.test.images
            batch_y_test = mnist.test.labels
            batch_x_test = batch_x_test.reshape([-1, steps, input_dim])
            test_acc = sess.run(accuracy, feed_dict={x: batch_x_test, y: batch_y_test})
            print(" Test accuracy: %.3f" % test_acc)

print("Optimization Finished.")

```

#### `执行结果`

```
NETWORK READY!!
Start optimization
Epoch: 001/050 cost: 2.033586812
 Training accuracy: 0.422
 Test accuracy: 0.511
Epoch: 011/050 cost: 0.101458139
 Training accuracy: 0.961
 Test accuracy: 0.961
Epoch: 021/050 cost: 0.096797398
 Training accuracy: 0.977
 Test accuracy: 0.956
Epoch: 031/050 cost: 0.105833270
 Training accuracy: 0.992
 Test accuracy: 0.958
Epoch: 041/050 cost: 0.117593214
 Training accuracy: 0.977
 Test accuracy: 0.959
Optimization Finished.

```

