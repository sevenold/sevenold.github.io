---
layout: post
title: "TensorFlow-cifar10-图像分类"
date: 2018-09-7
description: "TensorFlow, cifar10"
tag: TensorFlow
---

### **预处理数据**：

```
#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/9/6 15:31
# @Author  : Seven
# @Site    : 
# @File    : Read_data.py
# @Software: PyCharm


# TODO: 加载数据
import pickle
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelBinarizer


def load_cifar10_batch(path, batch_id):
    """
    加载batch的数据
    :param path: 数据存储的目录
    :param batch_id:batch的编号
    :return:features and labels
    """

    with open(path + '/data_batch_' + str(batch_id), mode='rb') as file:
        batch = pickle.load(file, encoding='latin1')

    # features and labels
    features = batch['data'].reshape((len(batch['data']), 3, 32, 32)).transpose(0, 2, 3, 1)
    labels = batch['labels']

    return features, labels


# 数据预处理
def pre_processing_data(x_train, y_train, x_test, y_test):
    # features
    minmax = MinMaxScaler()
    # 重塑数据
    # (50000, 32, 32, 3) --> (50000, 32*32*3)
    x_train_rows = x_train.reshape(x_train.shape[0], 32*32*3)
    # (10000, 32, 32, 3) --> (10000, 32*32*3)
    x_test_rows = x_test.reshape(x_test.shape[0], 32*32*3)
    # 归一化
    x_train_norm = minmax.fit_transform(x_train_rows)
    x_test_norm = minmax.fit_transform(x_test_rows)
    # 重塑数据
    x_train = x_train_norm.reshape(x_train_norm.shape[0], 32, 32, 3)
    x_test = x_test_norm.reshape(x_test_norm.shape[0], 32, 32, 3)

    # labels
    # 对标签进行one-hot
    n_class = 10
    label_binarizer = LabelBinarizer().fit(np.array(range(n_class)))
    y_train = label_binarizer.transform(y_train)
    y_test = label_binarizer.transform(y_test)

    return x_train, y_train, x_test, y_test


def cifar10_data():
    # 加载训练数据
    cifar10_path = 'data'
    # 一共是有5个batch的训练数据
    x_train, y_train = load_cifar10_batch(cifar10_path, 1)
    for n in range(2, 6):
        features, labels = load_cifar10_batch(cifar10_path, n)
        x_train = np.concatenate([x_train, features])
        y_train = np.concatenate([y_train, labels])

    # 加载测试数据
    with open(cifar10_path + '/test_batch', mode='rb') as file:
        batch = pickle.load(file, encoding='latin1')
        x_test = batch['data'].reshape((len(batch['data']), 3, 32, 32)).transpose(0, 2, 3, 1)
        y_test = batch['labels']

    x_train, y_train, x_test, y_test = pre_processing_data(x_train, y_train, x_test, y_test)

    return x_train, y_train, x_test, y_test
```

### **配置文件**：

```
#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/9/6 16:02
# @Author  : Seven
# @Site    : 
# @File    : config.py
# @Software: PyCharm
import tensorflow as tf
import matplotlib.pyplot as plt

# 初始化卷积神经网络参数
keep_prob = 0.8
epochs = 20
batch_size = 128
n_classes = 10  # 总共10类

# 定义输入和标签的placeholder
inputs = tf.placeholder(tf.float32, [None, 32, 32, 3], name='inputs')
targets = tf.placeholder(tf.float32, [None, 10], name='logits')

learning_rate = 0.001


# 显示图片
def show_images(images):
    fig, axes = plt.subplots(nrows=3, ncols=3, sharex=True, sharey=True, figsize=(9, 9))
    img = images[: 60]
    for image, row in zip([img[: 20], img[20: 40], img[40: 60]], axes):
        for img, ax in zip(image, row):
            ax.imshow(img)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

    fig.tight_layout(pad=0.1)
    # plt.show()


# 存储alexnet所有的网络参数
weights = {
    'wc1': tf.Variable(tf.random_normal(shape=[11, 11, 3, 96])),
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

```

### **LeNet网络**：

```
def LeNet(inputs):

    mu = 0
    sigma = 0.1
    print(inputs.shape)
    # TODO: 第一层卷积：输入=32x32x3, 输出=28x28x6
    conv1_w = tf.Variable(tf.truncated_normal(shape=[5, 5, 3, 6], mean=mu, stddev=sigma))
    conv1_b = tf.Variable(tf.zeros(6))

    conv1 = tf.nn.conv2d(inputs, conv1_w, strides=[1, 1, 1, 1], padding='VALID') + conv1_b
    print(conv1.shape)
    # 激活函数
    conv1_out = tf.nn.relu(conv1)

    # 池化层， 输入=28x28x6, 输出=14x14x6
    pool_1 = tf.nn.max_pool(conv1_out, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    print(pool_1.shape)

    # TODO: 第二层卷积： 输入=14x14x6， 输出=10x10x16
    conv2_w = tf.Variable(tf.truncated_normal(shape=[5, 5, 6, 16], mean=mu, stddev=sigma))
    conv2_b = tf.Variable(tf.zeros(16))

    conv2 = tf.nn.conv2d(pool_1, conv2_w, strides=[1, 1, 1, 1], padding='VALID') + conv2_b
    print(conv2.shape)
    # 激活函数
    conv2_out = tf.nn.relu(conv2)

    # 池化层， 输入=10x10x16, 输出=5x5x16
    pool_2 = tf.nn.max_pool(conv2_out, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    print(pool_2.shape)
    # Flatten 输入=5x5x16， 输出=400
    pool_2_flat = tf.reshape(pool_2, [-1, 400])

    # TODO: 第三层全连接层， 输入=400， 输出=120
    fc1_w = tf.Variable(tf.truncated_normal(shape=[400, 120], mean=mu, stddev=sigma))
    fc1_b = tf.Variable(tf.zeros(120))

    fc1 = tf.matmul(pool_2_flat, fc1_w) + fc1_b

    # 激活函数
    fc1_out = tf.nn.relu(fc1)
    print(fc1_out.shape)

    # TODO: 第四层全连接层： 输入=120， 输出=84
    fc2_w = tf.Variable(tf.truncated_normal(shape=[120, 84], mean=mu, stddev=sigma))
    fc2_b = tf.Variable(tf.zeros(84))

    fc2 = tf.matmul(fc1_out, fc2_w) + fc2_b

    # 激活函数
    fc2_out = tf.nn.relu(fc2)
    print(fc2_out.shape)

    # TODO: 第五层全连接层： 输入=84， 输出=10
    fc3_w = tf.Variable(tf.truncated_normal(shape=[84, 10], mean=mu, stddev=sigma))
    fc3_b = tf.Variable(tf.zeros(10))

    fc3_out = tf.matmul(fc2_out, fc3_w) + fc3_b
    print(fc3_out.shape)

    return fc3_out

```

### **AlexNet网络**：

```
# 卷积操作
def conv2d(name, l_input, w, b):
    return tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(l_input, w, strides=[1, 1, 1, 1], padding='SAME'), b), name=name)


# 最大下采样操作
def max_pool(name, l_input, k):
    return tf.nn.max_pool(l_input, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME', name=name)


# 归一化操作
def norm(name, l_input, lsize=4):
    return tf.nn.lrn(l_input, lsize, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name=name)


# 定义整个网络
def alex_net(_X, _weights, _biases, _dropout):
    # 向量转为矩阵
    # _X = tf.reshape(_X, shape=[-1, 28, 28, 3])
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
```

### **VGG16Net网络**：

```
def VGG16(inputs):
    print(inputs.shape)
    # (32x32x3) --> (32x32x64)
    with tf.name_scope('conv_1'):
         conv_1_out = tf.layers.conv2d(inputs, 64, [3, 3],
                                       padding='same',
                                       activation=tf.nn.relu,
                                       kernel_initializer=tf.truncated_normal_initializer(mean=0., stddev=0.1))
    print(conv_1_out.shape)
    # (32x32x64) --> (32x32x64)
    with tf.name_scope('conv_2'):
        conv_2_out = tf.layers.conv2d(conv_1_out, 64, [3, 3],
                                      padding='same',
                                      activation=tf.nn.relu,
                                      kernel_initializer=tf.truncated_normal_initializer(mean=0., stddev=0.1))
    print(conv_2_out.shape)
    # (32x32x64) --> (16x16x64)
    with tf.name_scope('pool_1'):
        pool_1_out = tf.layers.max_pooling2d(conv_2_out, pool_size=[2, 2], strides=[2, 2], padding='same')

    print(pool_1_out.shape)
    # (16x16x64) --> (16x16x128)
    with tf.name_scope('conv_3'):
         conv_3_out = tf.layers.conv2d(pool_1_out, 128, [3, 3],
                                       padding='same',
                                       activation=tf.nn.relu,
                                       kernel_initializer=tf.truncated_normal_initializer(mean=0., stddev=0.1))
    print(conv_3_out.shape)
    # (16x16x128) --> (16x16x128)
    with tf.name_scope('conv_4'):
         conv_4_out = tf.layers.conv2d(conv_3_out, 128, [3, 3],
                                       padding='same',
                                       activation=tf.nn.relu,
                                       kernel_initializer=tf.truncated_normal_initializer(mean=0., stddev=0.1))
    print(conv_4_out.shape)
    # (16x16x128) --> (8x8x128)
    with tf.name_scope('pool_2'):
        pool_2_out = tf.layers.max_pooling2d(conv_4_out, pool_size=[2, 2], strides=[2, 2], padding='same')

    print(pool_2_out.shape)
    # (8x8x128) --> (8x8x256)
    with tf.name_scope('conv_5'):
         conv_5_out = tf.layers.conv2d(pool_2_out, 256, [3, 3],
                                       padding='same',
                                       activation=tf.nn.relu,
                                       kernel_initializer=tf.truncated_normal_initializer(mean=0., stddev=0.1))
    print(conv_5_out.shape)
    # (8x8x256) --> (8x8x256)
    with tf.name_scope('conv_6'):
         conv_6_out = tf.layers.conv2d(conv_5_out, 256, [3, 3],
                                       padding='same',
                                       activation=tf.nn.relu,
                                       kernel_initializer=tf.truncated_normal_initializer(mean=0., stddev=0.1))
    print(conv_6_out.shape)
    # (8x8x256) --> (8x8x256)
    with tf.name_scope('conv_7'):
        conv_7_out = tf.layers.conv2d(conv_6_out, 256, [3, 3],
                                      padding='same',
                                      activation=tf.nn.relu,
                                      kernel_initializer=tf.truncated_normal_initializer(mean=0., stddev=0.1))
    print(conv_7_out.shape)
    # (8x8x256) --> (4x4x256)
    with tf.name_scope('pool_3'):
        pool_3_out = tf.layers.max_pooling2d(conv_7_out, pool_size=[2, 2], strides=[2, 2], padding='same')

    print(pool_3_out.shape)
    # (4x4x256) --> (4x4x512)
    with tf.name_scope('conv_8'):
        conv_8_out = tf.layers.conv2d(pool_3_out, 512, [3, 3],
                                      padding='same',
                                      activation=tf.nn.relu,
                                      kernel_initializer=tf.truncated_normal_initializer(mean=0., stddev=0.1))
    print(conv_8_out.shape)
    # (4x4x512) --> (4x4x512)
    with tf.name_scope('conv_9'):
        conv_9_out = tf.layers.conv2d(conv_8_out, 512, [3, 3],
                                      padding='same',
                                      activation=tf.nn.relu,
                                      kernel_initializer=tf.truncated_normal_initializer(mean=0., stddev=0.1))
    print(conv_9_out.shape)
    # (4x4x512) --> (4x4x512)
    with tf.name_scope('conv_10'):
        conv_10_out = tf.layers.conv2d(conv_9_out, 512, [3, 3],
                                       padding='same',
                                       activation=tf.nn.relu,
                                       kernel_initializer=tf.truncated_normal_initializer(mean=0., stddev=0.1))
    print(conv_10_out.shape)
    # (4x4x512) --> (2x2x512)
    with tf.name_scope('pool_4'):
        pool_4_out = tf.layers.max_pooling2d(conv_10_out, pool_size=[2, 2], strides=[2, 2], padding='same')

    print(pool_4_out.shape)
    # (2x2x512) --> (2x2x512)
    with tf.name_scope('conv_11'):
        conv_11_out = tf.layers.conv2d(pool_4_out, 512, [3, 3],
                                       padding='same',
                                       activation=tf.nn.relu,
                                       kernel_initializer=tf.truncated_normal_initializer(mean=0., stddev=0.1))
    print(conv_11_out.shape)
    # (2x2x512) --> (2x2x512)
    with tf.name_scope('conv_12'):
        conv_12_out = tf.layers.conv2d(conv_11_out, 512, [3, 3],
                                       padding='same',
                                       activation=tf.nn.relu,
                                       kernel_initializer=tf.truncated_normal_initializer(mean=0., stddev=0.1))
    print(conv_12_out.shape)
    # (2x2x512) --> (2x2x512)
    with tf.name_scope('conv_13'):
        conv_13_out = tf.layers.conv2d(conv_12_out, 512, [3, 3],
                                       padding='same',
                                       activation=tf.nn.relu,
                                       kernel_initializer=tf.truncated_normal_initializer(mean=0., stddev=0.1))
    print(conv_13_out.shape)
    # (2x2x512) --> (1x1x512)
    with tf.name_scope('pool_5'):
        pool_5_out = tf.layers.max_pooling2d(conv_13_out, pool_size=[2, 2], strides=[2, 2], padding='same')

    print(pool_5_out.shape)
    # (1x1x512) --> 512
    with tf.name_scope('fc_1'):
        pool_5_outz_flat = tf.layers.flatten(pool_5_out)
        fc_1_out = tf.layers.dense(pool_5_outz_flat, 512, activation=tf.nn.relu)
        fc_1_drop = tf.nn.dropout(fc_1_out, keep_prob=config.keep_prob)
    print(fc_1_drop.shape)
    # 512 --> 512
    with tf.name_scope('fc_2'):
        fc_2_out = tf.layers.dense(fc_1_drop, 512, activation=tf.nn.relu)
        fc_2_drop = tf.nn.dropout(fc_2_out, keep_prob=config.keep_prob)
    print(fc_2_drop.shape)
    # 512 --> 10
    with tf.name_scope('fc_3'):
        fc_3_out = tf.layers.dense(fc_2_drop, 10, activation=None)
    print(fc_3_out.shape)

    return fc_3_out
```

### **VGG19Net**:

```

def VGG19(inputs):
    print(inputs.shape)
    # (32x32x3) --> (32x32x64)
    with tf.name_scope('conv_1'):
         conv_1_out = tf.layers.conv2d(inputs, 64, [3, 3],
                                       padding='same',
                                       activation=tf.nn.relu,
                                       kernel_initializer=tf.truncated_normal_initializer(mean=0., stddev=0.1))
    print(conv_1_out.shape)
    # (32x32x64) --> (32x32x64)
    with tf.name_scope('conv_2'):
        conv_2_out = tf.layers.conv2d(conv_1_out, 64, [3, 3],
                                      padding='same',
                                      activation=tf.nn.relu,
                                      kernel_initializer=tf.truncated_normal_initializer(mean=0., stddev=0.1))
    print(conv_2_out.shape)
    # (32x32x64) --> (16x16x64)
    with tf.name_scope('pool_1'):
        pool_1_out = tf.layers.max_pooling2d(conv_2_out, pool_size=[2, 2], strides=[2, 2], padding='same')

    print(pool_1_out.shape)
    # (16x16x64) --> (16x16x128)
    with tf.name_scope('conv_3'):
         conv_3_out = tf.layers.conv2d(pool_1_out, 128, [3, 3],
                                       padding='same',
                                       activation=tf.nn.relu,
                                       kernel_initializer=tf.truncated_normal_initializer(mean=0., stddev=0.1))
    print(conv_3_out.shape)
    # (16x16x128) --> (16x16x128)
    with tf.name_scope('conv_4'):
         conv_4_out = tf.layers.conv2d(conv_3_out, 128, [3, 3],
                                       padding='same',
                                       activation=tf.nn.relu,
                                       kernel_initializer=tf.truncated_normal_initializer(mean=0., stddev=0.1))
    print(conv_4_out.shape)
    # (16x16x128) --> (8x8x128)
    with tf.name_scope('pool_2'):
        pool_2_out = tf.layers.max_pooling2d(conv_4_out, pool_size=[2, 2], strides=[2, 2], padding='same')

    print(pool_2_out.shape)
    # (8x8x128) --> (8x8x256)
    with tf.name_scope('conv_5'):
         conv_5_out = tf.layers.conv2d(pool_2_out, 256, [3, 3],
                                       padding='same',
                                       activation=tf.nn.relu,
                                       kernel_initializer=tf.truncated_normal_initializer(mean=0., stddev=0.1))
    print(conv_5_out.shape)
    # (8x8x256) --> (8x8x256)
    with tf.name_scope('conv_6'):
         conv_6_out = tf.layers.conv2d(conv_5_out, 256, [3, 3],
                                       padding='same',
                                       activation=tf.nn.relu,
                                       kernel_initializer=tf.truncated_normal_initializer(mean=0., stddev=0.1))
    print(conv_6_out.shape)
    # (8x8x256) --> (8x8x256)
    with tf.name_scope('conv_7'):
        conv_7_out = tf.layers.conv2d(conv_6_out, 256, [3, 3],
                                      padding='same',
                                      activation=tf.nn.relu,
                                      kernel_initializer=tf.truncated_normal_initializer(mean=0., stddev=0.1))
    print(conv_7_out.shape)
    # (8x8x256) --> (8x8x256)
    with tf.name_scope('conv_8'):
        conv_8_out = tf.layers.conv2d(conv_7_out, 256, [3, 3],
                                      padding='same',
                                      activation=tf.nn.relu,
                                      kernel_initializer=tf.truncated_normal_initializer(mean=0., stddev=0.1))
    print(conv_8_out.shape)
    # (8x8x256) --> (4x4x256)
    with tf.name_scope('pool_3'):
        pool_3_out = tf.layers.max_pooling2d(conv_8_out, pool_size=[2, 2], strides=[2, 2], padding='same')

    print(pool_3_out.shape)
    # (4x4x256) --> (4x4x512)
    with tf.name_scope('conv_9'):
        conv_9_out = tf.layers.conv2d(pool_3_out, 512, [3, 3],
                                      padding='same',
                                      activation=tf.nn.relu,
                                      kernel_initializer=tf.truncated_normal_initializer(mean=0., stddev=0.1))
    print(conv_9_out.shape)
    # (4x4x512) --> (4x4x512)
    with tf.name_scope('conv_10'):
        conv_10_out = tf.layers.conv2d(conv_9_out, 512, [3, 3],
                                       padding='same',
                                       activation=tf.nn.relu,
                                       kernel_initializer=tf.truncated_normal_initializer(mean=0., stddev=0.1))
    print(conv_10_out.shape)
    # (4x4x512) --> (4x4x512)
    with tf.name_scope('conv_11'):
        conv_11_out = tf.layers.conv2d(conv_10_out, 512, [3, 3],
                                       padding='same',
                                       activation=tf.nn.relu,
                                       kernel_initializer=tf.truncated_normal_initializer(mean=0., stddev=0.1))
    print(conv_11_out.shape)
    # (4x4x512) --> (4x4x512)
    with tf.name_scope('conv_12'):
        conv_12_out = tf.layers.conv2d(conv_11_out, 512, [3, 3],
                                       padding='same',
                                       activation=tf.nn.relu,
                                       kernel_initializer=tf.truncated_normal_initializer(mean=0., stddev=0.1))
    print(conv_12_out.shape)
    # (4x4x512) --> (2x2x512)
    with tf.name_scope('pool_4'):
        pool_4_out = tf.layers.max_pooling2d(conv_12_out, pool_size=[2, 2], strides=[2, 2], padding='same')

    print(pool_4_out.shape)
    # (2x2x512) --> (2x2x512)
    with tf.name_scope('conv_13'):
        conv_13_out = tf.layers.conv2d(pool_4_out, 512, [3, 3],
                                       padding='same',
                                       activation=tf.nn.relu,
                                       kernel_initializer=tf.truncated_normal_initializer(mean=0., stddev=0.1))
    print(conv_13_out.shape)
    # (2x2x512) --> (2x2x512)
    with tf.name_scope('conv_14'):
        conv_14_out = tf.layers.conv2d(conv_13_out, 512, [3, 3],
                                       padding='same',
                                       activation=tf.nn.relu,
                                       kernel_initializer=tf.truncated_normal_initializer(mean=0., stddev=0.1))
    print(conv_14_out.shape)
    # (2x2x512) --> (2x2x512)
    with tf.name_scope('conv_15'):
        conv_15_out = tf.layers.conv2d(conv_14_out, 512, [3, 3],
                                       padding='same',
                                       activation=tf.nn.relu,
                                       kernel_initializer=tf.truncated_normal_initializer(mean=0., stddev=0.1))
    print(conv_15_out.shape)
    # (2x2x512) --> (2x2x512)
    with tf.name_scope('conv_16'):
        conv_16_out = tf.layers.conv2d(conv_15_out, 512, [3, 3],
                                       padding='same',
                                       activation=tf.nn.relu,
                                       kernel_initializer=tf.truncated_normal_initializer(mean=0., stddev=0.1))
    print(conv_16_out.shape)
    # (2x2x512) --> (1x1x512)
    with tf.name_scope('pool_5'):
        pool_5_out = tf.layers.max_pooling2d(conv_16_out, pool_size=[2, 2], strides=[2, 2], padding='same')

    print(pool_5_out.shape)
    # (1x1x512) --> 512
    with tf.name_scope('fc_1'):
        pool_5_outz_flat = tf.layers.flatten(pool_5_out)
        fc_1_out = tf.layers.dense(pool_5_outz_flat, 512, activation=tf.nn.relu)
        fc_1_drop = tf.nn.dropout(fc_1_out, keep_prob=config.keep_prob)
    print(fc_1_drop.shape)
    # 512 --> 512
    with tf.name_scope('fc_2'):
        fc_2_out = tf.layers.dense(fc_1_drop, 512, activation=tf.nn.relu)
        fc_2_drop = tf.nn.dropout(fc_2_out, keep_prob=config.keep_prob)
    print(fc_2_drop.shape)
    # 512 --> 10
    with tf.name_scope('fc_3'):
        fc_3_out = tf.layers.dense(fc_2_drop, 10, activation=None)
    print(fc_3_out.shape)

    return fc_3_out

```

### **训练文件**:

```
#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/9/5 22:12
# @Author  : Seven
# @Site    : 
# @File    : TrainModel.py
# @Software: PyCharm

# TODO：导入环境
from sklearn.model_selection import train_test_split
import tensorflow as tf
import Read_data
import config
import TestModel


def run(logits):
    # 读取数据
    global train_loss
    x_train, y_train, x_test, y_test = Read_data.cifar10_data()
    print(y_train.shape)
    # 构造验证集和训练集
    train_rate = 0.8
    x_train, x_validation, y_train, y_validation = train_test_split(x_train, y_train, train_size=train_rate)

    # 初始化卷积神经网络参数
    epochs = config.epochs
    batch_size = config.batch_size

    # 定义输入和标签的placeholder
    inputs = config.inputs
    targets = config.targets

    # TODO: 计算损失值并初始化optimizer
    learning_rate = config.learning_rate

    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=targets)
    loss_operation = tf.reduce_mean(cross_entropy)

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    training_operation = optimizer.minimize(loss_operation)

    # TODO: 初始化变量
    init = tf.global_variables_initializer()

    print("FUNCTION READY!!")

    # TODO: 保存模型
    saver = tf.train.Saver()

    # TODO: 训练模型

    with tf.Session() as sess:
        sess.run(init)
        num_examples = len(x_train)

        print("Training.....")

        for n in range(epochs):
            for offset in range(0, num_examples, batch_size):
                batch_x, batch_y = x_train[offset:offset+batch_size], y_train[offset:offset+batch_size]
                train_loss, _ = sess.run([loss_operation, training_operation],
                                         feed_dict={inputs: batch_x, targets: batch_y})

            print("EPOCH {} ...".format(n + 1))
            print("Train Loss {:.4f}" .format(train_loss))
            print("Validation Accuracy = {:.3f}".format(TestModel.evaluate(x_validation,
                                                                           y_validation,
                                                                           inputs,
                                                                           logits,
                                                                           targets)))

        saver.save(sess, './model/cifar.model')
        print("Model saved")


```

### **验证模型**：

```
#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/9/6 15:43
# @Author  : Seven
# @Site    : 
# @File    : TestModel.py
# @Software: PyCharm
import tensorflow as tf
import config
import Read_data


def evaluate(X_data, y_data, inputs, logits, targets):
    batch_size = config.batch_size
    # TODO: 模型评估
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(targets, 1))
    accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, batch_size):
        batch_x, batch_y = X_data[offset:offset+batch_size], y_data[offset:offset+batch_size]

        accuracy = sess.run(accuracy_operation, feed_dict={inputs: batch_x, targets: batch_y})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples


def run(inputs, logits, targets):
    print('TESTING....')
    x_train, y_train, x_test, y_test = Read_data.cifar10_data()
    # TODO: 保存模型
    saver = tf.train.Saver()
    with tf.Session() as sess:
        print("Evaluate The Model")
        # TODO: 读取模型
        saver.restore(sess, './model/cifar.model')
        test_accuracy = evaluate(x_test, y_test, inputs, logits, targets)
        print("Test Accuracy = {:.3f}".format(test_accuracy))


```

### **验证本地图片**：

```
import numpy as np
import tensorflow as tf
from PIL import Image
import Network
import config


def main():
    with tf.Session() as sess:
        photo_classes = {0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer',
                         5: 'dog', 6: 'frog', 7: 'horse', 8: 'ship', 9: 'truck'}
        logits = Network.VGG16(config.inputs)
        x = config.inputs
        saver = tf.train.Saver()
        saver.restore(sess, './model/cifar.model')
        # input
        im = Image.open('image/dog-1.jpg')
        # im.show()
        im = im.resize((32, 32))

        # print(im.size, im.mode)

        im = np.array(im).astype(np.float32)
        im = np.reshape(im, [-1, 32*32*3])
        im = (im - (255 / 2.0)) / 255
        batch_xs = np.reshape(im, [-1, 32, 32, 3])

        output = sess.run(logits, feed_dict={x: batch_xs})
        print(output)
        print('the out put is :', photo_classes[np.argmax(output)])


if __name__ == '__main__':
    main()

```

### **输出：**

```
[[-0.4174456  -0.8239337   0.14274524  0.7429316   0.48366657 -0.00303274
   0.22222063 -0.6791349   0.09340217 -0.09914112]]
the out put is : cat
```

