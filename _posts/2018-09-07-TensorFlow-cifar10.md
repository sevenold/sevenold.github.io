---
layout: post
title: "TensorFlow-cifar10-图像分类之数据预处理及配置"
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

