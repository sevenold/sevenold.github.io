---
layout: post
title: "TensorFlow实现简单的卷积神经网络-CNN"
date: 2018-08-23
description: "python实现简单的卷积神经网络"
tag: TensorFlow
---



### 环境设定

```
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
```

### 数据准备

```
# 使用tensorflow自带的工具加载MNIST手写数字集合
mnist = input_data.read_data_sets('data/mnist', one_hot=True)

# 查看数据的维度和target的维度
print(mnist.train.images.shape)
print(mnist.train.labels.shape)
```

### 准备好placeholder

```
X = tf.placeholder(tf.float32, [None, 784])
Y = tf.placeholder(tf.float32, [None, 10])
learnRate = tf.placeholder(tf.float32)
```

### 构建网络计算图结构

```
# 把输入数据reshape--28x28=784, 单通道， -1表示None
with tf.name_scope('reshape'):
    x_image = tf.reshape(x, [-1, 28, 28, 1])

# 构建第一层卷积计算层--将一个灰度图像映射到32个feature maps, 卷积核为5x5
with tf.name_scope('conv1'):
    shape = [5, 5, 1, 32]
    W_conv1 = tf.Variable(tf.truncated_normal(shape=shape, stddev=0.1),
                          collections=[tf.GraphKeys.GLOBAL_VARIABLES, 'WEIGHTS'])

    shape = [32]
    b_conv1 = tf.Variable(tf.constant(0.1, shape=shape))

    net_conv1 = tf.nn.conv2d(x_image, W_conv1, strides=[1, 1, 1, 1],
                             padding='SAME') + b_conv1

    out_conv1 = tf.nn.relu(net_conv1)

# 构建池化层--采用最大池化
with tf.name_scope('pool1'):
    h_pool1 = tf.nn.max_pool(out_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                             padding='VALID')

# 构建第二层卷积计算层--maps 32 feature maps to 64.
with tf.name_scope('conv2'):
    shape = [5, 5, 32, 64]
    W_conv2 = tf.Variable(tf.truncated_normal(shape=shape, stddev=0.1),
                          collections=[tf.GraphKeys.GLOBAL_VARIABLES, 'WEIGHTS'])

    shape = [64]
    b_conv2 = tf.Variable(tf.constant(0.1, shape=shape))

    net_conv2 = tf.nn.conv2d(h_pool1, W_conv2, strides=[1, 1, 1, 1],
                             padding='SAME') + b_conv2

    out_conv2 = tf.nn.relu(net_conv2)


# 构建第二层池化层--采用最大池化
with tf.name_scope('pool2'):
    h_pool2 = tf.nn.max_pool(out_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                             padding='VALID')

# 构建全连接层--经过的两层的下采样（池化），28x28x1的图像-->7x7x64，然后映射到1024个特征
with tf.name_scope('fc1'):
    shape = [7*7*64, 1024]
    W_fc1 = tf.Variable(tf.truncated_normal(shape=shape, stddev=0.1),
                        collections=[tf.GraphKeys.GLOBAL_VARIABLES, 'WEIGHTS'])

    shape = [1024]
    b_fc1 = tf.Variable(tf.constant(0.1, shape=shape))

    shape = [-1, 7*7*64]
    h_pool2_flat = tf.reshape(h_pool2, shape)

    out_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# Dropout--防止过拟合
with tf.name_scope('dropout'):
    keep_prob = tf.placeholder(tf.float32)
    out_fc1_drop = tf.nn.dropout(out_fc1, keep_prob=keep_prob)


# 构建第二层全连接层--将1024个特性映射到10个类，每个类对应一个数字
with tf.name_scope('fc2'):
    shape = [1024, 10]
    W_fc2 = tf.Variable(tf.truncated_normal(shape=shape, stddev=0.1),
                        collections=[tf.GraphKeys.GLOBAL_VARIABLES, 'WEIGHTS'])

    shape = [10]
    b_fc2 = tf.Variable(tf.constant(0.1, shape=shape))

    out = tf.matmul(out_fc1_drop, W_fc2) + b_fc2
```

### 计算损失函数并初始化optimizer

```
print(y.shape, out.shape)
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=out))

l2_loss = tf.add_n([tf.nn.l2_loss(w) for w in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)])

total_loss = cross_entropy + 7e-5*l2_loss

train_step = tf.train.AdamOptimizer(learnRate).minimize(total_loss)
print("FUNCTIONS READY!!")
```

### 初始化变量

```
init = tf.global_variables_initializer()
```

### 在session中执行graph定义的运算

```
with tf.Session() as sess:
    sess.run(init)

    for step in range(3000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        lr = 0.01

        _, loss, l2_loss_value, total_loss_value = sess.run(
            [train_step, cross_entropy, l2_loss, total_loss],
            feed_dict={x: batch_xs, y: batch_ys, learnRate: lr, keep_prob: 0.5})

        if (step + 1) % 100 == 0:
            print("step %d, entropy loss: %f, l2_loss: %f, total loss: %f" %
                  (step + 1, loss, l2_loss_value, total_loss_value))

            # 验证训练的模型
            correct_prediction = tf.equal(tf.argmax(out, 1), tf.argmax(y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            print("Train accuracy:", sess.run(accuracy, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 0.5}))

        if (step + 1) % 1000 == 0:
            print("Text accuracy:", sess.run(accuracy, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 0.5}))
```

### 运行结果

```
step 2100, entropy loss: 0.127328, l2_loss: 1764.112305, total loss: 0.250816
Train accuracy: 0.99
step 2200, entropy loss: 0.275415, l2_loss: 1819.110107, total loss: 0.402753
Train accuracy: 0.97
step 2300, entropy loss: 0.147261, l2_loss: 1773.890869, total loss: 0.271434
Train accuracy: 0.99
step 2400, entropy loss: 0.100819, l2_loss: 1699.657349, total loss: 0.219795
Train accuracy: 0.97
step 2500, entropy loss: 0.148775, l2_loss: 1701.602661, total loss: 0.267887
Train accuracy: 0.96
step 2600, entropy loss: 0.142768, l2_loss: 1800.301880, total loss: 0.268789
Train accuracy: 0.96
step 2700, entropy loss: 0.134457, l2_loss: 2041.165161, total loss: 0.277339
Train accuracy: 0.96
step 2800, entropy loss: 0.071112, l2_loss: 2003.971924, total loss: 0.211390
Train accuracy: 0.99
step 2900, entropy loss: 0.217092, l2_loss: 1879.004272, total loss: 0.348622
Train accuracy: 0.98
step 3000, entropy loss: 0.207697, l2_loss: 1841.294556, total loss: 0.336588
Train accuracy: 0.97
Text accuracy: 0.95
Optimization Finished
```

#### 结论：简单写了个卷积神经网络，没有进行参数调试。