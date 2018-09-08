---
layout: post
title: "TensorFlow-cifar10-图像分类之网络结构"
date: 2018-09-7
description: "TensorFlow, cifar10"
tag: TensorFlow
---

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