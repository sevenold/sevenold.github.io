---
layout: post
title: "TensorFlow实现简单的生成对抗网络-GAN"
date: 2018-09-3
description: "对抗生成网络、GAN,tensorflow"
tag: TensorFlow
---

#### **示例代码**：

```
#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/9/1 16:38
# @Author  : Seven
# @Site    : 
# @File    : GAN.py
# @Software: PyCharm

# TODO: 0.导入环境
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data


# TODO: 1：读取数据
mnist = input_data.read_data_sets('data')

# TODO: 2：初始化参数
img_size = mnist.train.images[0].shape[0]
noise_size = 100
g_units = 128
d_units = 128
learning_rate = 0.001
alpha = 0.01

# 真实数据和噪音数据的placeholder
real_img = tf.placeholder(tf.float32, [None, img_size])
noise_img = tf.placeholder(tf.float32, [None, noise_size])


# 显示生成的图像
def view_samples(epoch, samples):
    fig, axes = plt.subplots(figsize=(7, 7), nrows=5, ncols=5, sharey=True, sharex=True)
    for ax, img in zip(axes.flatten(), samples[epoch][1]):  # 这里samples[epoch][1]代表生成的图像结果，而[0]代表对应的logits
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        ax.imshow(img.reshape((28, 28)), cmap='Greys_r')
    plt.show()
    return fig, axes


# TODO: 4.生成器
def get_generator(noise_img, n_units, out_dim, reuse=False, alpha=0.01):
    with tf.variable_scope("generator", reuse=reuse):
        # hidden layer
        hidden1 = tf.layers.dense(noise_img, n_units)
        # leaky ReLU
        hidden1 = tf.maximum(alpha * hidden1, hidden1)
        # dropout
        hidden1 = tf.layers.dropout(hidden1, rate=0.2)

        # logits & outputs
        logits = tf.layers.dense(hidden1, out_dim)
        outputs = tf.tanh(logits)

        return logits, outputs


# 生成器生成数据
g_logits, g_outputs = get_generator(noise_img, g_units, img_size)


# TODO: 5.判别器
def get_discriminator(img, n_units, reuse=False, alpha=0.01):
    with tf.variable_scope("discriminator", reuse=reuse):
        # hidden layer
        hidden1 = tf.layers.dense(img, n_units)
        hidden1 = tf.maximum(alpha * hidden1, hidden1)

        # logits & outputs
        logits = tf.layers.dense(hidden1, 1)
        outputs = tf.sigmoid(logits)

        return logits, outputs


# 判别真实的数据
d_logits_real, d_outputs_real = get_discriminator(real_img, d_units)
# 判别生成器生成的数据
d_logits_fake, d_outputs_fake = get_discriminator(g_outputs, d_units, reuse=True)

# TODO: 6.损失值的计算
# 判别器的损失值
# 识别真实图片的损失值
d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_real,
                                                                     labels=tf.ones_like(d_logits_real)))
# 识别生成的图片的损失值
d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake,
                                                                     labels=tf.zeros_like(d_logits_fake)))
# 总体loss
d_loss = tf.add(d_loss_real, d_loss_fake)
# generator的loss
g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake,
                                                                labels=tf.ones_like(d_logits_fake)))

# TODO:7.始化optimizer
train_vars = tf.trainable_variables()

# generator
g_vars = [var for var in train_vars if var.name.startswith("generator")]
# discriminator
d_vars = [var for var in train_vars if var.name.startswith("discriminator")]

# optimizer
d_train_opt = tf.train.AdamOptimizer(learning_rate).minimize(d_loss, var_list=d_vars)
g_train_opt = tf.train.AdamOptimizer(learning_rate).minimize(g_loss, var_list=g_vars)

print("FUNCTION READY!!!")
print("TRAINING.....")

# TODO:8.开始训练

batch_size = 64
# 训练迭代轮数
epochs = 300
# 抽取样本数
n_sample = 25
# 存储测试样例
samples = []
# 存储loss
losses = []
# 初始化所有变量
init = tf.global_variables_initializer()

show_imgs = []

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(epochs):
        for batch_i in range(mnist.train.num_examples // batch_size):

            batch = mnist.train.next_batch(batch_size)
            batch_images = batch[0].reshape((batch_size, 784))

            # 对图像像素进行缩放，这是因为tanh输出的结果介于(-1,1),real和fake图片共享discriminator的参数
            batch_images = batch_images * 2 - 1

            # generator的输入噪声
            batch_noise = np.random.uniform(-1, 1, size=(batch_size, noise_size))

            # Run optimizers
            sess.run(d_train_opt, feed_dict={real_img: batch_images, noise_img: batch_noise})
            sess.run(g_train_opt, feed_dict={noise_img: batch_noise})

        if (epoch+1) % 30 == 0:
            # 每一轮结束计算loss
            train_loss_d = sess.run(d_loss,
                                    feed_dict={real_img: batch_images,
                                               noise_img: batch_noise})
            # real img loss
            train_loss_d_real = sess.run(d_loss_real,
                                         feed_dict={real_img: batch_images,
                                                    noise_img: batch_noise})
            # fake img loss
            train_loss_d_fake = sess.run(d_loss_fake,
                                         feed_dict={real_img: batch_images,
                                                    noise_img: batch_noise})
            # generator loss
            train_loss_g = sess.run(g_loss,
                                    feed_dict={noise_img: batch_noise})

            print("Epoch {}/{}...\n".format(epoch + 1, epochs),
                  "判别器损失: {:.4f}-->(判别真实的: {:.4f} + 判别生成的: {:.4f})...\n".format(train_loss_d, train_loss_d_real,
                                                                                train_loss_d_fake),
                  "生成器损失: {:.4f}".format(train_loss_g))

            losses.append((train_loss_d, train_loss_d_real, train_loss_d_fake, train_loss_g))
            # 抽取样本后期进行观察
            sample_noise = np.random.uniform(-1, 1, size=(n_sample, noise_size))
            gen_samples = sess.run(get_generator(noise_img, g_units, img_size, reuse=True),
                                   feed_dict={noise_img: sample_noise})
            samples.append(gen_samples)

    # 显示生成的图像
    view_samples(-1, samples)

print("OPTIMIZER END")




```



#### **执行结果**：

```
 判别器损失: 0.8938-->(判别真实的: 0.4657 + 判别生成的: 0.4281)...
 生成器损失: 1.9035
Epoch 60/300...
 判别器损失: 0.9623-->(判别真实的: 0.5577 + 判别生成的: 0.4046)...
 生成器损失: 1.7722
Epoch 90/300...
 判别器损失: 0.9523-->(判别真实的: 0.3698 + 判别生成的: 0.5825)...
 生成器损失: 1.3028
Epoch 120/300...
 判别器损失: 0.8671-->(判别真实的: 0.3948 + 判别生成的: 0.4723)...
 生成器损失: 1.5518
Epoch 150/300...
 判别器损失: 1.0439-->(判别真实的: 0.3626 + 判别生成的: 0.6813)...
 生成器损失: 1.1374
Epoch 180/300...
 判别器损失: 1.3034-->(判别真实的: 0.6210 + 判别生成的: 0.6824)...
 生成器损失: 1.3377
Epoch 210/300...
 判别器损失: 0.8368-->(判别真实的: 0.4397 + 判别生成的: 0.3971)...
 生成器损失: 1.7115
Epoch 240/300...
 判别器损失: 1.0776-->(判别真实的: 0.5503 + 判别生成的: 0.5273)...
 生成器损失: 1.4761
Epoch 270/300...
 判别器损失: 0.9964-->(判别真实的: 0.5351 + 判别生成的: 0.4612)...
 生成器损失: 1.8451
Epoch 300/300...
 判别器损失: 0.9810-->(判别真实的: 0.5085 + 判别生成的: 0.4725)...
 生成器损失: 1.5440
OPTIMIZER END

```

#### 生成的图像：

![images](/images/dl/118.png)