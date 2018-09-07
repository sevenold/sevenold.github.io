---
layout: post
title: "TensorFlow数据可视化"
date: 2018-09-7
description: "TensorFlow, 数据可视化"
tag: TensorFlow
---

#### **代码**：

```
#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/9/7 16:59
# @Author  : Seven
# @Site    :
# @File    : demo.py
# @Software: PyCharm

import numpy as np
import tensorflow as tf

n_observations = 100
xs = np.linspace(-3, 3, n_observations)
ys = 0.8*xs + 0.1 + np.random.uniform(-0.5, 0.5, n_observations)

X = tf.placeholder(tf.float32, name='X')
Y = tf.placeholder(tf.float32, name='Y')

W = tf.Variable(tf.random_normal([1]), name='weight')
tf.summary.histogram('weight', W)
b = tf.Variable(tf.random_normal([1]), name='bias')
tf.summary.histogram('bias', b)


Y_pred = tf.add(tf.multiply(X, W), b)

loss = tf.square(Y - Y_pred, name='loss')
tf.summary.scalar('loss', tf.reshape(loss, []))

learning_rate = 0.01
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

n_samples = xs.shape[0]
init = tf.global_variables_initializer()
with tf.Session() as sess:
    # 记得初始化所有变量
    sess.run(init)
    merged = tf.summary.merge_all()
    log_writer = tf.summary.FileWriter("./logs/linear_regression", sess.graph)

    # 训练模型
    for i in range(50):
        total_loss = 0
        for x, y in zip(xs, ys):
            # 通过feed_dic把数据灌进去
            _, loss_value, merged_summary = sess.run([optimizer, loss, merged], feed_dict={X: x, Y: y})
            total_loss += loss_value

        if i % 5 == 0:
            print('Epoch {0}: {1}'.format(i, total_loss / n_samples))
            log_writer.add_summary(merged_summary, i)

    # 关闭writer
    log_writer.close()

    # 取出w和b的值
    W, b = sess.run([W, b])

print(W, b)
print("W:"+str(W[0]))
print("b:"+str(b[0]))
```

#### **执行结果**：

```
Epoch 0: [0.5815637]
Epoch 5: [0.08926834]
Epoch 10: [0.08926827]
Epoch 15: [0.08926827]
Epoch 20: [0.08926827]
Epoch 25: [0.08926827]
Epoch 30: [0.08926827]
Epoch 35: [0.08926827]
Epoch 40: [0.08926827]
Epoch 45: [0.08926827]
[0.7907032] [0.10920969]
W:0.7907032
b:0.10920969
```

#### **Tensoboard**：

#### 在终端执行代码：

```
tensorboard --logdir log (你保存文件所在位置)
```

#### 输出：

```
TensorBoard 0.4.0 at http://seven:6006 (Press CTRL+C to quit)
```

然后打开网页：`http://seven:6006`。

#### 显示结果：

![image](/images/dl/123.png)

![image](/images/dl/124.png)

![image](/images/dl/125.png)

![image](/images/dl/126.png)

