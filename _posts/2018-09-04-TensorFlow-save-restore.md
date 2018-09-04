---
layout: post
title: "TensorFlow--保存神经网络参数和加载神经网络参数"
date: 2018-09-4
description: "保存神经网络参数和加载神经网络参数、模型参数保存加载"
tag: TensorFlow
---

### 保存神经网络参数

#### **代码**：

```
#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/9/4 16:11
# @Author  : Seven
# @Site    : 
# @File    : save.py
# @Software: PyCharm
import tensorflow as tf


# 保存神经网络参数
def save_para():
    # 定义权重参数
    W = tf.Variable([[1, 2, 3], [4, 5, 6]], dtype = tf.float32, name = 'weights')
    # 定义偏置参数
    b = tf.Variable([[1, 2, 3]], dtype = tf.float32, name = 'biases')
    # 参数初始化
    init = tf.global_variables_initializer()
    # 定义保存参数的saver
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(init)
        # 保存session中的数据
        save_path = saver.save(sess, './save_net.ckpt')
        # 输出保存路径
        print('Save to path: ', save_path)


save_para()
```

#### **执行结果**：

```
Save to path:  ./save_net.ckpt
```



### 加载神经网络参数

#### **代码**：

```
#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/9/4 16:14
# @Author  : Seven
# @Site    : 
# @File    : restore.py
# @Software: PyCharm
import tensorflow as tf
import numpy as np


# 恢复神经网络参数
def restore_para():
    # 定义权重参数
    W = tf.Variable(np.arange(6).reshape((2, 3)), dtype = tf.float32, name = 'weights')
    # 定义偏置参数
    b = tf.Variable(np.arange(3).reshape((1, 3)), dtype = tf.float32, name = 'biases')
    # 定义提取参数的saver
    saver = tf.train.Saver()

    with tf.Session() as sess:
        # 加载文件中的参数数据，会根据name加载数据并保存到变量W和b中
        save_path = saver.restore(sess, 'save_net.ckpt')
        # 输出保存路径
        print('Weights: ', sess.run(W))
        print('biases:  ', sess.run(b))


restore_para()
```

#### **执行结果**：

```
Weights:  [[1. 2. 3.]
 [4. 5. 6.]]
biases:   [[1. 2. 3.]]
```

