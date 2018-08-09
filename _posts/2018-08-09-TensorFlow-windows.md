---
layout: post
title: "TensorFlow-CPU/GPU安装windows版"
date: 2018-08-09
description: "TensorFlow安装，TensorFlow"
tag: TensorFlow
---



### Windows TensorFlow-CPU安装

环境：window7或以上

python版本要求：`3.5.x`以上

打开**window cmd**，直接使用CPU-only命令安装，如下： 

```
pip3 install --upgrade tensorflow
```

![image](/images/dl/11.png)

然后等待，直至安装成功。

### 验证安装

```
import tensorflow as tf

hello = tf.constant('Hello, TensorFlow!')

sess = tf.Session()

print(sess.run(hello))

```

看到如下的输出，表示安装正确。 

```
Hello, TensorFlow!
```

