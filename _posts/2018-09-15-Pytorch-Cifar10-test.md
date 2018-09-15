---
layout: post
title: "Pytorch实现CIFAR10之读取模型训练本地图片"
date: 2018-09-15
description: "pytorch, cifar10"
tag: Pytorch
---

### 示例代码：

```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/9/14 12:51
# @Author  : Seven
# @Site    : 
# @File    : test.py
# @Software: PyCharm

import torch
import numpy as np
from PIL import Image

# 读取模型
model = torch.load('LeNet.pkl')
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


def local_photos():
    # input
    im = Image.open('plane.jpg')
    # im = im.convert('L')

    im = im.resize((32, 32))
    # im.show()
    im = np.array(im).astype(np.float32)
    im = np.reshape(im, [-1, 32*32*3])
    im = (im - (255 / 2.0)) / 255

    batch_xs = np.reshape(im, [-1, 3, 32, 32])
    batch_xs = torch.FloatTensor(batch_xs)

    # 预测
    pred_y, _ = model(batch_xs)
    pred_y = torch.max(pred_y, 1)[1].data.numpy().squeeze()

    print("The predict is : ", classes[pred_y])


local_photos()
```

### 测试图片：

![images](/images/dl/127.jpg)

### 测试结果：

```
The predict is :  plane
```

