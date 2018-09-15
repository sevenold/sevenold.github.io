---
layout: post
title: "Pytorch实现CIFAR-10之数据预处理"
date: 2018-09-15
description: "pytorch, cifar10"
tag: Pytorch
---

### 示例代码：

```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/9/13 19:45
# @Author  : Seven
# @Site    : 
# @File    : train.py
# @Software: PyCharm

import torch
import os
import torchvision
import torchvision.transforms as transforms
import torch.utils.data as Data
import torch.nn as nn
import torch.optim as optim

print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])


DOWNLOAD_MNIST = False

if not(os.path.exists('./data/')) or not os.listdir('./data/'):  # 判断数据是否存在
    DOWNLOAD_MNIST = True

trainset = torchvision.datasets.CIFAR10(root='./data',
                                        train=True,
                                        download=DOWNLOAD_MNIST,
                                        transform=transform_train)

trainloader = torch.utils.data.DataLoader(trainset,
                                          batch_size=128,
                                          shuffle=True,
                                          num_workers=0)

testset = torchvision.datasets.CIFAR10(root='./data',
                                       train=False,
                                       download=DOWNLOAD_MNIST,
                                       transform=transform_test)

testloader = torch.utils.data.DataLoader(testset,
                                         batch_size=100,
                                         shuffle=False,
                                         num_workers=0)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

```

