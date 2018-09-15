---
layout: post
title: "Pytorch实现LeNet"
date: 2018-09-15
description: "pytorch, cifar10"
tag: Pytorch
---

### 示例代码：

```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/9/13 20:26
# @Author  : Seven
# @Site    : 
# @File    : LeNet.py
# @Software: PyCharm
import torch.nn as nn

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        # 32*32*3 --28*28*6--> 14*14*6
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3,
                      out_channels=6,
                      kernel_size=5,
                      stride=1,
                      padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        # 14*14*6 --10*10*16--> 5*5*16
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=6,
                      out_channels=16,
                      kernel_size=5,
                      stride=1,
                      padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        # 5*5*16 --> 120
        self.fc1 = nn.Sequential(
            nn.Linear(5 * 5 * 16, 120),
            nn.ReLU(),
            nn.Dropout(p=0.8)
        )

        # 120 --> 84
        self.fc2 = nn.Sequential(
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Dropout(p=0.8)
        )
        # 84 --> 10
        self.out = nn.Linear(84, 10)

    def forward(self, inputs):
        network = self.conv1(inputs)
        network = self.conv2(network)
        network = network.view(network.size(0), -1)
        network = self.fc1(network)
        network = self.fc2(network)
        out = self.out(network)
        return out, network
```





