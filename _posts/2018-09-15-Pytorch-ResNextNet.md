---
layout: post
title: "Pytorch实现ResNextNet"
date: 2018-09-15
description: "pytorch, cifar10"
tag: Pytorch
---

### 示例代码：

```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/9/15 20:48
# @Author  : Seven
# @Site    : 
# @File    : ResnextNet.py
# @Software: PyCharm
import torch
import torch.nn as nn
import torch.nn.functional as F


class Block(nn.Module):
    growth = 2

    def __init__(self, inputs, cardinality=32, block_width=4, stride=1):
        super(Block, self).__init__()
        outs = cardinality*block_width
        self.left = nn.Sequential(
            nn.Conv2d(inputs, outs, kernel_size=1, bias=False),
            nn.BatchNorm2d(outs),
            nn.ReLU(),
            nn.Conv2d(outs, outs, kernel_size=3, stride=stride, padding=1, groups=cardinality, bias=False),
            nn.BatchNorm2d(outs),
            nn.ReLU(),
            nn.Conv2d(outs, self.growth * outs, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.growth * outs)
        )

        self.shortcut = nn.Sequential()
        if stride != 1 or inputs != self.growth * outs:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inputs, self.growth * outs, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.growth * outs)
            )

    def forward(self, inputs):
        network = self.left(inputs)
        network += self.shortcut(inputs)
        out = F.relu(network)
        return out


class ResnextNet(nn.Module):
    def __init__(self, layers, cardinality, block_width):
        super(ResnextNet, self).__init__()
        self.inputs = 64
        self.cardinality = cardinality
        self.block_width = block_width

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1, stride=1, bias=False),
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.conv2 = self._block(layers=layers[0], stride=1)
        self.conv3 = self._block(layers=layers[1], stride=2)
        self.conv4 = self._block(layers=layers[2], stride=2)
        self.conv5 = self._block(layers=layers[3], stride=2)

        self.linear = nn.Linear(8 * cardinality * block_width, 10)

    def forward(self, inputs):
        network = self.conv1(inputs)
        network = self.conv2(network)
        network = self.conv3(network)
        network = self.conv4(network)
        network = self.conv5(network)
        print(network.shape)
        network = F.avg_pool2d(network, kernel_size=network.shape[2]//2)
        print(network.shape)
        network = network.view(network.size(0), -1)
        out = self.linear(network)

        return out, network

    def _block(self, layers, stride):
        strides = [stride] + [1] * (layers - 1)  # strides=[1,1]
        layers = []
        for stride in strides:
            layers.append(Block(self.inputs, self.cardinality, self.block_width, stride))
            self.inputs = self.block_width*self.cardinality*Block.growth
        return nn.Sequential(*layers)


def ResNext50_32x4d():
    return ResnextNet(layers=[3, 4, 6, 3], cardinality=32, block_width=4)


def ResNext50_4x32d():
    return ResnextNet(layers=[3, 4, 6, 3], cardinality=4, block_width=32)


def ResNext50_64x4d():
    return ResnextNet(layers=[3, 4, 6, 3], cardinality=64, block_width=4)


def ResNext101_32x4d():
    return ResnextNet(layers=[3, 4, 23, 3], cardinality=32, block_width=4)


def ResNext101_64x4d():
    return ResnextNet(layers=[3, 4, 23, 3], cardinality=64, block_width=4)

```

