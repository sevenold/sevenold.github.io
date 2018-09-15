---
layout: post
title: "Pytorch实现ResNet"
date: 2018-09-15
description: "pytorch, cifar10"
tag: Pytorch
---

### 示例代码：

```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/9/14 21:57
# @Author  : Seven
# @Site    : 
# @File    : ResNet.py
# @Software: PyCharm
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    growth = 1

    def __init__(self, inputs, outs, stride=1):
        super(BasicBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inputs, outs, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(outs),
            nn.ReLU(),
            nn.Conv2d(outs, outs, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(outs)
        )

        self.shortcut = nn.Sequential()
        if stride != 1 or inputs != outs:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inputs, outs, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outs)
            )

    def forward(self, inputs):
        network = self.left(inputs)
        network += self.shortcut(inputs)
        out = F.relu(network)
        return out


class UpgradeBlock(nn.Module):
    growth = 4

    def __init__(self, inputs, outs, stride=1):
        super(UpgradeBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inputs, outs, kernel_size=1, bias=False),
            nn.BatchNorm2d(outs),
            nn.ReLU(),
            nn.Conv2d(outs, outs, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(outs),
            nn.ReLU(),
            nn.Conv2d(outs, self.growth*outs, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.growth*outs)
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


class ResNet(nn.Module):
    def __init__(self, block, layers):
        super(ResNet, self).__init__()
        self.inputs = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1, stride=1, bias=False),
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.conv2 = self._block(block, layers=layers[0], channels=64, stride=1)
        self.conv3 = self._block(block, layers=layers[1], channels=128, stride=2)
        self.conv4 = self._block(block, layers=layers[2], channels=256, stride=2)
        self.conv5 = self._block(block, layers=layers[3], channels=512, stride=2)

        self.linear = nn.Linear(512*block.growth, 10)

    def forward(self, inputs):
        network = self.conv1(inputs)
        network = self.conv2(network)
        network = self.conv3(network)
        network = self.conv4(network)
        network = self.conv5(network)
        network = F.avg_pool2d(network, kernel_size=network.shape[2])
        network = network.view(network.size(0), -1)
        out = self.linear(network)

        return out, network

    def _block(self, block, layers, channels, stride):
        strides = [stride] + [1] * (layers - 1)  # strides=[1,1]
        layers = []
        for stride in strides:
            layers.append(block(self.inputs, channels, stride))
            self.inputs = channels*block.growth
        return nn.Sequential(*layers)


def ResNet18():
    return ResNet(block=BasicBlock, layers=[2, 2, 2, 2])


def ResNet34():
    return ResNet(block=BasicBlock, layers=[3, 4, 6, 3])


def ResNet50():
    return ResNet(UpgradeBlock, [3, 4, 6, 3])


def ResNet101():
    return ResNet(UpgradeBlock, [3, 4, 23, 3])


def ResNet152():
    return ResNet(UpgradeBlock, [3, 8, 36, 3])

```





