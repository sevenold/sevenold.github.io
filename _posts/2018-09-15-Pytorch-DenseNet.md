---
layout: post
title: "Pytorch实现DenseNet"
date: 2018-09-15
description: "pytorch, cifar10"
tag: Pytorch
---

### 示例代码：

```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/9/14 16:02
# @Author  : Seven
# @Site    : 
# @File    : DenseNet.py
# @Software: PyCharm
import math
import torch
import torch.nn as nn


class Bn_act_conv_drop(nn.Module):
    def __init__(self, inputs, outs, kernel_size, padding):
        super(Bn_act_conv_drop, self).__init__()
        self.bn = nn.Sequential(
            nn.BatchNorm2d(inputs),
            nn.ReLU()
        )
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels=inputs,
                out_channels=outs,
                kernel_size=kernel_size,
                padding=padding,
                stride=1),
            nn.ReLU(),
            nn.Dropout()
        )

    def forward(self, inputs):
        network = self.bn(inputs)
        network = self.conv(network)
        return network


class Transition(nn.Module):
    def __init__(self, inputs, outs):
        super(Transition, self).__init__()
        self.conv = Bn_act_conv_drop(inputs, outs, kernel_size=1, padding=0)
        self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, inputs):
        network = self.conv(inputs)
        network = self.avgpool(network)
        return network


class Block(nn.Module):
    def __init__(self, inputs, growth):
        super(Block, self).__init__()
        self.conv1 = Bn_act_conv_drop(inputs, 4*growth, kernel_size=1, padding=0)
        self.conv2 = Bn_act_conv_drop(4*growth, growth, kernel_size=3, padding=1)

    def forward(self, inputs):
        network = self.conv1(inputs)
        network = self.conv2(network)

        out = torch.cat([network, inputs], 1)
        return out


class DenseNet(nn.Module):
    def __init__(self, blocks, growth):
        super(DenseNet, self).__init__()
        num_planes = 2*growth
        inputs = 3
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels=inputs,
                out_channels=num_planes,
                kernel_size=3,
                # stride=2,
                padding=1),
            nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.block1 = self._block(blocks[0], num_planes, growth)
        num_planes += blocks[0] * growth
        out_planes = int(math.floor(num_planes * 0.5))
        self.tran1 = Transition(inputs=num_planes, outs=out_planes)
        num_planes = out_planes

        self.block2 = self._block(blocks[1], num_planes, growth)
        num_planes += blocks[1] * growth
        out_planes = int(math.floor(num_planes * 0.5))
        self.tran2 = Transition(inputs=num_planes, outs=out_planes)
        num_planes = out_planes

        self.block3 = self._block(blocks[2], num_planes, growth)
        num_planes += blocks[2] * growth
        out_planes = int(math.floor(num_planes * 0.5))
        self.tran3 = Transition(inputs=num_planes, outs=out_planes)
        num_planes = out_planes

        self.block4 = self._block(blocks[3], num_planes, growth)
        num_planes += blocks[3] * growth

        self.bn = nn.Sequential(
            nn.BatchNorm2d(num_planes),
            nn.ReLU()
        )
        self.avgpool = nn.AvgPool2d(kernel_size=4)
        self.linear = nn.Linear(num_planes, 10)

    def forward(self, inputs):
        network = self.conv(inputs)
        network = self.block1(network)
        network = self.tran1(network)

        network = self.block2(network)
        network = self.tran2(network)

        network = self.block3(network)
        network = self.tran3(network)

        network = self.block4(network)
        network = self.bn(network)

        network = self.avgpool(network)
        network = network.view(network.size(0), -1)
        out = self.linear(network)

        return out, network

    @staticmethod
    def _block(layers, inputs, growth):
        block_layer = []
        for layer in range(layers):
            network = Block(inputs, growth)
            block_layer.append(network)
            inputs += growth
        block_layer = nn.Sequential(*block_layer)
        return block_layer


def DenseNet121():
    return DenseNet(blocks=[6, 12, 24, 16], growth=32)


def DenseNet169():
    return DenseNet(blocks=[6, 12, 32, 32], growth=32)


def DenseNet201():
    return DenseNet(blocks=[6, 12, 48, 32], growth=32)


def DenseNet161():
    return DenseNet(blocks=[6, 12, 36, 24], growth=48)


def DenseNet_cifar():
    return DenseNet(blocks=[6, 12, 24, 16], growth=12)


```





