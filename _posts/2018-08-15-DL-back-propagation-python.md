---
layout: post
title: "神经网络入门-反向传播python实现"
date: 2018-08-15
description: "神经网络基础知识点，反向传播python实现、正则项、softmax、交叉熵"
tag: 深度学习
---



### 反向传播（back propagation）算法`python`实现

#### 根据上一章所推导的[反向传播（back propagation）算法](https://sevenold.github.io/2018/08/DL-back-propagation/)所示例代码

```
#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/8/15 21:32
# @Author  : Seven
# @Site    : 
# @File    : bp.py
# @Software: PyCharm


import numpy as np


def sigmod(x):
    return 1 / (1 + np.exp(-x))


def sigmodDerivative(x):

    return np.multiply(x, np.subtract(1, x))


def forward(weightsA, weightsB, bias):
    # 前向传播
    # 隐层
    neth1 = inputX[0] * weightsA[0][0] + inputX[1] * weightsA[0][1] + bias[0]
    outh1 = sigmod(neth1)
    print("隐层第一个神经元", neth1, outh1)

    neth2 = inputX[0] * weightsA[1][0] + inputX[1] * weightsA[1][1] + bias[1]
    outh2 = sigmod(neth2)
    print("隐层第二个神经元", neth2, outh2)
    # 输出层
    neto1 = outh1 * weightsB[0][0] + outh2 * weightsB[0][1] + bias[2]
    outo1 = sigmod(neto1)
    print("输出层第一个神经元", neto1, outo1)

    neto2 = outh1 * weightsB[1][0] + outh2 * weightsB[1][1] + bias[3]
    outo2 = sigmod(neto2)
    print("输出层第二个神经元", neto2, outo2)

    # 向量化
    outA = np.array([outh1, outh2])
    outB = np.array([outo1, outo2])

    Etotal = 1 / 2 * np.subtract(y, outB) ** 2
    print("误差值：", Etotal)

    return outA, outB


def backpagration(outA, outB):
    # 反向传播
    deltaB = np.multiply(np.subtract(outB, y), sigmodDerivative(outB))
    print("deltaB：", deltaB)

    deltaA = np.multiply(np.matmul(np.transpose(weightsB), deltaB), sigmodDerivative(outA))
    print("deltaA：", deltaA)

    deltaWB = np.matmul(deltaB.reshape(2, 1), outA.reshape(1, 2))
    print("deltaWB：", deltaWB)

    deltaWA = np.matmul(deltaA.reshape(2, 1), inputX.reshape(1, 2))
    print("deltaWA", deltaWA)

    # 权重参数更新
    weightsB_new = np.subtract(weightsB, deltaWB)
    print("weightsB_new", weightsB_new)

    bias[3] = np.subtract(bias[3], deltaB[1])
    print("biasB", bias[3])
    bias[2] = np.subtract(bias[2], deltaB[0])
    print("biasB", bias[2])

    weightsA_new = np.subtract(weightsA, deltaWA)
    print("weightsA_new", weightsA_new)

    bias[1] = np.subtract(bias[1], deltaA[1])
    print("biasA", bias[1])
    bias[0] = np.subtract(bias[0], deltaA[0])
    print("biasA", bias[0])
    print("all bias", bias)

    return weightsA_new, weightsB_new, bias


if __name__=="__main__":
    # 初始化数据
    # 权重参数
    bias = np.array([0.5, 0.5, 1.0, 1.0])
    weightsA = np.array([[0.1, 0.3], [0.2, 0.4]])
    weightsB = np.array([[0.6, 0.8], [0.7, 0.9]])
    # 期望值
    y = np.array([1, 0])
    # 输入层
    inputX = np.array([0.5, 1.0])

    print("第一次前向传播")
    outA, outB = forward(weightsA, weightsB, bias)
    print("反向传播-参数更新")
    weightsA_new, weightsB_new, bias = backpagration(outA, outB)
    # 更新完毕
    # 验证权重参数--第二次前向传播
    print("第二次前向传播")
    forward(weightsA_new, weightsB_new, bias)

```

#### 程序执行结果：

```
第一次前向传播
隐层第一个神经元 0.85 0.7005671424739729
隐层第二个神经元 1.0 0.7310585786300049
输出层第一个神经元 2.0051871483883876 0.8813406204337122
输出层第二个神经元 2.1483497204987856 0.8955144637754225
误差值： [0.00704002 0.40097308]
反向传播-参数更新
deltaB： [-0.01240932  0.08379177]
deltaA： [0.01074218 0.01287516]
deltaWB： [[-0.00869356 -0.00907194]
 [ 0.05870176  0.0612567 ]]
deltaWA [[0.00537109 0.01074218]
 [0.00643758 0.01287516]]
weightsB_new [[0.60869356 0.80907194]
 [0.64129824 0.8387433 ]]
biasB 0.9162082259892468
biasB 1.0124093185565075
weightsA_new [[0.09462891 0.28925782]
 [0.19356242 0.38712484]]
biasA 0.48712483967909465
biasA 0.48925781687016445
all bias [0.48925782 0.48712484 1.01240932 0.91620823]
第二次前向传播
隐层第一个神经元 0.8258300879578699 0.6954725026123048
隐层第二个神经元 0.971030889277963 0.7253249281967498
输出层第一个神经元 2.022578998544453 0.8831474198595102
输出层第二个神经元 1.970574942645405 0.8776728542726611
误差值： [0.00682726 0.38515482]
```

#### 我们可以看出误差值明显下降了，说明我们的权重参数更新是正确的。