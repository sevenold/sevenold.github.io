---
layout: post
title: "递归神经网络之BackPropagation Through Time"
date: 2018-08-29
description: "卷积神经网络，反向传播，BPTT"
tag: 深度学习
---



### 循环神经网络的训练算法：BPTT

**BackPropagation Through Time (BPTT)**经常出现用于学习递归神经网络（RNN）。

与前馈神经网络相比，RNN的特点是可以处理过去很长的信息。因此特别适合顺序模型。

BPTT扩展了普通的BP算法来适应递归神经网络。

BPTT算法是针对**循环层**的训练算法，它的基本原理和BP算法是一样的，也包含同样的三个步骤：

1. 前向计算每个神经元的输出值；
2. 反向计算每个神经元的**误差项**$\delta_j$值，它是误差函数`E`对神经元`j`的**加权输入**$net_j$的偏导数；
3. 计算每个权重的梯度。

最后再用**随机梯度下降**算法更新权重。

### 前向计算

网络结构：

![images](\images/dl/90.png)

循环层如下图所示：

![images](\images/dl/95.png)

由图可知：

#### $S_t=f(U \cdot X_t+W \cdot S_{t-1}) $

我们假设输入向量`x`的维度是`m`，输出向量`s`的维度是`n`，则矩阵`U`的维度是$n \times m$，矩阵`W`的维度是$n \times n$。接下来我们将上式展开成矩阵：

#### $\begin{align} \begin{bmatrix} s_1^t \\\ s_2^t \\\ .\\\ . \\\ s_n^t \\\ \end{bmatrix}=f( \begin{bmatrix} u_{11} u_{12} ... u_{1m}  \\\ u_{21} u_{22} ... u_{2m} \\\ .\\\ . \\\ u_{n1} u_{n2} ... u_{nm} \\\ \end{bmatrix} \begin{bmatrix} x_1 \\\ x_2 \\\ . \\\ . \\\ x_m \\\ \end{bmatrix}+ \begin{bmatrix} w_{11} w_{12} ... w_{1n} \\\ w_{21} w_{22} ... w_{2n} \\\ . \\\ . \\\  w_{n1} w_{n2} ... w_{nn} \\\ \end{bmatrix} \begin{bmatrix} s_1^{t-1} \\\ s_2^{t-1} \\\ . \\\ . \\\ s_n^{t-1}\\\ \end{bmatrix}) \end{align}$







