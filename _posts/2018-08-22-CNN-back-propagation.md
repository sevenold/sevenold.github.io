---
layout: post
title: "神经网络之卷积神经网络反向传播算法"
date: 2018-08-22
description: "卷积神经网络的反向传播算法推导，前向传播"
tag: 深度学习
---



### 回顾DNN的反向传播算法

需要在理解深度神经网络的反向传播算来做预先理解，见[神经网络入门-反向传播](https://sevenold.github.io/2018/08/DL-back-propagation/)所总结的反向传播四部曲：

#### **反向传播的四项基本原则**：

#### 基本形式:

- #### $\delta_i^{(L)}= \bigtriangledown_{out} E_{total} \times   \sigma^{\prime}(net_i^{(L)}) $

- #### $\delta_i^{(l)} = \sum_j \delta_j^{(l+1)} w_{ji}^{(l+1)}  \sigma^{\prime}(net_i^{(l)})$

- #### $\frac{\partial E_{total}}{\partial bias_i^{(l)}}=\delta_i^{(l)}$

- #### $\frac{\partial E_{total}}{\partial w_{ij}^{(l)}}=outh_j^{(l-1)}\delta_i^{(l)}$

#### 矩阵形式：

- #### $\delta_i^{(L)}= \bigtriangledown_{out} E_{total} \bigodot   \sigma^{\prime}(net_i^{(L)}) $ ， $\bigodot$是Hadamard乘积（对应位置相乘）

- #### $\delta^{(l)} = (w^{(l+1)})^T \delta^{(l+1)} \bigodot  \sigma^{\prime}(net^{(l)})$

- #### $\frac{\partial E_{total}}{\partial bias^{(l)}}=\delta^{(l)}$

- #### $\frac{\partial E_{total}}{\partial w^{(l)}}=\delta^{(l)}(outh^{(l-1)})^T$

现在我们想把同样的思想用到卷积神经网络中，但是很明显，CNN有很多不同的地方（卷积计算层，池化层），不能直接去套用DNN的反向传播公式。接下来，我们就从卷积神经网络的前向传播算法到反向传播算法进行推导总结。



### 卷积神经网络的前向传播算法

我们上一节已经总结了卷积神经网络的网络结构：`数据输入层`、`卷积计算层`、`激活层`、`池化层`、`全连接层`，我们现在来根据这个网络结构进行前向传播。



