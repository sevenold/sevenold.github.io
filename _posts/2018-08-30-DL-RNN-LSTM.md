---
layout: post
title: "递归神经网络之LSTM"
date: 2018-08-30
description: "卷积神经网络，长短时记忆，LSTM"
tag: 深度学习
---

### LSTM模型

由于RNN也有梯度消失的问题，因此很难处理长序列的数据，大牛们对RNN做了改进，得到了RNN的特例LSTM（Long Short-Term Memory），它可以避免常规RNN的梯度消失，因此在工业界得到了广泛的应用。



### 从RNN到LSTM

在RNN模型中，我们总结了RNN具有如下结构：

![images](/images/dl/97.png)

RNN的模型可以简化成如下图的形式，所有 RNN 都具有一种重复神经网络模块的链式的形式。在标准的 RNN 中，这个重复的模块只有一个非常简单的结构，例如一个 tanh 层。：

![images](/images/dl/98.png)

由于RNN梯度消失的问题，大牛们对于序列索引位置t的隐藏结构做了改进，可以说通过一些技巧让隐藏结构复杂了起来，来避免梯度消失的问题，这样的特殊RNN就是我们的LSTM。由于LSTM有很多的变种，这里我们以最常见的LSTM为例讲述。LSTM的结构如下图：

![images](/images/dl/99.png)

![images](/images/dl/100.png)

### LSTM模型结构剖析

上面我们给出了LSTM的模型结构，下面我们就一点点的剖析LSTM模型在每个序列索引位置t时刻的内部结构。



