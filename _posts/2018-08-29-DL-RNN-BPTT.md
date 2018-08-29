---
layout: post
title: "递归神经网络之BackPropagation Through Time"
date: 2018-08-29
description: "卷积神经网络，反向传播，BPTT"
tag: 深度学习
---

### BackPropagation Through Time

**BackPropagation Through Time (BPTT)**经常出现用于学习递归神经网络（RNN）。与前馈神经网络相比，RNN的特点是可以处理过去很长的信息。因此特别适合顺序模型。BPTT扩展了普通的BP算法来适应递归神经网络。



### 基本定义

