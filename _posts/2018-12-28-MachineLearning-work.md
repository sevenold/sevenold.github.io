---
layout: post
title: "机器学习系列（一）任务类型"
date: 2018-12-28
description: "机器学习任务类型"
tag: 机器学习
---


### 监督学习(Supervised Learning)

> 监督学习：学习到一个**x**->y的映射$f$, 从而对新输入的**x**进行预测$f(x)$

训练数据包含要预测的标签$y$ (标签在训练数据中是`可见变量`)

![监督学习](http://eveseven.oss-cn-shanghai.aliyuncs.com/18-12-28/73828827.jpg)

注意：黑体的小写字母表示向量，如(**x**)， 斜体的小写字母表示标量，如（$y$）, 缺省的向量为列向量。

#### 分类 (Classification)

在监督学习任务中，若输出y为离散值，我们称之为分类，标签空间：$y = {1, 2,3 , ... C}$ 

> 分类：学习从输入**x**到输出$y$的映射$f$:

![分类公式](http://eveseven.oss-cn-shanghai.aliyuncs.com/18-12-28/9435271.jpg)

学习的目标： 训练集上预测值与真是值之间的差异最小

`损失函数`

> 度量模型预测值与真值之间的差异

![损失函数](http://eveseven.oss-cn-shanghai.aliyuncs.com/18-12-28/43953727.jpg)



####  回归 (Regression)

在监督学习任务中，若输出$y \in R$为连续值，则我们称之为一个`回归任务`。

![回归](http://eveseven.oss-cn-shanghai.aliyuncs.com/18-12-28/74009127.jpg)



#### 排序 (Ranking)

### 非监督学习(Unsupervised Learning)

> 非监督学习：发现数据中的“有意义的模式”，也就被称为知识发现。
>
> - 训练数据不包括标签
> - 标签在训练数据中为`隐含变量`

![隐含变量](http://eveseven.oss-cn-shanghai.aliyuncs.com/18-12-28/21386662.jpg)

#### 聚类 (Clustering)

![聚类](http://eveseven.oss-cn-shanghai.aliyuncs.com/18-12-28/22031298.jpg)

####  降维 (Dimensionality Reduction)

> 样本x通常有多维特征，有些特征之间会相关而存在冗余。
>
> -  如图像中相邻像素的值通常相同或差异很小
> - 降维是一种将原高维空间中的数据点映射到低维度空间的技术。其本质是学习一个映射函数$f:x\to x' $，其中x是原始数据点的表达， 是数据点映射后的低维向量表达。
> - 在很多算法中，降维算法为数据预处理的一部分，如主成分分析（ Principal Components Analysis, PCA）

#### 概率密度估计 (density estimation)

### 增强学习(Reinforcement Learning)

>  增强学习：从行为的反馈（奖励或惩罚）中学习
>
> - 设计一个回报函数（reward function），如果learning agent（如机器人、回棋AI程序）在决定一步后，获得了较好的结果，那么我们给agent一些回报（比如回报函数结果为正），得到较差的结果，那么回报函数为负
> - 增强学习的任务：找到一条回报值最大的路径

### 半监督学习(Semi-supervised Learning)

> 根据带标签数据+ 不带标签数据进行学习
>
> - 监督学习+非监督学习的组合
> -  当标注数据“昂贵”时有用
> - 如：标注3D 姿态、蛋白质功能等等

![半监督学习](http://eveseven.oss-cn-shanghai.aliyuncs.com/18-12-28/85449262.jpg)

### 迁移学习（Transfer Learning）

### 其他类型的学习任务

![其他类型的学习任务](http://eveseven.oss-cn-shanghai.aliyuncs.com/18-12-28/54822627.jpg)

转载请注明：[Seven的博客](http://sevenold.github.io)