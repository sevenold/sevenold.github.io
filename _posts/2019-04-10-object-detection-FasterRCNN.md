---
layout: post
title: "【RCNN系列】Faster RCNN"
date: 2019-06-12
description: "物体检测之Faster RCNN"
tag: 物体检测
---


### 论文原文

链接：[Faster R-CNN](https://arxiv.org/abs/1506.01497)

![](https://eveseven.oss-cn-shanghai.aliyuncs.com/20190415191821.png)

### 摘要

​	最先进的目标检测网络依靠区域提出算法来假设目标的位置。`SPPnet`和`Fast R-CNN`等研究已经减少了这些检测网络的运行时间，使得区域提出计算成为一个瓶颈。在这项工作中，我们引入了一个`区域提出网络（RPN）`，该网络与检测网络共享全图像的卷积特征，从而使近乎零成本的区域提出成为可能。RPN是一个全卷积网络，可以同时在每个位置预测目标边界和目标分数。RPN经过端到端的训练，可以生成高质量的区域提出，由Fast R-CNN用于检测。我们将RPN和Fast R-CNN通过共享卷积特征进一步合并为一个单一的网络——使用最近流行的具有“注意力”机制的神经网络术语，RPN组件告诉统一网络在哪里寻找。对于非常深的VGG-16模型，我们的检测系统在GPU上的帧率为5fps（包括所有步骤），同时在PASCAL VOC 2007，2012和MS COCO数据集上实现了最新的目标检测精度，每个图像只有300个提出。

### 网络结构

- 主干网络：13Conv+13relu+4pooling
- RPN：3x3+背景前景区分+初步定位
- ROIPooling
- 分类+位置精确定位

![](https://eveseven.oss-cn-shanghai.aliyuncs.com/20190415195258.png)

#### 【VGG-Model】

![](https://eveseven.oss-cn-shanghai.aliyuncs.com/20190415202356.png)

#### 【RPN】

​	经典的检测方法生成检测框都非常耗时, 如RCNN使用SS(Selective Search)方法生成检测框。而Faster RCNN则抛弃了传统的滑动窗口和SS方法，直接使用RPN生成检测框，这也是Faster RCNN的巨大优势，能极大提升检测框的生成速度。

![](https://eveseven.oss-cn-shanghai.aliyuncs.com/20190415202436.png)

上图展示了RPN网络的具体结构。可以看到RPN网络实际分为2条线

- 粗分类：上面一条通过softmax分类anchors获得foreground和background（检测目标是foreground）
- 粗定位：下面一条用于计算对于anchors的bounding box regression偏移量，以获得粗略的proposal。

而最后的Proposal层则负责综合foreground anchors和bounding box regression偏移量获取proposals，同时剔除太小和超出边界的proposals。其实整个网络到了Proposal Layer这里，就完成了相当于目标定位的功能。

#### 【anchors】

​	首先使用一个(3, 3)的sliding window在feature map上滑动(就是做卷积，卷积的本质就是卷积核滑动求和)，然后取(3, 3)滑窗的中心点，将这个点对应到原图上，然后取三种不同的尺寸(128, 256, 512)和三种比例(1:1, 2:1, 1:2)在原图上产生9个anchor，所以feature map上的每个点都会在原图上对应点的位置生成9个anchor，而我们从上边的分析已经得知feature map的大小为输入图片的1/16倍，所以最终一共有 $\frac m {16} \times  \frac n {16} \times 9$ 个anchor。

![](https://eveseven.oss-cn-shanghai.aliyuncs.com/20190612191124.png)

### 训练流程

和之前的目标检测网络训练类似，都是在一个预训练网络的基础上进行训练的。一般使用4-step交替训练法来训练Faster R-CNN，主要步骤如下：

1. 在ImageNet预训练的网络来训练RPN网络。
2. 使用ImageNet预训练的网络来训练一个单独Fast R-CNN检测网络，网络中的proposal是第一步训练RPN网络中得到的porposal。
3. 使用上一步训练的detection网络的参数来初始化练RPN网络，并且在训练过程中固定shared conv layers这部分，只更新RPN网络专有的那些层。
4. 固定shared conv layers这部分，只更新Fast R-CNN专有的那些层，其中proposal也是上一步训练RPN网络中得到的proposal。

### 总结

| **使用方法** | **缺点**                                                     | **改进**                                                     |                                                              |
| ------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| R-CNN        | 1、SS提取RP；2、CNN提取特征；3、SVM分类；4、BB盒回归。       | 1、 训练步骤繁琐（微调网络+训练SVM+训练bbox）；2、 训练、测试均速度慢 ；3、 训练占空间 | 1、 从DPM HSC的34.3%直接提升到了66%（mAP）；2、 引入RP+CNN   |
| Fast R-CNN   | 1、SS提取RP；2、CNN提取特征；3、softmax分类；4、多任务损失函数边框回归。 | 1、 依旧用SS提取RP(耗时2-3s，特征提取耗时0.32s)；2、 无法满足实时应用，没有真正实现端到端训练测试；3、 利用了GPU，但是区域建议方法是在CPU上实现的。 | 1、 由66.9%提升到70%；2、 每张图像耗时约为3s。               |
| Faster R-CNN | 1、RPN提取RP；2、CNN提取特征；3、softmax分类；4、多任务损失函数边框回归。 | 1、 还是无法达到实时检测目标；2、 获取region proposal，再对每个proposal分类计算量还是比较大。 | 1、 提高了检测精度和速度；2、  真正实现端到端的目标检测框架；3、  生成建议框仅需约10ms。 |