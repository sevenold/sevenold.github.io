---
layout: post
title: "【九】直线检测"
date: 2019-03-22
description: "视觉特征提取-直线检测"
tag: 机器视觉
---

### Hough变换

#### 【简介】

> Hough变换是图像处理中从图像中识别几何形状的基本方法之一。Hough变换的基本原理在于利用点与线的对偶性，将原始图像空间的给定的曲线通过曲线表达形式变为参数空间的一个点。这样就把原始图像中给定曲线的检测问题转化为寻找参数空间中的峰值问题。也即把检测整体特性转化为检测局部特性。比如直线、椭圆、圆、弧线等。

#### 【**原理**】

> - 采用参数空间变换的方法，对噪声和不间断直线的检测具有鲁棒性。
> - 可用于检测圆和其他参数形状。
> - 核心思想：直线$y = kx +b $, 每一条直线对应一个k,b，极坐标下对应一个点($\rho, \theta$).

![hough](https://eveseven.oss-cn-shanghai.aliyuncs.com/20190118142323.png)

#### 【**直线**】

> - 直角坐标系的一点$(x, y)$，对应极坐标系下的一条正弦曲线$\rho= x cos\theta+y\sin\theta$
> - 同一条直线上的多个点，在极坐标下必相交于一点。

![直线](https://eveseven.oss-cn-shanghai.aliyuncs.com/20190118143525.png)

#### 【**步骤**】

> - 将($\rho, \theta$)空间量化为许多小格。
> - 根据$x-y$平面每一个直线点代入$\theta$的量化值，算出各个$\rho$，将对应格计数累加。
> - 当全部点变换后，对小格进行检验。设置累计阈值T，计数器大于T的小格对应于共线点，其可以用作直线拟合参数。小于T的反映非共线点，丢弃不用。
>
> 注意：$\theta$的取值为$(0,2\pi)$.

#### 【**总结**】

> 1. Hough变换的核心思想基于参数空间下的直线模型及投票原理。
> 2. Hough变换通过特征值判断灰度是否发生突变。
> 3. Hough变换想法直接，易于实现，是其它角点提取算法的基础。

### 代码实例

```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/1/18 14:39
# @Author  : Seven
# @File    : HoughDemo.py
# @Software: PyCharm
import cv2
import numpy as np

img = cv2.imread('sudoku.png')
edgeImage = cv2.Canny(img, 50, 200, apertureSize=3)
grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def basicHough(edgeImage):
    """
    HoughLines(image, rho, theta, threshold, lines=None, srn=None, stn=None, min_theta=None, max_theta=None)
    image:边缘检测的输出图像. 它应该是个灰度图 (但事实上是个二值化图)

    lines:储存着检测到的直线的参数对  的容器 

    rho:参数极径  以像素值为单位的分辨率. 我们使用 1 像素.

    theta:参数极角  以弧度为单位的分辨率. 我们使用 1度 (即CV_PI/180)

    theta:要”检测” 一条直线所需最少的的曲线交点

    srn and stn: 参数默认为0.

    :param img:
    :return:
    """
    cv2.imshow('edge', edgeImage)
    lines = cv2.HoughLines(edgeImage, 1, np.pi / 180, 150)
    for rho, theta in lines[:, 0, :]:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * a)
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * a)
        cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

    cv2.imshow("hough", img)


def uphough(edgeImage):
    """
    HoughLinesP(image, rho, theta, threshold, lines=None, minLineLength=None, maxLineGap=None)
    image： 必须是二值图像，推荐使用canny边缘检测的结果图像； 
    rho: 线段以像素为单位的距离精度，double类型的，推荐用1.0 
    theta： 线段以弧度为单位的角度精度，推荐用numpy.pi/180 
    threshod: 累加平面的阈值参数，int类型，超过设定阈值才被检测出线段，值越大，基本上意味着检出的线段越长，检出的线段个数越少。根据情况推荐先用100试试
    lines：这个参数的意义未知，发现不同的lines对结果没影响，但是不要忽略了它的存在 
    minLineLength：线段以像素为单位的最小长度，根据应用场景设置 
    maxLineGap：同一方向上两条线段判定为一条线段的最大允许间隔（断裂），超过了设定值，则把两条线段当成一条线段，值越大，允许线段上的断裂越大，越有可能检出潜在的直线段
    :param img:
    :return:
    """
    cv2.imshow('ep', edgeImage)
    linesP = cv2.HoughLinesP(edgeImage, 1, np.pi/180, threshold=100, maxLineGap=100, minLineLength=10)
    for x1, x2, y1, y2 in linesP[:, 0, :]:
        cv2.line(img, (x1, x2), (y1, y2), (0, 0, 255), 2, cv2.LINE_AA)

    cv2.imshow('houghP', img)

basicHough(edgeImage)
uphough(edgeImage)
cv2.waitKey(0)

```

![](https://eveseven.oss-cn-shanghai.aliyuncs.com/20190322214512.png)

转载请注明：[Seven的博客](http://sevenold.github.io)