---
layout: post
title: "【十】Harris角点检测"
date: 2019-03-22
description: "视觉特征提取-Harris角点检测"
tag: 机器视觉
---

### Harris角点检测

#### 【原理】

> 人眼对角点的识别通常是在一个局部的小区域或小窗口完成的。如果在各个方向上移动这个特征的小窗口，窗口内区域的灰度发生了较大的变化，那么就认为在窗口内遇到了角点。如果这个特定的窗口在图像各个方向上移动时，窗口内图像的灰度没有发生变化，那么窗口内就不存在角点；如果窗口在某一个方向移动时，窗口内图像的灰度发生了较大的变化，而在另一些方向上没有发生变化，那么，窗口内的图像可能就是一条直线的线段。

> - 在灰度变化平缓区域，窗口内像素灰度积分近似保持不变
> - 在 边缘区域，边缘方向：灰度积分近似不变， 其余任意方向：剧烈变化。
> - 在角点出：任意方向剧烈变化。

![原理](https://eveseven.oss-cn-shanghai.aliyuncs.com/20190118163820.png)

#### 【**数学推导**】

- 定义灰度积分变化

  ![灰度积分](https://eveseven.oss-cn-shanghai.aliyuncs.com/20190118183652.png)

- 定义灰度积分变化

  ![2](https://eveseven.oss-cn-shanghai.aliyuncs.com/20190118183813.png)

- 如果$u、v$很小， 则有：

  ![2](https://eveseven.oss-cn-shanghai.aliyuncs.com/20190118183858.png)

- 其中

  ![2](https://eveseven.oss-cn-shanghai.aliyuncs.com/20190118183922.png)

- 注意$E$是一个二次型，即：

  ![3](https://eveseven.oss-cn-shanghai.aliyuncs.com/20190118184508.png)

- $\lambda_1^{-\frac12}、\lambda_2^{-\frac12}$是椭圆的长短轴

  - 当$\lambda_1、\lambda_2$都比较小时，点(x, y)处于灰度变化平缓区域.
  - 当$\lambda_1>\lambda_2$或者$\lambda_1<\lambda_2$时，点(x, y)为边界元素.
  - 当$\lambda_1、\lambda_2$都比较大时，且近似相等，点(x, y)为角点.

  ![3](https://eveseven.oss-cn-shanghai.aliyuncs.com/20190118184902.png)

- 角点响应函数

  ![4](https://eveseven.oss-cn-shanghai.aliyuncs.com/20190118185002.png)

- 当R接近于零时，处于灰度变化平缓区域

- 当R<0时，点为边界像素

- 当R>0时，点为角点



### 代码实例

```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/1/18 18:51
# @Author  : Seven
# @File    : HarrisDemo.py
# @Software: PyCharm
import cv2
import numpy as np

img = cv2.imread('chess.png')
img = cv2.pyrDown(img)
cv2.imshow('source', img)


def harrisDemo(img):
    """
    cornerHarris(src, blockSize, ksize, k, dst=None, borderType=None)
    src，输入图像，即源图像，填Mat类的对象即可，且需为单通道8位或者浮点型图像。
    dst，函数调用后的运算结果存在这里，即这个参数用于存放Harris角点检测的输出结果，和源图片有一样的尺寸和类型。
    blockSize，表示邻域的大小，更多的详细信息在cornerEigenValsAndVecs中有讲到。
    ksize，表示Sobel()算子的孔径大小。
    k，Harris 角点检测方程中的自由参数，取值参数为[0,04，0.06]
    borderType，图像像素的边界模式，注意它有默认值BORDER_DEFAULT。更详细的解释，参考borderInterpolate函数。
    :param img:
    :return:
    """
    grayImage = np.float32(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    # 输入图像必须是float32，最后一个参数在0.04 到0.05 之间
    # harris角点检测
    harrisImg = cv2.cornerHarris(grayImage, 2, 3, 0.04)
    # 膨胀
    dst = cv2.dilate(harrisImg, None)
    # 0.01是人为设定的阈值
    # 把角点设为红色
    img[dst > 0.01 * dst.max()] = [0, 0, 255]
    cv2.imshow('harris', img)


def harrisSubPix(img):
    """
    cornerSubPix(image, corners, winSize, zeroZone, criteria)
    image：输入图像

    corners：输入角点的初始坐标以及精准化后的坐标用于输出。

    winSize：搜索窗口边长的一半，例如如果winSize=Size(5,5)，则一个大小为(5*2+1)*(5*2+1)=11*11的搜索窗口将被使用。

    zeroZone：搜索区域中间的dead region边长的一半，有时用于避免自相关矩阵的奇异性。如果值设为(-1,-1)则表示没有这个区域。
    criteria：角点精准化迭代过程的终止条件。也就是当迭代次数超过criteria.maxCount，或者角点位置变化小于criteria.epsilon时，停止迭代过程。

    :param img:
    :return:
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # harris角点检测
    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray, 2, 3, 0.04)
    dst = cv2.dilate(dst, None)
    ret, dst = cv2.threshold(dst, 0.01 * dst.max(), 255, 0)
    dst = np.uint8(dst)

    # 获取质心坐标-centroids 是每个域的质心坐标
    ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)

    # 设置停止和转换条件
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    corners = cv2.cornerSubPix(gray, np.float32(centroids), (5, 5), (-1, -1), criteria)

    # Now draw them
    res = np.hstack((centroids, corners))
    res = np.int0(res)
    img[res[:, 1], res[:, 0]] = [0, 0, 255]
    img[res[:, 3], res[:, 2]] = [0, 255, 0]

    cv2.imshow('SubPix', img)


def harrisImage(img):
    grayImage = np.float32(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    # 输入图像必须是float32，最后一个参数在0.04 到0.05 之间
    # harris角点检测
    harrisImg = cv2.cornerHarris(grayImage, 2, 3, 0.04)
    # 膨胀
    dst = cv2.dilate(harrisImg, None)
    # 对角点图像进行阈值化
    ret, dst = cv2.threshold(dst, 0.01 * dst.max(), 255, 0)
    # 转换为8位图像
    dst = np.uint8(dst)
    # 获取质心坐标-centroids 是每个域的质心坐标
    ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    #  计算得出角点坐标
    corners = cv2.cornerSubPix(grayImage, np.float32(centroids), (5, 5), (-1, -1), criteria)
    # 对所有角点用0圈出来
    for point in corners:
        cv2.circle(img, tuple(point), 5, (0, 0, 255), 2)
    cv2.imshow('harris-2', img)


harrisDemo(img)
harrisSubPix(img)
harrisImage(img)
cv2.waitKey(0)
```

![](https://eveseven.oss-cn-shanghai.aliyuncs.com/20190322214936.png)

