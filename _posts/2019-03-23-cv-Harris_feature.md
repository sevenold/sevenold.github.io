---
layout: post
title: "基于Harris的特征匹配"
date: 2019-03-23
description: "Harris特征匹配"
tag: 机器视觉
---

```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/3/4 14:35
# @Author  : Seven
# @File    : ImageMatching-Harris.py
# @Software: PyCharm
# function : 使用Harris进行提取特征点进行图片匹配
# 选择了一种速度、特征点数量和精度都比较好的组合方案：
# FAST角点检测算法+SURF特征描述子+FLANN(Fast Library for Approximate Nearest Neighbors) 匹配算法。
import cv2
# 加载图片
imgL = cv2.imread('image/A.jpg')
imgR = cv2.imread('image/B.jpg')
# 转换为灰度图
grayL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
grayR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)
# 提取特征点
harris = cv2.xfeatures2d_HarrisLaplaceFeatureDetector.create()
kL = harris.detect(grayL, None)
kR = harris.detect(grayR, None)
# 提取描述子
br = cv2.BRISK_create()
kL, dL = br.compute(grayL, kL)
kR, dR = br.compute(grayR, kR)

# 创建 BFMatcher 对象
bf = cv2.BFMatcher(cv2.NORM_L2)
# 根据描述子匹配特征点.
matches = bf.match(dL, dR)
# 画出匹配点
img3 = cv2.drawMatches(imgL, kL, imgR, kR, matches, None, flags=2)
cv2.imshow("Harris", img3)
# ------------------------------------------------------------------------------------------------
# 初始化Bruteforce匹配器
bf = cv2.BFMatcher()
# 通过KNN匹配两张图片的描述子
matches = bf.knnMatch(dL, dR, k=2)
# 筛选比较好的匹配点
good = []
for i, (m, n) in enumerate(matches):
    if m.distance < 0.6 * n.distance:
        good.append(m)
# 画出匹配点
img3 = cv2.drawMatches(imgL, kL, imgR, kR, good, None, flags=2)
cv2.imshow("Harris-BF", img3)
cv2.waitKey(0)
```

![](https://eveseven.oss-cn-shanghai.aliyuncs.com/20190323143639.png)

转载请注明：[Seven的博客](http://sevenold.github.io)