---
layout: post
title: "基于ORB的特征匹配"
date: 2019-03-23
description: "ORB特征匹配"
tag: 机器视觉
---

```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/3/4 12:54
# @Author  : Seven
# @File    : ImageMatching-ORB.py
# @Software: PyCharm
# function : 使用ORB进行提取特征点进行图片匹配
import cv2
# 加载图片
imgL = cv2.imread('image/A.jpg')
imgR = cv2.imread('image/B.jpg')
# 转换为灰度图
grayL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
grayR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)
# 提取特征点
orb = cv2.ORB_create()
kL, dL = orb.detectAndCompute(grayL, None)
kR, dR = orb.detectAndCompute(grayR, None)

# 创建 BFMatcher 对象
bf = cv2.BFMatcher(cv2.NORM_L2)
# 根据描述子匹配特征点.
matches = bf.match(dL, dR)
# 画出匹配点
img3 = cv2.drawMatches(imgL, kL, imgR, kR, matches, None, flags=2)
cv2.imshow("ORB", img3)
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
cv2.imshow("ORB-BF", img3)
cv2.waitKey(0)

```

![](https://eveseven.oss-cn-shanghai.aliyuncs.com/20190323143807.png)

转载请注明：[Seven的博客](http://sevenold.github.io)