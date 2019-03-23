---
layout: post
title: "基于SURF的特征匹配"
date: 2019-03-23
description: "SURF特征匹配"
tag: 机器视觉
---

```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/3/2 21:36
# @Author  : Seven
# @File    : ImageMatching-SURF.py
# @Software: PyCharm
# function : 使用SURF进行提取特征点进行图片匹配
import cv2
# 加载图片
imgL = cv2.imread('image/A.jpg')
imgR = cv2.imread('image/B.jpg')
# 转换为灰度图
grayL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
grayR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)
# 提取特征点
surf = cv2.xfeatures2d.SURF_create(hessianThreshold=800)   # hessian矩阵阈值，在这里调整精度，值越大，点越少，越精准
kL, dL = surf.detectAndCompute(grayL, None)
kR, dR = surf.detectAndCompute(grayR, None)

# 创建 BFMatcher 对象
bf = cv2.BFMatcher(cv2.NORM_L2)
# 根据描述子匹配特征点.
matches = bf.match(dL, dR)
# 画出匹配点
img3 = cv2.drawMatches(imgL, kL, imgR, kR, matches, None, flags=2)
cv2.imshow("SURF", img3)
# ------------------------------------------------------------------------------------------------
# 设置FLANN 超参数
FLANN_INDEX_KDTREE = 0
# K-D树索引超参数
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
# 搜索超参数
search_params = dict(checks=50)
# 初始化FlannBasedMatcher匹配器
flann = cv2.FlannBasedMatcher(index_params, search_params)
# 通过KNN的方式匹配两张图的描述子
matches = flann.knnMatch(dL, dR, k=2)
# 筛选比较好的匹配点
good = []
for i, (m, n) in enumerate(matches):
    if m.distance < 0.6 * n.distance:
        good.append(m)
# 画出匹配点
img3 = cv2.drawMatches(imgL, kL, imgR, kR, good, None, flags=2)
cv2.imshow("SURF-FLANN", img3)
cv2.waitKey(0)
```

![](https://eveseven.oss-cn-shanghai.aliyuncs.com/20190323144253.png)

转载请注明：[Seven的博客](http://sevenold.github.io)