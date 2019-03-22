---
layout: post
title: "【七】图像特征描述"
date: 2019-03-22
description: "视觉处理算-图像特征描述"
tag: 机器视觉
---

### 简单描述

#### 【简单描述符】

- 区域面积：区域包含的像素数

- 区域重心：

  ![区域重心](https://eveseven.oss-cn-shanghai.aliyuncs.com/20190116161656.png)

> 注意：区域重心可能不是整数。

#### 【形状描述符】

- 形状参数

  ![形状参数](https://eveseven.oss-cn-shanghai.aliyuncs.com/20190116161905.png)

  > 注意：形状为圆形时：F=1；形状为其他时，F>1

- 偏心率：等效椭圆宽高比。

- 欧拉数：$E = C-H$

- 圆形性：

  ![圆形性](https://eveseven.oss-cn-shanghai.aliyuncs.com/20190116162134.png)

### 一般化描述

- 最小包围矩形（MER）

  ![mer](https://eveseven.oss-cn-shanghai.aliyuncs.com/20190116162239.png)

- 方向和离心率

  ![方向和离心率](https://eveseven.oss-cn-shanghai.aliyuncs.com/20190116162315.png)

### 不变矩

- 首先定义归一化的中心矩

  > 图像f(x,y)的p+q阶矩定义为：
  >
  > ![1](https://eveseven.oss-cn-shanghai.aliyuncs.com/20190116162435.png)
  >
  > f(x,y)的p+q阶中心矩定义为：
  >
  > ![2](https://eveseven.oss-cn-shanghai.aliyuncs.com/20190116162502.png)
  >
  > 其中：
  >
  > ![3](https://eveseven.oss-cn-shanghai.aliyuncs.com/20190116162528.png)
  >
  > 即前面定义的重心。
  >
  > f(x,y)的归一化中心矩定义为：
  >
  > ![4](https://eveseven.oss-cn-shanghai.aliyuncs.com/20190116162603.png)

- 然后定义不变矩

  > 常用的有七个不变矩，即对平移、旋转和尺度变化保持不变。这些可由归一化的二阶和三阶中心矩得到：
  >
  > ![6](https://eveseven.oss-cn-shanghai.aliyuncs.com/20190116162647.png)

- 计算示例

  ![计算示例](https://eveseven.oss-cn-shanghai.aliyuncs.com/20190116162804.png)



### 图像特征描述总结

> 1. 区域的基本特征包括面积、质心、圆形性等.
> 2. 最小包围矩形和多边形拟合能有效的描述区域形状
> 3. 七个不变矩特征可以有效的度量区域形状，对平移、旋转和比例放缩变换不敏感

### 代码实例

```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/1/16 16:39
# @Author  : Seven
# @File    : featureDescription.py
# @Software: PyCharm
import cv2

# 加载图片
img = cv2.imread('rice.png')
cv2.imshow('source', img)
# 转换为灰度图
grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# 形态学处理--五次开运算
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
morphImage_open = cv2.morphologyEx(grayImage, cv2.MORPH_OPEN, kernel, iterations=5)
# 原图减去背景图
riceImage = grayImage - morphImage_open
# 大津算法阈值化
thresholdImage = cv2.threshold(riceImage, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
# 图像分割
partition = cv2.findContours(thresholdImage[1], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
count = 0
for i in partition[1]:
    area = cv2.contourArea(i)
    # print("area:", area)
    if area <= 10:
        continue
    else:
        count += 1
        print("blob:{}:{}".format(count, area))
        # 得到米粒的坐标--x、y、w、h
        rect = cv2.boundingRect(i)
        # 在img中画出最大面积米粒
        cv2.drawContours(img, [i], -1, (255, 255, 0), 1)
        cv2.rectangle(img, (rect[0], rect[1]), (rect[0]+rect[2], rect[1]+rect[3]),  (0, 0, 255), 1)
        cv2.putText(img, str(count), (rect[0], rect[1]), cv2.FONT_HERSHEY_PLAIN, 0.5, (0, 255, 0))

print("米粒数量：%s" % count)
cv2.imshow("area", img)
cv2.waitKey(0)
```

![](https://eveseven.oss-cn-shanghai.aliyuncs.com/20190322212743.png)

