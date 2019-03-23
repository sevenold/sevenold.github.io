---
layout: post
title: "【十八】极线几何"
date: 2019-03-23
description: "极线几何与立体视觉-极线几何"
tag: 机器视觉
---
### 极线几何

![](https://eveseven.oss-cn-shanghai.aliyuncs.com/20190301202859.png)

> 有时候我们使用一个相机进行拍摄目标物体的时候，会 发现几个物体都重合了，但是当我们再放台相机的时候，就可以把这些物体的特征获取到。
>
> 也就是双目视觉对应关系，同时也可用于相邻两帧间的运动估计。

#### 【基本概念】

> 极线几何(Epipolar geometry)
>
> - (匹配)极线约束：匹配点必须在极线上
> - 基线：左右像机光心连线
> - 极平面：空间点，两像机光心决定的平面
> - 极点：基线与两摄像机图像平面的交点
> - 极线：极平面与图片平面的交线

![](https://eveseven.oss-cn-shanghai.aliyuncs.com/20190301203308.png)

#### 【本质矩阵】

> **本质矩阵** **E**（Essential Matrix）：反映【**空间一点** **P** **的像点】**在【**不同视角摄像机】**下【**摄像机坐标系】**中的表示之间的关系。

- 前面我们已经知道了各个坐标系之前的转换

  - 相机坐标系与世界坐标系

  ![](https://eveseven.oss-cn-shanghai.aliyuncs.com/20190301204745.png)

  - 相机坐标系与图像坐标系

  ![](https://eveseven.oss-cn-shanghai.aliyuncs.com/1551444580057.png)

- 两相机坐标系某点与对应图像坐标系的关系：

  ![](https://eveseven.oss-cn-shanghai.aliyuncs.com/20190301205229.png)

- 同一点在两相机坐标系之间的关系：

  ![](https://eveseven.oss-cn-shanghai.aliyuncs.com/20190301205312.png)

- 两边同时叉积$t$：

  ![](https://eveseven.oss-cn-shanghai.aliyuncs.com/20190301205438.png)

- 再与$p_r^\sim​$点积：

  ![](https://eveseven.oss-cn-shanghai.aliyuncs.com/1551445425466.png)

#### 【本质矩阵求解】

- 基本方程

  ![](https://eveseven.oss-cn-shanghai.aliyuncs.com/20190301210555.png)

- 线性方程求解

  ![](https://eveseven.oss-cn-shanghai.aliyuncs.com/20190301210713.png)

  > 有九个点(非共面)时，可获得线性解：
  >
  > ![](https://eveseven.oss-cn-shanghai.aliyuncs.com/20190301210826.png)
  >
  > 注意：解与真实解相差一个比例系数

- 使用SVD分解求解平移和旋转矩阵

  ![](https://eveseven.oss-cn-shanghai.aliyuncs.com/20190301210911.png)

  ![](https://eveseven.oss-cn-shanghai.aliyuncs.com/20190301210925.png)

  > 可以证明，本质矩阵有2个相同的非零特征值

  ![](https://eveseven.oss-cn-shanghai.aliyuncs.com/1551446959523.png)

  ![](https://eveseven.oss-cn-shanghai.aliyuncs.com/20190301212931.png)

  > 因此，最终可以得到4个解，但仅有一个合理解

#### 【扩展】

> **基本矩阵(Fundamental matrix)**： 反映【**空间一点** **P** **的像素点】**在【**不同视角摄像机】**下【**图像坐标系**】中的表示之间的关系。

![](https://eveseven.oss-cn-shanghai.aliyuncs.com/20190301213256.png)

> 两个对应点在像素座标系的对应关系(包含相机内参数信息)

#### 【总结】

> - 极线是极平面和像平面交线，极点是极线和基线交点
> - 本质矩阵确定了两帧图像中对应点的约束关系
> - 可以通过8个对应点求解本质矩阵，进一步分解得到R和t

### 代码实例

```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/3/1 22:03
# @Author  : Seven
# @File    : PolarGeometry.py
# @Software: PyCharm
# function : 极线几何，测量极线
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 加载图片
img1 = cv2.imread('image/l.jpg', 0)
img2 = cv2.imread('image/r.jpg', 0)

# 初始化SIFT方法
sift = cv2.xfeatures2d_SIFT.create()

# 获取关键点和描述子
k1, d1 = sift.detectAndCompute(img1, None)
k2, d2 = sift.detectAndCompute(img2, None)

# 设置FLANN 超参数
FLANN_INDEX_KDTREE = 0
# K-D树索引超参数
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
# 搜索超参数
search_params = dict(checks=50)

# 初始化FlannBasedMatcher匹配器
flann = cv2.FlannBasedMatcher(index_params, search_params)
# 通过KNN的方式匹配两张图的描述子
matches = flann.knnMatch(d1, d2, k=2)

good = []
pts1 = []
pts2 = []

# 筛选比较好的匹配点
for i, (m, n) in enumerate(matches):
    if m.distance < 0.8 * n.distance:
        good.append(m)
        pts2.append(k2[m.trainIdx].pt)
        pts1.append(k1[m.queryIdx].pt)

# 计算基础矩阵
pts1 = np.int32(pts1)
pts2 = np.int32(pts2)
# F为基本矩阵、mask是返回基本矩阵的值：没有找到矩阵，返回0，找到一个矩阵返回1，多个矩阵返回3
F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_LMEDS)

# 只选择有效数据
pts1 = pts1[mask.ravel() == 1]
pts2 = pts2[mask.ravel() == 1]


def drawlines(img1, img2, lines, pts1, pts2):
    """
    绘制图像极线
    :param img1:
    :param img2:
    :param lines:
    :param pts1:
    :param pts2:
    :return:
    """
    r, c = img1.shape
    img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)

    for r, pt1, pt2 in zip(lines, pts1, pts2):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        x0, y0 = map(int, [0, -r[2] / r[1]])
        x1, y1 = map(int, [c, -(r[2] + r[0] * c) / r[1]])

        img1 = cv2.line(img1, (x0, y0), (x1, y1), color, 1)
        img1 = cv2.circle(img1, tuple(pt1), 5, color, -1)
        img2 = cv2.circle(img2, tuple(pt2), 5, color, -1)
    return img1, img2


# 在右图（第二图）中找到与点相对应的极线，并在左图上画出它的线。
lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1, 1, 2), 2, F)
lines1 = lines1.reshape(-1, 3)
img5, img6 = drawlines(img1, img2, lines1, pts1, pts2)
# 找到与左图像（第一个图像）中的点对应的极线，以及在右图上画线
lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1, 1, 2), 1, F)
lines2 = lines2.reshape(-1, 3)
img3, img4 = drawlines(img2, img1, lines2, pts2, pts1)
plt.subplot(121), plt.imshow(img5)
plt.subplot(122), plt.imshow(img3)
plt.show()

```

![](https://eveseven.oss-cn-shanghai.aliyuncs.com/20190323142108.png)

转载请注明：[Seven的博客](http://sevenold.github.io)