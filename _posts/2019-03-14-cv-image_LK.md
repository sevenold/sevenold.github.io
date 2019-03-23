---
layout: post
title: "【十四】光流估计"
date: 2019-03-23
description: "运动估计-光流估计"
tag: 机器视觉
---
### 光流估计

#### 【**简介**】

> ​	在计算机视觉中，Lucas–Kanade光流算法是一种两帧差分的光流估计算法。它由Bruce D. Lucas 和 Takeo Kanade提出。

#### 【**光流的概念**】

> (Optical flow or optic flow)它是一种运动模式，这种运动模式指的是一个物体、表面、边缘在一个视角下由一个观察者（比如眼睛、摄像头等）和背景之间形成的明显移动。
>
> ​	光流技术，如运动检测和图像分割，时间碰撞，运动补偿编码，三维立体视差，都是利用了这种边缘或表面运动的技术。
>
> ​	二维图像的移动相对于观察者而言是三维物体移动的在图像平面的投影。
>
> ​	有序的图像可以估计出二维图像的瞬时图像速率或离散图像转移。

#### 【光流算法】

> ​	它评估了两幅图像的之间的变形，它的`基本假设是体素和图像像素守恒`。它假设一个物体的颜色在前后两帧没有巨大而明显的变化。
>
> ​	基于这个思路，我们可以得到图像约束方程。不同的光流算法解决了假定了不同附加条件的光流问题。

#### 【基本模型】

![](https://eveseven.oss-cn-shanghai.aliyuncs.com/20190123154113.png)

![](https://eveseven.oss-cn-shanghai.aliyuncs.com/20190123160039.png)

> 基本思想：虽然$t+1$的时候位置进行了移动，如果我们知道`灰度是连续变化`的，那么我们就可以通过像素的变化推演当前这个中心点大概移动了多少。
>
> 通过当前位置亮度在$I_x，I_y$以及相邻两帧中灰度的变化值$I_t$，来推演$\delta _x，\delta_y$.

![](https://eveseven.oss-cn-shanghai.aliyuncs.com/20190123154311.png)

> `我们在检测物体上建立一个小方格里的所有像素位移相同，建立矩阵方程组，解出u、v两个未知数。`

- 最优化问题（超定方程求解）

  $min \|Au-b\|$

- 最小二乘解

  $u=(A^TA)^{-1}A^Tb$

> - 区域像素只有2个时，就是2元1次方程组求解！
> - 多个像素，比如3∗3时，则是求上述最小二乘解.

#### 【Lucas–Kanade方法】

> ​	这个算法是最常见，最流行的。它计算两帧在时间t 到t + δt之间每个每个像素点位置的移动。 由于它是基于图像信号的泰勒级数，这种方法称为差分，这就是对于空间和时间坐标使用偏导数。

![](https://eveseven.oss-cn-shanghai.aliyuncs.com/20190123161352.png)

> 可信度判断：矩阵求逆是否能实现？

![](https://eveseven.oss-cn-shanghai.aliyuncs.com/20190123161715.png)

> - 该表达式是和沿着x和y的梯度是密切相关的，只要其中一个为0时，$A^TA$就不可逆。只要当该区域的像素点比较平滑，矩阵就可逆
> - 通过求特征值来判断计算可信。
>   - 两个特征值都远大于0，那么就可逆，否则就不可逆或者很难求逆。

#### 【金字塔L-K方法】

> 针对目标运动很快的情况，用原始的L-K方法就没有办法处理了， 需要采用图像金字塔的方法。把图像从小到大的顺序排列。

> ​	金字塔特征跟踪算法描述如下：首先，光流和仿射变换矩阵在最高一层的图像上计算出；将上一层的计算结果作为初始值传递给下一层图像，这一层的图像在这个初始值的基础上，计算这一层的光流和仿射变化矩阵；再将这一层的光流和仿射矩阵作为初始值传递给下一层图像，直到传递给最后一层，即原始图像层，这一层计算出来的光流和仿射变换矩阵作为最后的光流和仿射变换矩阵的结果。

- 用不同像素的图片建立金字塔

  ![](https://eveseven.oss-cn-shanghai.aliyuncs.com/20190123185018.png)

- 运行L-K方法

  ![](https://eveseven.oss-cn-shanghai.aliyuncs.com/20190123185057.png)

> 优点：可处理大位移；对噪声更不敏感。

#### 【**数学模型**】

![](https://eveseven.oss-cn-shanghai.aliyuncs.com/20190123190149.png)

### 总结

> 1. 光流估计基于恒定亮度假设模型
> 2. L-K光流估计方法利用了邻域内的运动不变性
> 3. 金字塔L-K方法可有效提高光流计算对大位移的鲁棒性

### 代码实例

```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/1/22 22:20
# @Author  : Seven
# @File    : sightDemo.py
# @Software: PyCharm
import cv2
import numpy as np
# 加载视频
cap = cv2.VideoCapture()
cap.open('768x576.avi')
if not cap.isOpened():
    print("无法打开视频文件")

lk_params = dict(winSize=(15, 15),
                 maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

feature_params = dict(maxCorners=500,
                      qualityLevel=0.3,
                      minDistance=7,
                      blockSize=7)


class App:
    def __init__(self, video_src):  # 构造方法，初始化一些参数和视频路径
        self.track_len = 10
        self.detect_interval = 5
        self.tracks = []
        self.cam = cv2.VideoCapture(video_src)
        self.frame_idx = 0

    def run(self):  # 光流运行方法
        while True:
            ret, frame = self.cam.read()  # 读取视频帧
            if ret:
                frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # 转化为灰度虚图像
                vis = frame.copy()

                if len(self.tracks) > 0:  # 检测到角点后进行光流跟踪
                    img0, img1 = self.prev_gray, frame_gray
                    p0 = np.float32([tr[-1] for tr in self.tracks]).reshape(-1, 1, 2)
                    p1, st, err = cv2.calcOpticalFlowPyrLK(img0, img1, p0, None,
                                                           **lk_params)  # 前一帧的角点和当前帧的图像作为输入来得到角点在当前帧的位置
                    p0r, st, err = cv2.calcOpticalFlowPyrLK(img1, img0, p1, None,
                                                            **lk_params)  # 当前帧跟踪到的角点及图像和前一帧的图像作为输入来找到前一帧的角点位置
                    d = abs(p0 - p0r).reshape(-1, 2).max(-1)  # 得到角点回溯与前一帧实际角点的位置变化关系
                    good = d < 1  # 判断d内的值是否小于1，大于1跟踪被认为是错误的跟踪点
                    new_tracks = []
                    for tr, (x, y), good_flag in zip(self.tracks, p1.reshape(-1, 2), good):  # 将跟踪正确的点列入成功跟踪点
                        if not good_flag:
                            continue
                        tr.append((x, y))
                        if len(tr) > self.track_len:
                            del tr[0]
                        new_tracks.append(tr)
                        cv2.circle(vis, (x, y), 2, (0, 255, 0), -1)
                    self.tracks = new_tracks
                    cv2.polylines(vis, [np.int32(tr) for tr in self.tracks], False,
                                  (0, 255, 0))  # 以上一振角点为初始点，当前帧跟踪到的点为终点划线
                    # draw_str(vis, (20, 20), 'track count: %d' % len(self.tracks))

                if self.frame_idx % self.detect_interval == 0:  # 每5帧检测一次特征点
                    mask = np.zeros_like(frame_gray)  # 初始化和视频大小相同的图像
                    mask[:] = 255  # 将mask赋值255也就是算全部图像的角点
                    for x, y in [np.int32(tr[-1]) for tr in self.tracks]:  # 跟踪的角点画圆
                        cv2.circle(mask, (x, y), 5, 0, -1)
                    p = cv2.goodFeaturesToTrack(frame_gray, mask=mask, **feature_params)  # 像素级别角点检测
                    if p is not None:
                        for x, y in np.float32(p).reshape(-1, 2):
                            self.tracks.append([(x, y)])  # 将检测到的角点放在待跟踪序列中

                self.frame_idx += 1
                self.prev_gray = frame_gray
                cv2.imshow('lk_track', vis)

            ch = 0xFF & cv2.waitKey(100)
            if ch == 27:
                break

App('768x576.avi').run()
```

![](https://eveseven.oss-cn-shanghai.aliyuncs.com/123.gif)

转载请注明：[Seven的博客](http://sevenold.github.io)