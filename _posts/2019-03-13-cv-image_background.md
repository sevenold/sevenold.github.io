---
layout: post
title: "【十三】背景建模"
date: 2019-03-23
description: "运动估计-背景建模"
tag: 机器视觉
---

### 背景建模

#### 相对运动的基本方式

> - 相机静止，目标运劢——背景提取(减除)
> - 相机运劢，目标静止——光流估计(全局运劢)
> - 相机和目标均运劢——光流估计

#### 帧差法运动目标检测

![](https://eveseven.oss-cn-shanghai.aliyuncs.com/20190123141057.png)

​	如图可见，由目标运动引起的运动变化区域包括运动目标在前后两帧中的共同位置(图中黑色区域)、在当前帧中新显露出的背景区域和新覆盖的背景区域三部分。

##### 【**数学模型**】

$sign(x)=\begin{cases}
1,&如果I(x， y， t)-I(x, y, t-1)>T \\ 0,&如果其他情况
\end{cases}$

> - D(x, y): 帧差
> - I(x,y,t): 当前帧(t时刻)图像
> - I(x,y,t): 上一帧(t-1时刻)图像
> - T: 像素灰度差阈值

#### 混合高斯背景建模

##### 【**高斯背景**】

> 像素灰度值随时间变化符合高斯分布.
>
> $I(x, y) \thicksim N(u, \sigma^2)$

![](https://eveseven.oss-cn-shanghai.aliyuncs.com/20190123143447.png)

> - 如果$I(x, y, t)-u>3\sigma$ 为`前景`，否则就是`背景`。
>   - $I(x, y, t)$是当前帧。
>   - $u$为均值、$\sigma$是方差。
>
> 注意：当前帧的灰度值变化落在$3\sigma$间的就是背景，否则就是前景。

##### 【**混合高斯模型**】

> 任何一种分布函数都可以看做是多个高斯分布的组合.

![](https://eveseven.oss-cn-shanghai.aliyuncs.com/20190123144119.png)

​	混合高斯背景建模是`基于像素样本统计信息的背景表示方法`，利用像素在较长时间内大量样本值的概率密度等统计信息(如模式数量、每个模式的均值和标准差)表示背景，然后使用`统计差分(如3σ原则)进行目标像素判断`，可以对复杂动态背景进行建模，计算量较大。
​	在混合高斯背景模型中，认为`像素之间的颜色信息互不相关，对各像素点的处理都是相互独立的`。对于视频图像中的每一个像素点，其值在序列图像中的变化可看作是不断产生像素值的随机过程，即`用高斯分布来描述每个像素点的颜色呈现规律{单模态(单峰)，多模态(多峰)}`。

​	对于多峰高斯分布模型，`图像的每一个像素点按不同权值的多个高斯分布的叠加来建模，每种高斯分布对应一个可能产生像素点所呈现颜色的状态，各个高斯分布的权值和分布参数随时间更新`。当处理彩色图像时，假定图像像素点R、G、B三色通道相互独立并具有相同的方差。对于随机变量X的观测数据集${x_1,x_2,…,x_N}，x_t=(r_t,g_t,b_t)$为t时刻像素的样本，则单个采样点$x_t$其服从的混合高斯分布概率密度函数：

![](https://eveseven.oss-cn-shanghai.aliyuncs.com/20190123144742.png)

> ​	其中Q为分布模式总数，$N(I,u_q, \sigma_q^2)$为高斯分布，$μ_q$为其均值，$δ_q$为方差，I为三维单位矩阵，$ω_q$为高斯分布的权重。

##### 【**混合高斯背景建模步骤**】

> 任务：在线计算：$μ_q$，$δ_q$，$ω_q$

> 1. 模型初始化 将采到的第一帧图像的每个象素的灰度值作为均值，再赋以较大的方差。初值Q=1, w=1.0。
> 2. 模型学习 将当前帧的对应点象素的灰度值与已有的Q个高斯模型作比较，若满足$\|x_k-u_{q, k}\|<2.5\sigma_{q,k}$ ，则按概率密度公式调整第q个高斯模型的参数和权重；否则转入(3)：
> 3. 增加/替换高斯分量 若不满足条件，且q<Q，则增加一个新分量；若q=Q，则替换
> 4. 判断背景  $B = argmin_b(\sum_{q=1}^bw_q>T)$
> 5. 判断前景

##### 【**混合高斯模型迭代计算原理**】

**迭代计算：**

![](https://eveseven.oss-cn-shanghai.aliyuncs.com/20190123150252.png)

> - $M_q(k)$为二值化函数，仅当像素值匹配第q类时取1，其余取0
> - 类别数值取值不大于5

### 总结

> 1. 背景静止时，可以使用基于背景提取的运动估计斱法计算运动目标。
> 2. 混合高斯模型可模拟任意概率密度函数，是背景建模的主流方法
> 3. 混合高斯模型参数采用迭代方式计算。



### 代码实例

```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/1/22 21:09
# @Author  : Seven
# @File    : backgroundDemo.py
# @Software: PyCharm
# function : 基于背景提取的运动估计

import cv2

# 加载视频
cap = cv2.VideoCapture()
cap.open('768x576.avi')
if not cap.isOpened():
    print("无法打开视频文件")

pBgModel = cv2.createBackgroundSubtractorMOG2()


def labelTargets(img, mask, threshold):
    seg = mask.copy()
    cnts = cv2.findContours(seg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    count = 0
    for i in cnts[1]:
        area = cv2.contourArea(i)
        if area < threshold:
            continue
        count += 1
        rect = cv2.boundingRect(i)
        print("矩形：X:{} Y:{} 宽：{} 高：{}".format(rect[0], rect[1], rect[2], rect[3]))
        cv2.drawContours(img, [i], -1, (255, 255, 0), 1)
        cv2.rectangle(img, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 0, 255), 1)
        cv2.putText(img, str(count), (rect[0], rect[1]), cv2.FONT_HERSHEY_PLAIN, 0.5, (0, 255, 0))
    return count


while True:
    flag, source = cap.read()
    if not flag:
        break
    image = cv2.pyrDown(source)

    fgMask = pBgModel.apply(image)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    morphImage_open = cv2.morphologyEx(fgMask, cv2.MORPH_OPEN, kernel, iterations=5)
    mask = fgMask - morphImage_open
    _, Mask = cv2.threshold(mask, 30, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Mask = cv2.GaussianBlur(Mask, (5, 5), 0)
    targets = labelTargets(image, Mask, 30)

    print("共检测%s个目标" % targets)
    backGround = pBgModel.getBackgroundImage()
    foreGround = image - backGround
    cv2.imshow('source', image)
    cv2.imshow('background', backGround)
    cv2.imshow('foreground', Mask)
    key = cv2.waitKey(10)
    if key == 27:
        break
```

![](https://eveseven.oss-cn-shanghai.aliyuncs.com/GIF.gif)

