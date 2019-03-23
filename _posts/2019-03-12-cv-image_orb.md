---
layout: post
title: "【十二】ORB角点检测"
date: 2019-03-22
description: "视觉处理算-ORB角点检测"
tag: 机器视觉
---

### ORB角点检测

#### 【简介】

> - ORB（Oriented FAST and Rotated BRIEF）是一种快速特征点提取和描述的算法。
> - 这个算法是由Ethan Rublee, Vincent Rabaud, Kurt Konolige以及Gary R.Bradski在2011年一篇名为“ORB：An Efficient Alternative to SIFT or SURF”的文章中提出。
> - ORB算法分为两部分，分别是特征点提取和特征点描述。特征提取是由FAST（Features from Accelerated Segment Test）算法发展来的，特征点描述是根据BRIEF（Binary Robust Independent Elementary Features）特征描述算法改进的。
> - ORB = oFast + rBRIEF。据称ORB算法的速度是sift的100倍，是surf的10倍。
> - ORB算子在SLAM及无人机视觉等领域得到广泛应用

#### 【oFAST特征提取】

> - ORB算法的特征提取是由FAST算法改进的，这里称为oFAST（FAST keypoint Orientation）。
> - 在使用FAST提取出特征点之后，给其定义一个特征点方向，以此来实现特征点的旋转不变性。

##### `粗提取`

> - 判断特征点：从图像中选取一点P，以P为圆心画一个半径为3像素的圆。圆周上如果有连续N个像素点的灰度值比P点的灰度值大戒小，则认为P为特征点。这就是大家经常说的FAST-N。有FAST-9、FAST-10、FAST-11、FAST-12，大家使用比较多的是FAST-9和FAST-12。
> - 快速算法：为了加快特征点的提取，快速排出非特征点，首先检测1、5、9、13位置上的灰度值，如果P是特征点，那么这四个位置上有3个或3个以上的的像素值都大于或者小于P点的灰度值。如果不满足，则直接排出此点。

![粗提取](https://eveseven.oss-cn-shanghai.aliyuncs.com/20190120215840.png)

##### `筛选最优特征点`

机器学习的方法筛选最优特征点。简单来说就是使用ID3算法训练一个决策树，将特征点圆周上的16个像素输入决策树中，以此来筛选出最优的FAST特征点。具体步骤如下：

> 1. 选取进行角点提取的应用场景下的一组训练图像。
> 2. 使用FAST角点检测算法找出训练图像上的所有角点。
> 3. 对于每一个角点，将其周围的16个像素存储成一个向量。对所有图像都这样做构建一个特征向量。
> 4. 每一个角点的16像素点都属于下列三类中的一种，像素点因此被分成三个子集:$P_d、P_s、P_b$
>
> ![子集](https://eveseven.oss-cn-shanghai.aliyuncs.com/20190120220130.png)
>
> 5. 定义一个新的布尔变量Kp ，如果是角点就设置为True，否则就设置为False。
> 6. 使用ID3算法来查询每一个子集。
> 7. 递归计算所有的子集直到它的熵为0。
>
> 注意：被构建好的决策树可用于其它图像的FAST检测。

##### `使用非极大值抑制算法去除临近位置多个特征点`

> 1. 计算特征点出的FAST得分值s（像素点不周围16个像素点差值的绝对值之和）
> 2. 以特征点p为中心的一个邻域（如3x3或5x5）内，若有多个特征点，则判断每个特征点的s值
> 3. 若p是邻域所有特征点中响应值最大的，则保留；否则，抑制。若邻域内只有一个特征点，则保留。得分计算公式如下（公式中用V表示得分，t表示阈值）：
>
> ![抑制](https://eveseven.oss-cn-shanghai.aliyuncs.com/20190120220601.png)

##### `建立金字塔以实现特征点多尺度不变性`

> - 设置一个比例因子scaleFactor（opencv默认为1.2）和金字塔的层数nlevels（Opencv默认为8）。
> - 将原图像按比例因子缩小成nlevels幅图像。
> - 缩放后的图像为：I'= I/scaleFactork(k=1,2,…, nlevels)。nlevels幅不同比例的图像提取特征点总和作为这幅图像的oFAST特征点。
>
> ![](https://eveseven.oss-cn-shanghai.aliyuncs.com/20190120220736.png)
>
> - 特征点的旋转不变性。ORB算法提出使用矩（moment）法来确定FAST特征点的方向。也就是说通过矩来计算特征点以r为半径范围内的质心，特征点坐标到质心形成一个向量作为该特征点的方向。矩定义如下:$m_{pq}=\Sigma_{x,y} x^p y^q I(x,y), x,y \in [-r,r]$
> - 其中，I(x,y)为图像灰度表达式。该矩的质心为：$C = ( \dfrac{m_{10}}{m_{00}}, \dfrac{m_{01}}{m_{00}})$
> - 假设角点坐标为O，则向量的角度即为该特征点的方向。计算公式如下：$\theta = arctan(m_{01}/m_{10})$

##### `算法特点`

> - FAST算法比其他角点检测算法要快
> - 受图像噪声以及设定阈值影响较大
> - 当设置n<12n<12时，不能用快速方法过滤非角点
> - FAST不产生多尺度特征，不具备旋转不变性，而且检测到的角点不是最优

#### 【rBRIEF】

> ORB算法的特征描述是由BRIEF算法改进的，这里称为rBRIEF（Rotation-Aware Brief）。也就是说，在BRIEF特征描述的基础上加入旋转因子从而改进BRIEF算法。

##### `算法描述`

> 得到特征点后我们需要以某种方式描述这些特征点的属性。这些属性的输出我们称之为该特征点的描述子（Feature Descritors）.ORB采用BRIEF算法来计算一个特征点的描述子。BRIEF算法的**核心思想**是在关键点P的周围以一定模式选取N个点对，把这N个点对的比较结果组合起来作为描述子。
>
> - BRIEF算法计算出来的是一个二进制串的特征描述符。它是在一个特征点的邻域内，选择n对像素点$p_i、q_i（i=1,2,…,n）$。
> - 比较每个点对的灰度值的大小，如果$I(p_i)> I(q_i)$，则生成二进制串中的1，否则为0。
> - 所有的点对都进行比较，则生成长度为n的二进制串。一般n取128、256戒512，opencv默认为256。

![](https://eveseven.oss-cn-shanghai.aliyuncs.com/20190120221620.png)

##### `算法步骤`

> - 以关键点P为圆心，以d为半径做圆O。
> - 在圆O内某一模式选取N个点对。这里为方便说明，N=4，实际应用中N可以取512.
>
> 假设当前选取的4个点对如上图所示分别标记为：![1](https://eveseven.oss-cn-shanghai.aliyuncs.com/20190120221752.png)
>
> - 定义T：
>
> ![](https://eveseven.oss-cn-shanghai.aliyuncs.com/20190120221847.png)
>
> - 分别对已选取的点对进行T操作，将得到的结果进行组合。
>
> ![1](https://eveseven.oss-cn-shanghai.aliyuncs.com/20190120221906.png)
>
> - 则最终的描述子为：1011

##### `具有旋转不变性的BRIEF`

​	ORB并没有解决**尺度一致性**问题，在OpenCV的ORB实现中采用了图像金字塔来改善这方面的性能。ORB主要解决BRIEF描述子不具备**旋转不变性**的问题。

> steered BRIEF（旋转不变性改进）
>
> - 在使用oFast算法计算出的特征点中包括了特征点的方向角度。假设原始的BRIEF算法在特征点SxS（一般S取31）邻域内选取n对点集。
>
> ![](https://eveseven.oss-cn-shanghai.aliyuncs.com/20190120222241.png)
>
> - 经过旋转角度θ旋转，得到新的点对$D_\theta=R_\theta D$
> - 在新的点集位置上比较点对的大小形成二进制串的描述符。

![](https://eveseven.oss-cn-shanghai.aliyuncs.com/20190120222336.png)

##### `描述子的区分性`

  通过上述方法得到的特征描述子具有旋转不变性，称为steered BRIEF(sBRIEF)，但匹配效果却不如原始BRIEF算法，因为可区分性减弱了。特征描述子的一个要求就是要尽可能地表达特征点的独特性，便于区分不同的特征点。如下图所示，为几种特征描述子的均值分布，横轴为均值与0.5之间的距离，纵轴为相应均值下特征点的统计数量。可以看出，BRIEF描述子所有比特位的均值接近于0.5，且方差很大；方差越大表明可区分性越好。不同特征点的描述子表现出较大的差异性，不易造成无匹配。但steered BRIEF进行了坐标旋转，损失了这个特性，导致可区分性减弱，相关性变强，不利于匹配。

![](https://eveseven.oss-cn-shanghai.aliyuncs.com/20190120222445.png)

为了解决steered BRIEF可区分性降低的问题，ORB使用了一种基于学习的方法来选择一定数量的随机点对。

> - 首先建立一个大约300k特征点的数据集(特征点来源于PASCAL2006中的图像)，对每个特征点，考虑其31×31的邻域Patch，为了去除噪声的干扰，选择5×5的子窗口的灰度均值代替单个像素的灰度，这样每个Patch内就有N=(31−5+1)×(31−5+1)=27×27=729个子窗口，从中随机选取2个非重复的子窗口，一共有$M=C_N^2$中方法。
> - 这样，每个特征点便可提取出一个长度为M的二进制串，所有特征点可构成一个300k×M的二进制矩阵Q，矩阵中每个元素取值为0或1。现在需要从M个点对中选取256个`相关性最小、可区分性最大的点对，作为最终的二进制编码`。筛选方法如下：
>   - 对矩阵Q的每一列求取均值，并根据均值与0.5之间的距离从小到大的顺序，依次对所有列向量进行重新排序，得到矩阵T
>   - 将T中的第一列向量放到结果矩阵R中
>   - 取出T中的下一列向量，计算其与矩阵R中所有列向量的相关性，如果相关系数小于给定阈值，则将T中的该列向量移至矩阵R中，否则丢弃
>   - 循环执行上一步，直到R中有256个列向量；如果遍历T中所有列，R中向量列数还不满256，则增大阈值，重复以上步骤。

这样，最后得到的就是相关性最小的256对随机点，该方法称为`rBRIEF`。

#### 【总结】

> 1. ORB = oFAST + rBRIEF
> 2. oFAST是一类快速角点检测算法，并具备旋转不变性
> 3. rBRIEF是一类角点描述(编码算法)，并且编码具有良好的可区分性

### 代码实例

```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/1/19 13:30
# @Author  : Seven
# @File    : ORBDemo.py
# @Software: PyCharm
import cv2
import numpy as np

img1 = cv2.imread('csdn.png')
img2 = np.rot90(img1)

orb = cv2.ORB_create(50)
# 特征点检测和提取特征点描述子
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

# 创建 BFMatcher 对象
bf = cv2.BFMatcher(cv2.NORM_L2)
# 根据描述子匹配特征点.
matches = bf.match(des1, des2)
# 画出50个匹配点
img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches[:50], None, flags=2)

cv2.imshow("ORB", img3)

cv2.waitKey(0)
```

![](https://eveseven.oss-cn-shanghai.aliyuncs.com/20190322220304.png)

转载请注明：[Seven的博客](http://sevenold.github.io)