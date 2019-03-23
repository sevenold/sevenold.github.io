---
layout: post
title: "【十九】立体视觉"
date: 2019-03-23
description: "极线几何与立体视觉-立体视觉"
tag: 机器视觉
---
### 立体视觉

#### 【基本概念】

- 我们在二维视角中，结构和深度是不确定的

  ![](https://eveseven.oss-cn-shanghai.aliyuncs.com/20190302140557.png)

- 立体视觉：第2个照相机可以解决这种歧义性，通过三角化实现深度测量

  ![](https://eveseven.oss-cn-shanghai.aliyuncs.com/20190301202859.png)

#### 【平行双目视觉】

- 假设双目完全平行

  ![](https://eveseven.oss-cn-shanghai.aliyuncs.com/20190302140858.png)

- 空间点三维座标位置求解

  ![](https://eveseven.oss-cn-shanghai.aliyuncs.com/20190302161109.png)

- 视差和深度成反比关系: $z_1 = \frac{bf_x}{u_1-u_2}$

  ![](https://eveseven.oss-cn-shanghai.aliyuncs.com/20190302141519.png)

- 空间点三维座标位置求解：一般情况

  > - 二摄像机坐标系与世界坐标系位姿关系已知
  >
  > ![](https://eveseven.oss-cn-shanghai.aliyuncs.com/20190302141616.png)
  >
  > - 共4个方程，三个未知数，可求解坐标

#### 【三维重构】

- 三维重构步骤

  > 1. 提取特征点，建立特征匹配
  > 2. 计算视差
  > 3. 计算世界坐标
  > 4. 三角剖分
  > 5. 三维重构 

- 提取特征点并建立匹配

  ![](https://eveseven.oss-cn-shanghai.aliyuncs.com/20190302142133.png)

- 特征匹配方式

  > - 特征点提取+特征匹配
  > - 光流匹配
  > - 块匹配
  > - 立体矫正+平行匹配

- 视差计算

  ![](https://eveseven.oss-cn-shanghai.aliyuncs.com/20190302142321.png)

- 计算世界坐标--形成点云数据

  ![](https://eveseven.oss-cn-shanghai.aliyuncs.com/20190302142340.png)

- 三角剖分：采用经典的Delauney算法

  ![](https://eveseven.oss-cn-shanghai.aliyuncs.com/20190302142431.png)

  > Delauney算法示意：
  >
  > ![](https://eveseven.oss-cn-shanghai.aliyuncs.com/20190302142536.png)
  >
  > Delaunay三角网是唯一的（任意四点不能共圆），在Delaunay三角形网中任一三角形的外接圆范围内不会有其它点存在。

- 三维重构：基于计算坐标，采用OpenGL绘制三角片

  ![](https://eveseven.oss-cn-shanghai.aliyuncs.com/20190302142640.png)

  > 效果不好的主要原因，是图像中深度变化较大，同时灰度变换，而特征点选取的比较稀疏.

#### 【总结】

> - 立体视觉可计算空间点的三维坐标。基线越长，距离越近，精度越高
> - 根据双目视觉进行三维重构包括特征点提取、匹配、坐标计算、三角剖分、三维重构等几个步骤

转载请注明：[Seven的博客](http://sevenold.github.io)