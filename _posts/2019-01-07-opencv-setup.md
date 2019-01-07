---
layout: post
title: "OpenCV环境配置（win10+opencv3.4.1+vs2017）"
date: 2019-01-07
description: "OpenCV环境配置（win10+opencv3.4.1+vs2017）"
tag: OpenCV
---


**简介：** OpenCV是一个基于BSD许可（开源）发行的跨平台计算机视觉库，可以运行在Linux、Windows、Android和Mac OS操作系统上。它轻量级而且高效——由一系列 C 函数和少量 C++ 类构成，同时提供了Python、Java、Ruby、MATLAB等语言的接口，实现了图像处理和计算机视觉方面的很多通用算法。

### 系统环境

系统：windows 10 - 1809

OpenCV版本：3.4.1

Microsoft Visio Studio：2017

### 环境安装

> 默认vs2017已经安装。

**下载路径**：从[OpenCV Library](https://link.jianshu.com/?t=https%3A%2F%2Fopencv.org%2F)下载安装包并安装

- 进入官网首页，点击【**Releases**】

  ![opencv下载](http://eveseven.oss-cn-shanghai.aliyuncs.com/19-1-7/71143712.jpg)

-  在**Releases** 界面下找到**3.4.1**版本, 因为是Windows平台且使用安装包方式，故点击**【Win pack】**

![opencv3.4.1](http://eveseven.oss-cn-shanghai.aliyuncs.com/19-1-7/4000755.jpg)

- 下载完毕后，【双击安装包】开始安装

  ![opencv安装](http://eveseven.oss-cn-shanghai.aliyuncs.com/19-1-7/51001757.jpg)

  > 我这里是安装在D盘，不用再新建文件夹，解压后会自动创建一个【opencv】的文件夹。
  >
  > 注意：安装路径不要有英文，避免出现问题。

- 等待，直到安装完毕。

### 配置OpenCV的系统环境

- 在【桌面】或者【资源管理器】-> 【右击】【此电脑】-> 选择【属性】

  ![属性](http://eveseven.oss-cn-shanghai.aliyuncs.com/19-1-7/1812340.jpg)

- 选择【高级系统设置】-> 点击【环境变量】

  ![环境变量](http://eveseven.oss-cn-shanghai.aliyuncs.com/19-1-7/3290596.jpg)

- 环境变量界面分为两部分，上方为**【用户变量】**，下方为**【系统变量】**

  ![path](http://eveseven.oss-cn-shanghai.aliyuncs.com/19-1-7/18148003.jpg)

- 在**【系统变量】**列表中找到**【Path】**，双击打开

  ![系统环境变量](http://eveseven.oss-cn-shanghai.aliyuncs.com/19-1-7/43977585.jpg)

- 在打开界面的右侧列表中选择**【浏览】**,进入**opencv安装路径**并选取**XXXX\opencv\build\x64\vc14\bin**目录，逐级**【确定】**保存。

- 系统环境变量设置完毕。

### 在VS2017项目中配置OpenCV

#### 【**新建空项目**】

![新建项目](http://eveseven.oss-cn-shanghai.aliyuncs.com/19-1-7/70133617.jpg)

![空项目](http://eveseven.oss-cn-shanghai.aliyuncs.com/19-1-7/85326264.jpg)

#### 【**新建cpp文件**】

右击->【**源文件**】-> 【**添加**】-> 【**新建项**】

![新建文件](http://eveseven.oss-cn-shanghai.aliyuncs.com/19-1-7/31670791.jpg)

![cpp文件](http://eveseven.oss-cn-shanghai.aliyuncs.com/19-1-7/71795105.jpg)

#### 【**属性管理**】

**VS2017**上方菜单栏找到**【视图】**-->**【其他窗口】**-->**【属性管理器】**，点击开启

![属性管理器](http://eveseven.oss-cn-shanghai.aliyuncs.com/19-1-7/12879883.jpg)

![属性管理器](http://eveseven.oss-cn-shanghai.aliyuncs.com/19-1-7/18122186.jpg)

#### **【Debug X64设置】**

右击【**Debug\|x64**】,点击【**属性**】

![属性](http://eveseven.oss-cn-shanghai.aliyuncs.com/19-1-7/31611889.jpg)

- 在左侧**【通用属性】**菜单下找到**【VC++目录】**，点击，右侧找到**【包含目录】**，点击，右侧出现**【倒三角】**，点击，在弹出下拉列表中点击**【编辑】**

  ![编辑属性](http://eveseven.oss-cn-shanghai.aliyuncs.com/19-1-7/83946786.jpg)

- 在弹出界面中通过点击右上角**【新行】**按钮，添加三条路径：
  **【XXXX\opencv\build\include\opencv2】**
  **【XXXX\opencv\build\include\opencv】**
  **【XXXX\opencv\build\include】**

  ![新行](http://eveseven.oss-cn-shanghai.aliyuncs.com/19-1-7/34773414.jpg)

- 在【**库目录**】添加路径

  **【XXXX\opencv\build\x64\vc14\lib】**

  ![库目录](http://eveseven.oss-cn-shanghai.aliyuncs.com/19-1-7/41741966.jpg)

- 在左侧**【通用属性】**菜单下找到**【链接器】**，展开菜单，选择**【输入】**，在右侧找到**【附加依赖项】**，以**【第4步】**中同样方式打开**【编辑】**，在上方空白处手动键入:**【opencv_world341d.lib】**。

  ![依赖项](http://eveseven.oss-cn-shanghai.aliyuncs.com/19-1-7/9953370.jpg)

  ![附加依赖项](http://eveseven.oss-cn-shanghai.aliyuncs.com/19-1-7/3510846.jpg)

- 保存退出

>  此处键入的文件**随版本号不同而不同**，若非OpenCV 3.4.1版本可自行进入**【XXXX\opencv\build\x64\vc14\lib】**目录查看文件名并键入。

#### 【**Release\|x64设置**】

重复执行**【Debug\|X64设置】**--> 不同的就是【**依赖项**】的选择。

>  文件夹下会看到几个几乎同名的文件，区别仅仅为文件名**末尾有无字母d**，其中d代表debug版本，其他为release版本，以3.4.1版本为例，配置**【Debug\|x64】**时使用**opencv_world341d.lib**，配置**【Release\|x64】**时使用**opencv_world341.lib**.

#### 【**环境搭建完成**】

### 功能测试



#### 【测试代码】

```c++
#include <opencv2/opencv.hpp>
#include <iostream>
using namespace std;
using namespace cv;

int main()
{
    Mat image = imread("opencv.png");  
    imshow("显示图像", image);
    waitKey(0);
    return 0;
}
```

#### 【运行环境】

![运行环境](http://eveseven.oss-cn-shanghai.aliyuncs.com/19-1-7/59828793.jpg)

#### 【**运行结果**】

![运行结果](http://eveseven.oss-cn-shanghai.aliyuncs.com/19-1-7/83983634.jpg)

#### 【**环境搭建成功**】

### 配置属性表

> 我们知道了如何对新建的项目进行属性配置，但若每次都从头进行一遍是十分繁琐的，所以可以通过**【配置属性表并导出】**，下次新建项目时**【导入】**，来减少这些不必要的重复劳动。

#### 【**新建属性表**】

- 以对【**Debug\|x64**】操作为例，右击，选择**【添加新项目属性表】**

![新建属性表](http://eveseven.oss-cn-shanghai.aliyuncs.com/19-1-7/42846413.jpg)

- 在跳出的窗口中选择**【属性表】**格式文件，对文件**【命名】**,**【选择保存位置】**，点击**【添加】**新建成功

![属性表](http://eveseven.oss-cn-shanghai.aliyuncs.com/19-1-7/38050191.jpg)

- 此时我们在**【Debug\|x64】**目录下会看到创建好的属性表文件，对其双击打开，`以同样的方式配置`**【包含目录】**、**【库目录】**、**【附加依赖项】**，保存，对于【**Release\|x64**】同理

  ![属性文件](http://eveseven.oss-cn-shanghai.aliyuncs.com/19-1-7/81378791.jpg)

- 相应的位置即可找到这两个文件

  ![属性文件](http://eveseven.oss-cn-shanghai.aliyuncs.com/19-1-7/31368933.jpg)



#### 【**导入属性表**】

以后**再次新建项目**只需在属性管理器中通过**【添加现有属性表】**导入即可完成配置

![现有属性](http://eveseven.oss-cn-shanghai.aliyuncs.com/19-1-7/25516145.jpg)


转载请注明：[Seven的博客](http://sevenold.github.io)