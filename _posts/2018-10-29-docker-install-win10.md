---
layout: post
title: "Docker安装-windows"
date: 2018-10-29
description: "Docker安装及系列教程"
tag: Docker
---

### windows安装Docker

Docker发布了Docker for Windows的正式版，于是就可以在Windows下运行Docker容器。

### 版本要求

要在Windows下运行Docker，需要满足以下先决条件：

- 64位Windows 10 Pro、Enterprise或者Education版本（Build 10586以上版本，需要安装1511 November更新）
- 在系统中启用Hyper-V。如果没有启用，Docker for Windows在安装过程中会自动启用Hyper-V（这个过程需要重启系统）
   不过，如果不是使用的Windows 10，也没有关系，可以使用[Docker Toolbox](https://link.jianshu.com?t=http%3A%2F%2Fwww.docker.com%2Fproducts%2Fdocker-toolbox)作为替代方案。

### 启用系统的Hper-V 

打开**程序**和**功能**，右击**开始**选择**应用和功能**。

![1](/images/docker/1.png)

打开右边的**程序**和**功能**

![1](/images/docker/2.png)

打开左边的**启用或关闭windows功能**

![1](/images/docker/3.png)

启用**Hper-V**服务 

![1](/images/docker/4.png)

然后**重启计算机**，使服务生效

#### Docker for Windows的安装

在Windows 10中，请[点击此处](https://link.jianshu.com/?t=https%3A%2F%2Fdownload.docker.com%2Fwin%2Fstable%2FInstallDocker.msi)下载Docker for Windows的安装包，下载好之后双击 Docker for Windows Installer.exe 开始安装。安装好后，桌面会出现如下快捷方式，表明docker安装成功。

![1](/images/docker/5.png)



Docker CE 启动之后会在 Windows 任务栏出现鲸鱼图标。

等待片刻，点击 Got it 开始使用 Docker CE。

![1](/images/docker/6.png)