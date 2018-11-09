---
layout: post
title: "Anaconda python3.6及pycharm安装和简单使用--Windows"
date: 2018-11-9
description: "Anaconda python3.6安装、pycharm安装和简单使用"
tag: 工具
---

### Windows 安装Anaconda基于python3.6

- anaconda是python发行的包的管理工具，其中自带python的版本，还带很多python的包.安装它比安装python好。可以省掉再安装python包的时间。推荐使用Anaconda,用Anaconda安装python的包是非常便捷高效的，比如安装scrapy框架，如果用原生python的pip安装，要安装很多依赖的包，还经常报错，但是用Anaconda直接输入conda install scrapy就可以了，其他的都不用管了，就是这么方便。另外它自带了 180多个python常用的包，简直就是好得没朋友。

- 要利用 Python 进行科学计算，就需要一一安装所需的模块， 而这些模块可能又依赖于其它的软件包或库，安装和使用起来相对麻烦。Anaconda 就是将科学计算所需要的模块都编译好，然后打包以发行版的形式供用户使用的一个常用的科学计算环境。 

#### 它包含了众多流行的科学、数学、工程、数据分析的 Python 包。

### 下载地址

#### [官网](https://www.anaconda.com/)：https://www.anaconda.com/

#### [最新版本下载地址](https://www.anaconda.com/download/)：https://www.anaconda.com/download/

#### [历史版本](https://repo.anaconda.com/archive/)：https://repo.anaconda.com/archive/

对应有python3.6和python2.7的版本，可自行选择。

我们这里下载的是**Anaconda3-4.4.0-Windows-x86_64.exe(对应是64位python3.6版本)**

### 开始安装

- #### 运行可执行文件

  ![1](/images/py/1.png)

- #### 同意协议

  ![2](/images/py/2.png)

- #### 选择使用者

  ![1](/images/py/3.png)

- #### 选择安装路径

  ![1](/images/py/4.png)

- #### 选择是否添加环境变量和默认启动的是Anaconda python

  > - 增加Anaconda到系统环境变量
  > - Anaconda python为默认解释器
  >   - 勾选后如本地还有其他python解释器，则不是默认使用

  ![1](/images/py/5.png)

- #### installing

  ![1](/images/py/6.png)

- #### 安装完成

  ![1](/images/py/7.png)

  ![1](/images/py/8.png)

### 简单使用

- #### 点击打开

![1](/images/py/9.png)

- #### 打开Anaconda Navigator

  ![1](/images/py/10.png)

- #### 主界面

  ![1](/images/py/11.png)

- #### 添加或删除python库

  - ##### 打开environment

    ![1](/images/py/12.png)

  - ##### 搜索库文件

    ![1](/images/py/13.png)

  - ##### 安装库文件

    ![1](/images/py/14.png)

  - ##### 删除或更新

    ![1](/images/py/15.png)

- #### 创建虚拟环境

  ![1](/images/py/16.png)

  ![1](/images/py/17.png)

  ![1](/images/py/18.png)

- #### 删除虚拟环境

![1](/images/py/19.png)

### windows 安装pycharm

PyCharm是一种Python IDE，其带有一整套可以帮助用户在使用Python语言开发时提高其效率的工具，比如， 调试、语法高亮、Project管理、代码跳转、智能提示、自动完成、单元测试、版本控制等等。此外，该IDE提供了一些高级功能，以用于支持Django框架下的专业Web开发。同时支持Google App Engine，更酷的是，PyCharm支持IronPython！这些功能在先进代码分析程序的支持下，使 PyCharm 成为 Python 专业开发人员和刚起步人员使用的有力工具。

### 下载地址

#### [官网](https://www.jetbrains.com/pycharm/download/#section=windows)：https://www.jetbrains.com/pycharm/download/#section=windows

![1](/images/py/20.png)

### 开始安装

- #### 运行可执行文件

  ![1](/images/py/21.png)

- #### 选择安装地址

  ![1](/images/py/22.png)

- #### 选择安装版本及依赖

  ![1](/images/py/23.png)

- #### 生成快捷方式

  ![1](/images/py/24.png)

- #### installing

  ![1](/images/py/25.png)

- #### 安装完成

  ![1](/images/py/26.png)

### 简单使用

- #### 是否导入设置

  ![1](/images/py/27.png)

- #### 查看并接受协议

  ![1](/images/py/28.png)

- #### 选择主题

  ![1](/images/py/29.png)

- #### 选择是否下载插件

  ![1](/images/py/30.png)

- #### 开始使用

  ![1](/images/py/31.png)

- #### 创建项目或打开项目

  ![1](/images/py/32.png)

- #### 项目保存路径及解释器路径

  ![1](/images/py/33.png)

  ![1](/images/py/34.png)

  ![1](/images/py/35.png)

  ![1](/images/py/36.png)

- #### 创建代码-右键项目文件

  ![1](/images/py/37.png)

- #### 编写代码并执行

  ![1](/images/py/38.png)

- #### 修改字体--依次点击File-settings

  ![1](/images/py/39.png)

- #### 运行结果

  ![1](/images/py/40.png)

- #### 切换python解释器

  - ##### 依次点击File-settings

    ![1](/images/py/41.png)

  - ##### 增加解释器-add

    ![1](/images/py/42.png)

  - ##### 找到我们刚刚用Anaconda 新建的TensorFlow-seven的虚拟环境

    ![1](/images/py/43.png)

  - ##### 保存并退出

    ![1](/images/py/44.png)

### 总结：

上述方法是采用**Anaconda** 配置 的**python**的开发环境

如果你不用，自己配置，只需要把**pycharm**的解释器路径设置为你自己的**python路径**。