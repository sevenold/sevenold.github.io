---
layout: post
title: "Anaconda python3.6及pycharm安装和简单使用--Ubuntu16.04"
date: 2018-11-9
description: "Anaconda python3.6安装、pycharm安装和简单使用"
tag: 工具
---

### Ubuntu 安装Anaconda基于python3.6

- anaconda是python发行的包的管理工具，其中自带python的版本，还带很多python的包.安装它比安装python好。可以省掉再安装python包的时间。推荐使用Anaconda,用Anaconda安装python的包是非常便捷高效的，比如安装scrapy框架，如果用原生python的pip安装，要安装很多依赖的包，还经常报错，但是用Anaconda直接输入conda install scrapy就可以了，其他的都不用管了，就是这么方便。另外它自带了 180多个python常用的包，简直就是好得没朋友。

- 要利用 Python 进行科学计算，就需要一一安装所需的模块， 而这些模块可能又依赖于其它的软件包或库，安装和使用起来相对麻烦。Anaconda 就是将科学计算所需要的模块都编译好，然后打包以发行版的形式供用户使用的一个常用的科学计算环境。 

#### 它包含了众多流行的科学、数学、工程、数据分析的 Python 包。

### 下载地址

#### [官网](https://www.anaconda.com/)：https://www.anaconda.com/

#### [最新版本下载地址](https://www.anaconda.com/download/)：https://www.anaconda.com/download/

#### [历史版本](https://repo.anaconda.com/archive/)：https://repo.anaconda.com/archive/

对应有python3.6和python2.7的版本，可自行选择。

我们这里下载的是**[Anaconda3-4.4.0-Linux-x86_64.sh](https://repo.anaconda.com/archive/Anaconda3-4.4.0-Linux-x86_64.sh)(对应是64位python3.6版本)**

### 开始安装

- #### 进入文件所在路径

  ![1](/images/py/45.png)

- #### 开始安装

  - ##### 出现欢迎信息， 阅读许可文件

    ![1](/images/py/46.png)

  - ##### 同意许可

    ![1](/images/py/47.png)

  - ##### 选择安装目录-默认

    ![1](/images/py/48.png)

  - ##### installing

    ![1](/images/py/49.png)

  - ##### 添加系统环境

    ![1](/images/py/50.png)

  - ##### 安装完成

    ![1](/images/py/51.png)

  - ##### 使环境生效

  ```shell
  seven@ubuntu:~/Desktop$ source ~/.bashrc
  ```


### Conda的环境管理

-  创建一个名为python-seven的环境，指定Python版本是3.6（不用管是3.6.x，conda会为我们自动寻找3.6.x中的最新版本）

  ```shell
  seven@ubuntu:~$ conda create --name python-seven python=3.6
  ```

- 安装好后，使用activate激活某个环境

  ```shell
  seven@ubuntu:~$ source activate python-seven
  (python-seven) seven@ubuntu:~$ 
  ```

- 测试环境

  ```shell
  (python-seven) seven@ubuntu:~$ python --version
  Python 3.6.2 :: Continuum Analytics, Inc.
  (python-seven) seven@ubuntu:~$ 
  ```

- 退出虚拟环境

  ```shell
  (python-seven) seven@ubuntu:~$ source deactivate python-seven
  seven@ubuntu:~$ 
  ```

- 删除虚拟环境

  ```shell
  seven@ubuntu:~$ conda remove --name python-seven --all
  ```


### ubuntu16.04 安装pycharm

PyCharm是一种Python IDE，其带有一整套可以帮助用户在使用Python语言开发时提高其效率的工具，比如， 调试、语法高亮、Project管理、代码跳转、智能提示、自动完成、单元测试、版本控制等等。此外，该IDE提供了一些高级功能，以用于支持Django框架下的专业Web开发。同时支持Google App Engine，更酷的是，PyCharm支持IronPython！这些功能在先进代码分析程序的支持下，使 PyCharm 成为 Python 专业开发人员和刚起步人员使用的有力工具。

### 下载地址

[官网](https://www.jetbrains.com/pycharm/download/#section=windows)：https://www.jetbrains.com/pycharm/download/#section=windows

![1](/images/py/52.png)

### 开始安装

- #### 进入文件所在路径

  ```shell
  seven@ubuntu:~$ cd Desktop/
  seven@ubuntu:~/Desktop$ ls
  Anaconda3-4.4.0-Linux-x86_64.sh  Mnist  pycharm-community-2018.2.4.tar.gz
  ```

- #### 解压文件

  ```shell
  seven@ubuntu:~/Desktop$ sudo tar -zxvf pycharm-community-2018.2.4.tar.gz -C /opt/
  ```

- #### 开始安装

  ```shell
  seven@ubuntu:~/Desktop$ cd /opt/pycharm-community-2018.2.4/bin/
  seven@ubuntu:/opt/pycharm-community-2018.2.4/bin$ ./p
  printenv.py  pycharm.sh   
  seven@ubuntu:/opt/pycharm-community-2018.2.4/bin$ ./pycharm.sh 
  ```

#### 

### 简单使用

- #### 是否导入设置

  ![1](/images/py/54.png)

- #### 查看并接受协议

  ![1](/images/py/55.png)

- #### 选择主题

  ![1](/images/py/56.png)

- #### 创建项目保存路径

  ![1](/images/py/57.png)

- #### 选择是否下载插件

  ![1](/images/py/58.png)

- #### 开始使用

  ![1](/images/py/31.png)

- #### 创建项目或打开项目

  ![1](/images/py/32.png)

- #### 项目保存路径及解释器路径

  ![1](/images/py/59.png)

  ![1](/images/py/60.png)

  ![1](/images/py/61.png)

  ![1](/images/py/62.png)

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

    ![1](/images/py/63.png)

  - ##### 增加解释器-add

    ![1](/images/py/64.png)

  - ##### 找到我们刚刚用Anaconda 新建的python-seven的虚拟环境

    ![1](/images/py/65.png)

  - ##### 保存并退出

    ![1](/images/py/66.png)

### 总结：

上述方法是采用**Anaconda** 配置 的**python**的开发环境

如果你不用，自己配置，只需要把**pycharm**的解释器路径设置为你自己的**python路径**。