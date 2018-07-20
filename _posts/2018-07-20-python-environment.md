---
layout: post
title: "python环境配置及安装"
date: 2015-08-25 
description: "各大平台的python安装方法"
tag: python
---   



![img](https://timgsa.baidu.com/timg?image&quality=80&size=b9999_10000&sec=1512213234044&di=f62c5246289d1c0aafffcb7678628bb6&imgtype=jpg&src=http%3A%2F%2Fimg4.imgtn.bdimg.com%2Fit%2Fu%3D282855164%2C259270705%26fm%3D214%26gp%3D0.jpg)

## 

## python环境配置及安装

\----------------------------------------------------------------------------------------------

### Linux系统的安装

你的计算机是什么操作系统的？自己先弄懂。然后安装适合的Python，因为Python就是跨平台的。

我用Ubuntu、deepin。

只要装了Ubuntu这个操作系统，默认里面就已经把Python安装好了。

接下来就在shell(终端)中输入Python，如果看到了>>>，并且显示出Python的版本信息，恭喜你，这就进入到了Python的交互模式下（“交互模式”，这是一个非常有用的东西，从后面的学习中，你就能体会到，这里是学习Python的主战场）。

如果非要自己安装。参考下面的操作：

到官方网站下载源码。比如：

wget http://www.python.org/ftp/python/2.7.8/Python-2.7.8.tgz

解压源码包

tar -zxvf Python-2.7.8.tgz

编译

cd Python-2.7.8

./configure --prefix=/usr/local                  #指定了目录，如果不制定，可以使用默认的，直接运行 ./configure 即可。 

make&&sudo make install

安装好之后，进入shell，输入python，会看到如下：

\--------------------------------------------------------------------------------------------------------------------------

h77@h77 ~> python

Python 2.7.12 (default, Nov 20 2017, 18:23:56)

[GCC 5.4.0 20160609] on linux2

Type "help", "copyright", "credits" or "license" for more information.

\>>>

\--------------------------------------------------------------------------------------------------------------------------

### windows系统的安装

### 工具/原料

\--------------------------------------------------------------------------------------------------------------------------

win10

Python2.7.X

\--------------------------------------------------------------------------------------------------------------------------

1、先到[python的官方网站](https://www.python.org/downloads/)下载软件，打开官网后，选择downlad项目，然后选择需要下载的大版本，2.7还是3.4下载Python软件.

2、选择完版本后，进入后一个页面，在这个页面可以选择操作系统及对应的版本，win下注意分64位和32位版本，不要下错了。

3、双击下载好的exe文件,一直下一步,直到安装完成。

### 环境变量配置

右击桌面上的“此电脑”—>“属性”—>“高级系统设置”—>右下角“环境变量”—>双击“系统变量”里的“Path”—>点击“新建”—>输入刚才的安装位置“C:\Python27;”，得到新建后的结果，然后一步步确定回去。

![这里写图片描述](https://images-blogger-opensocial.googleusercontent.com/gadgets/proxy?url=http%3A%2F%2Fimg.blog.csdn.net%2F20160626154619779&container=blogger&gadget=a&rewriteMime=image%2F*)

[![img](https://3.bp.blogspot.com/-D5sERQHYoNY/WiSziAJKcUI/AAAAAAAAAC8/-8vanFb1m-c-HC7DFkGG4W14Oek3lrfrACLcBGAs/s1600/TIM%25E6%2588%25AA%25E5%259B%25BE20171204103057.png)](https://3.bp.blogspot.com/-D5sERQHYoNY/WiSziAJKcUI/AAAAAAAAAC8/-8vanFb1m-c-HC7DFkGG4W14Oek3lrfrACLcBGAs/s1600/TIM%25E6%2588%25AA%25E5%259B%25BE20171204103057.png)

**win+R，cmd调出命令行，输入命令“python”，就可以有相关显示**

[![img](https://2.bp.blogspot.com/-zW2eHPFBP6g/WiS0BAa3sfI/AAAAAAAAADA/Yw0V3X0A7dYpnEQMPXI4Aq_BqKusY0vBQCLcBGAs/s1600/2.png)](https://2.bp.blogspot.com/-zW2eHPFBP6g/WiS0BAa3sfI/AAAAAAAAADA/Yw0V3X0A7dYpnEQMPXI4Aq_BqKusY0vBQCLcBGAs/s1600/2.png)

转载请注明：[Seven的博客](http://seven.github.io) » [点击阅读原文](https://sevenold.github.io/2016/06/Develop_Tool/)
