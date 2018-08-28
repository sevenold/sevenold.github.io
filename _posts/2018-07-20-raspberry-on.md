---
layout: post
title: "树莓派开机配置"
date: 2015-08-25 
description: "树莓派，一个卡片式电脑，那么它怎么玩呢？"

tag: 树莓派
---   

## 树莓派配置系统

[![img](https://ss3.bdstatic.com/70cFv8Sh_Q1YnxGkpoWK1HF6hhy/it/u=3136800288,1698373933&fm=27&gp=0.jpg)](https://ss3.bdstatic.com/70cFv8Sh_Q1YnxGkpoWK1HF6hhy/it/u=3136800288,1698373933&fm=27&gp=0.jpg)

### 购买硬件

直接在某宝搜索入手。必须内容：

\----------------------------------------------------------------------------------------------

树莓派一个（Raspberry Pi 3）

小usb口电源（5V2A的充电器随便找一个）

8G或者更大存储空间的SD卡一张（树莓派本身不带存储空间）

\----------------------------------------------------------------------------------------------

以下非必需：

\----------------------------------------------------------------------------------------------

散热器三片（风扇什么的觉得也太夸张了）

无线网卡（本身有网卡入口，所以不是必须的，但是做项目最好需要）

SD卡读卡器（安装系统的时候会用到）

\----------------------------------------------------------------------------------------------

### 下载系统

树莓派的[系统](http://wiki.nxez.com/rpi:list-of-oses)下载地址（http://wiki.nxez.com/rpi:list-of-oses）

**注意：**在选择系统时，请选择适合您的树莓派的系统，不适配的系统是用不了的。

### 烧录系统

软件准备：[树莓派烧录工具](http://sourceforge.net/projects/win32diskimager/files/Archive/win32diskimager-v0.9-binary.zip/download)（http://sourceforge.net/projects/win32diskimager/files/Archive/win32diskimager-v0.9-binary.zip/download）

SD卡格式化工具（[SDFormatter.exe](http://www.waveshare.net/w/upload/d/d7/Panasonic_SDFormatter.zip)）

#### 一、格式化内存卡

插上SD 卡到电脑，使用[SDFormatter.exe](http://www.waveshare.net/w/upload/d/d7/Panasonic_SDFormatter.zip)软件格式化SD 卡。

#### 二、烧写系统

用[Win32DiskImager.exe](http://www.waveshare.net/w/upload/7/76/Win32DiskImager.zip)烧写镜像。选择要烧写的镜像，点击“Write”进行烧写。

![img](https://images-blogger-opensocial.googleusercontent.com/gadgets/proxy?url=http%3A%2F%2Fwww.servethehome.com%2Fwp-content%2Fuploads%2F2013%2F03%2FWin32-Disk-Imager-Raspbian-Image-Selection-from-Synology-NAS.png&container=blogger&gadget=a&rewriteMime=image%2F*)

接下来就是等待-------一直到OK，系统就烧录完成了

#### 三、配置系统

经过前面两步我们的树莓派已经正常的工作起来了，但是在真正用它开发之前还需要进行一些列的配置以及软件的安装，这样开发起来才会得心应手。

配置选项

树莓派第一次使用的时候需要进行一个简单的配置，在命令行模式下运行

以下**命令**： $sudo raspi-config

新旧版本的配置界面不太一样，下面列举两种比较常见的：

**旧版本**

![img](https://images-blogger-opensocial.googleusercontent.com/gadgets/proxy?url=http%3A%2F%2Fwiki.jikexueyuan.com%2Fproject%2Fraspberry-pi%2Fimages%2Fold.png&container=blogger&gadget=a&rewriteMime=image%2F*)

**expand_rootfs** – 将根分区扩展到整张 SD 卡（树莓派默认不使用 SD 卡的全部空间，有一部分保留，建议选中）

**overscan** – 可以扩充或缩小屏幕（旧版不能自适应屏幕，新版没有这个选项，貌似可以自适应，没仔细研究）

**configure_keyboard** - 键盘配置界面

**change_pass** – 默认的用户名是 pi，密码是 raspberry，用ssh 远程连接或串口登录时要用到这个用户名和密码，这里可以更改密码。

**change_locale** – 更改语言设置。在 Locales to be generated: 中，选择 en_US.UTF-8 和 zh_CN.UTF-8。在 Default locale for the -system environment:中，选择 en_US.UTF-8（等启动完机器，装完中文字体，再改回 zh_CN.UTF-8，否则第一次启动会出现方块）。

**change_timezone** – 因为树莓派没有内部时钟，是通过网络获取的时间，选择 Asia – Shanghai。

**memory_split** – 配置给桌面显示的显存。

**ssh** – 是否激活 sshd 服务。

**boot_behaviour** – 设置启动时启动图形界面，正常肯定是 Yes。

**新版本**（比较新的镜像大部分是这个界面，做了不少改变）

![img](https://images-blogger-opensocial.googleusercontent.com/gadgets/proxy?url=http%3A%2F%2Fwiki.jikexueyuan.com%2Fproject%2Fraspberry-pi%2Fimages%2Fnew.jpg&container=blogger&gadget=a&rewriteMime=image%2F*)

1.Expand Filesystem 扩展文件系统（同旧版）。

2.Change User Password 改变默认 pi 用户的密码，按回车后输入 pi 用户的新密码。

\----------------------------------------------------------------------------------------------

3.Enable Boot to Desktop/Scratch 启动时进入的环境选择。

  Console Text console, requiring login(default)

  启动时进入字符控制台，需要进行登录（默认项）。

  Desktop log in as user 'pi' at the graphical desktop

  启动时进入 LXDE 图形界面的桌面。

  Scratch Start the Scratch programming environment upon boot

  启动时进入 Scratch 编程环境。

\----------------------------------------------------------------------------------------------

4.Internationalisation Options 国际化选项，可以更改默认语言

  Change Locale：语言和区域设置，建议不要改，默认英文就好。想改中文，最好选安装      中文字体再进行这步，安装中文字体的方法：sudo apt-get update sudo apt-get install    ttf-wqy-zenhei ttf-wqy-microhei

  移动到屏幕底部，用空格键选中 zh-CN GB2312,zh-CN GB18030,zh-CN UTF-8,然后按回    车，然后默认语言选中 zh-cn 然后回车。

  Change Timezone：设置时区，如果不进行设置，PI 的时间就显示不正常。选择         Asia（亚洲）再选择 Chongqing（重庆）即可。

  Change Keyboard Layout：改变键盘布局

\----------------------------------------------------------------------------------------------

5.Enable Camera

启动 PI 的摄像头模块，如果想启用，选择 Enable，禁用选择 Disable就行了。

6.Add to Rastrack

把你的 PI 的地理位置添加到一个全世界开启此选项的地图，建议还是不要开了，免得被跟踪。

\----------------------------------------------------------------------------------------------

7.Overclock

SN描述None 不超频，运行在 700Mhz，核心频率 250Mhz，内存频率 400Mhz，不增加电压

Modest 适度超频，运行在 800Mhz，核心频率 250Mhz，内存频率 400Mhz，不增加电压

Medium 中度超频，运行在 900Mhz，核心频率 250Mhz，内存频率 450Mhz，增加电压 2

High 高度超频，运行在 950Mhz，核心频率 250Mhz，内存频率 450Mhz，增加电压 6

Turbo 终极超频，运行在 1000Mhz，核心频率 500Mhz，内存频率 600Mhz，增加电压 6

\----------------------------------------------------------------------------------------------

8.Advanced Options 高级设置

SN标准描述A1 Overscan 是否让屏幕内容全屏显示

A2 Hostname 在网上邻居或者路由器能看到的主机名称

A3 Memory Split 内存分配，选择给 GPU 多少内存

A4 SSH 是否运行 SSH 登录，建议开户此选项，以后操作 PI 方便，有网络就行，不用开屏幕了

A5 SPI 是否默认启动 SPI 内核驱动，新手就不用管了

A6 Audio 选择声音默认输出到模拟口还是 HDMI 口

A7 Update 把 raspi-config 这个工具自动升级到最新版本

\----------------------------------------------------------------------------------------------

A6

0 Auto 自动选择

1 Force 3.5mm ('headphone') jack强制输出到3.5mm模拟口

2 Force HDMI 强制输出到HDMI

\----------------------------------------------------------------------------------------------

9.About raspi-config

关于 raspi-config 的信息。

\----------------------------------------------------------------------------------------------

最开始的简单配置系统就完成了。。

\----------------------------------------------------------------------------------------------

载请注明：[Seven的博客](http://sevenold.github.io) » [点击阅读原文](https://sevenold.github.io/2015/08/raspberry-on//)