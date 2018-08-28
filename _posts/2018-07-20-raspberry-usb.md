---
layout: post
title: "树莓派USB声卡配置"
date: 2015-08-25 
description: "卡片式电脑，音质不好怎么可以能，USB声卡来解决"

tag: 树莓派
---   

## 树莓派USB声卡配置

[![img](https://gd4.alicdn.com/imgextra/i4/2567133025/TB2XTcubHBnpuFjSZFGXXX51pXa_!!2567133025.jpg)](https://gd4.alicdn.com/imgextra/i4/2567133025/TB2XTcubHBnpuFjSZFGXXX51pXa_!!2567133025.jpg)

\--------------------------------------------------------------------------------------------------------------------------

​        首先在树莓派没有连接的USB声卡时，/ proc / asound / cards显示的只有树莓派默认的音频设备bcm2835 ALSA，连接USB声卡后，可以看到USB音频设备USB Audio Device.ALSA是Advanced Linux Sound Architecture的缩写，现在的版本的Linux的通常都提供对ALSA的支持。

\--------------------------------------------------------------------------------------------------------------------------

###  插上USB声卡:

查询声卡设备: cat /proc/asound/cards 

\--------------------------------------------------------------------------------------------------------------------------

pi@raspberrypi:~ $ cat /proc/asound/cards

0 [ALSA ]: bcm2835 - bcm2835 ALSA bcm2835 ALSA

1 [Device ]: USB-Audio - USB Audio Device C-Media Electronics Inc. USB Audio Device at usb-3f980000.usb-1.5, full speed

默认的是树莓派自带声卡,而我们外置的USB声卡作为备选项,我们需要让系统默认声卡设置为USB声卡.

\--------------------------------------------------------------------------------------------------------------------------

### 切换声卡:

编辑配置文件:sudo vim /lib/modprobe.d/aliases.conf

将 options snd-usb-audio index=-2 这行改为options snd-usb-audio index=0。

[![img](https://images-blogger-opensocial.googleusercontent.com/gadgets/proxy?url=http%3A%2F%2Fimage.newnius.com%2Fblog%2Fraspberry-pi-3b-change-usb-audoi-card-order-3.png&container=blogger&gadget=a&rewriteMime=image%2F*)](http://image.newnius.com/blog/raspberry-pi-3b-change-usb-audoi-card-order-3.png)

### 重启树莓派

查看USB声卡是否设为系统首选声卡

查看命令:cat /proc/asound/cards

[![img](https://images-blogger-opensocial.googleusercontent.com/gadgets/proxy?url=http%3A%2F%2Fimage.newnius.com%2Fblog%2Fraspberry-pi-3b-change-usb-audoi-card-order-2.png&container=blogger&gadget=a&rewriteMime=image%2F*)](http://image.newnius.com/blog/raspberry-pi-3b-change-usb-audoi-card-order-2.png)

此时USB声卡的编号应该和系统声卡成功调换顺序。

载请注明：[Seven的博客](http://sevenold.github.io) » [点击阅读原文](https://sevenold.github.io/2015/08/raspberry-usb/)