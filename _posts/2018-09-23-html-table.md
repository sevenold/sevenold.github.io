---
layout: post
title: "HTML第八话表格"
date: 2018-12-13
description: "html表格"
tag: HTML
---


### table常用标签

1、table标签：声明一个表格

2、tr标签：定义表格中的一行

3、td和th标签：定义一行中的一个单元格，td代表普通单元格，th表示表头单元格

### table常用属性：

1、border 定义表格的边框

2、cellpadding 定义单元格内内容与边框的距离

3、cellspacing 定义单元格与单元格之间的距离

4、align 设置单元格中内容的水平对齐方式,设置值有：left \| center \| right

5、valign 设置单元格中内容的垂直对齐方式 top \| middle \| bottom

6、colspan 设置单元格水平合并

7、rowspan 设置单元格垂直合并

### 传统布局：

传统的布局方式就是使用table来做整体页面的布局，布局的技巧归纳为如下几点：

1、定义表格宽高，将border、cellpadding、cellspacing全部设置为0

2、单元格里面嵌套表格

3、单元格中的元素和嵌套的表格用align和valign设置对齐方式

4、通过属性或者css样式设置单元格中元素的样式

### 传统布局目前应用：

1、快速制作用于演示的html页面

2、商业推广EDM制作(广告邮件)

### 表格常用样式属性

border-collapse:collapse 设置边框合并，制作一像素宽的边线的表格

转载请注明：[Seven的博客](http://sevenold.github.io)