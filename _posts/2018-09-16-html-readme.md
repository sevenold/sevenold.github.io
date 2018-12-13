---
layout: post
title: "HTML第一话概述和基本框架"
date: 2018-12-13
description: "html概述和基本结构"
tag: HTML
---

### html概述

HTML是 HyperText Mark-up Language 的首字母简写，意思是超文本标记语言，超文本指的是超链接，标记指的是标签，是一种用来制作网页的语言，这种语言由一个个的标签组成，用这种语言制作的文件保存的是一个文本文件，文件的扩展名为html或者htm，一个html文件就是一个网页，html文件用编辑器打开显示的是文本，可以用文本的方式编辑它，如果用浏览器打开，浏览器会按照标签描述内容将文件渲染成网页，显示的网页可以从一个网页链接跳转到另外一个网页。

### html基本结构

一个html的基本结构如下：

```
<!DOCTYPE html>
<html lang="en">
    <head>            
        <meta charset="UTF-8">
        <title>网页标题</title>
    </head>
    <body>
          网页显示内容
    </body>
</html>
```

第一行是文档声明，第二行“<html>”标签和最后一行“</html>”定义html文档的整体，“<html>”标签中的‘lang=“en”’定义网页的语言为英文，定义成中文是'lang="zh-CN"',不定义也没什么影响，它一般作为分析统计用。 “<head>”标签和“<body>”标签是它的第一层子元素，“<head>”标签里面负责对网页进行一些设置以及定义标题，设置包括定义网页的编码格式，外链css样式文件和javascript文件等，设置的内容不会显示在网页上，标题的内容会显示在标题栏，“<body>”内编写网页上显示的内容。

### HTML文档类型

目前常用的两种文档类型是xhtml 1.0和html5

##### xhtml 1.0

xhtml 1.0 是html5之前的一个常用的版本，目前许多网站仍然使用此版本。
此版本文档用sublime text创建方法： html:xt + tab
文档示例：

```html
<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Transitional//EN" "http://www.w3.org/TR/xhtml1/DTD/xhtml1-transitional.dtd">
<html xmlns="http://www.w3.org/1999/xhtml" xml:lang="en">
<head>
    <meta http-equiv="Content-Type" content="text/html;charset=UTF-8">
    <title> xhtml 1.0 文档类型 </title>
</head>
<body>

</body>
</html>
```

##### html5

pc端可以使用xhtml 1.0，也可以使用html5，html5是向下兼容的
此版本文档用sublime text创建方法： html:5 + tab 或者 ! + tab
文档示例：

```
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title> html5文档类型 </title>
</head>
<body>

</body>
</html>
```

### 两种文档的区别

1、文档声明和编码声明

2、html5新增了标签元素以及元素属性

### html文档规范

xhtml制定了文档的编写规范，html5可部分遵守，也可全部遵守，看开发要求。

1、所有的标签必须小写

2、所有的属性必须用双引号括起来

3、所有标签必须闭合

4、img必须要加alt属性(对图片的描述)

### html注释：

html文档代码中可以插入注释，注释是对代码的说明和解释，注释的内容不会显示在页面上，html代码中插入注释的方法是：

```html
<!-- 这是一段注释  -->
```

### html标签特点：

html的标签大部分是成对出现的,少量是单个出现的，特定标签之间可以相互嵌套，嵌套就是指一个标签里面可以包含一个或多个其他的标签，包含的标签和父标签可以是同类型的，也可以是不同类型的：

```
<!-- 成对出现的标签  -->
<body>......</body>
<p>......</p>
<div>......</div>
<b>......</b>

<!-- 单个出现的标签  -->
<br />
<img src="..." />
<input type="..." />

<!-- 标签之间的嵌套  -->
<p>
    <span>...</span>
    <a href="...">...</a>
</p>
<div>
      <h3>...</h3>
      <div>
              <span>...</span>
              <p>...</p>
      </div>
</div>
```

转载请注明：[Seven的博客](http://sevenold.github.io)