---
layout: post
title: "Python图像处理-Pillow"
date: 2018-11-21
description: "Python图像处理-Pillow"
tag: python
---

转载自：简书-[王南北](https://www.jianshu.com/u/948da055a416)

本文地址：https://www.jianshu.com/p/3740dec1f436

------

## 简介

Python传统的图像处理库`PIL`(Python Imaging Library )，可以说基本上是Python处理图像的标准库，功能强大，使用简单。

但是由于`PIL`不支持Python3，而且更新缓慢。所以有志愿者在`PIL`的基础上创建了一个分支版本，命名为`Pillow`，`Pillow`目前最新支持到python3.6，更新活跃，并且增添了许多新的特性。所以我们安装Pillow即可。

------

## 安装

`Pillow`的安装比较的简单，直接pip安装即可：

```python
pip install Pillow
```

但是要注意的一点是，`Pillow`和`PIL`不能共存在同一个环境中，所以如果安装的有`PIL`的话，那么安装`Pillow`之前应该删除`PIL`。

由于是继承自`PIL`的分支，所以`Pillow`的导入是这样的：

```python
import PIL 
# 或者
from PIL import Image
```

------

## 使用手册

### Image

`Image`是Pillow中最为重要的类，实现了Pillow中大部分的功能。要创建这个类的实例主要有三个方式：

1. 从文件加载图像
2. 处理其他图像获得
3. 创建一个新的图像

#### 读取图像

一般来说，我们都是都过从文件加载图像来实例化这个类，如下所示：

```python
from PIL import Image
picture = Image.open('happy.png')
```

如果没有指定图片格式的话，那么`Pillow`会自动识别文件内容为文件格式。

#### 新建图像

`Pillow`新建空白图像使用`new()`方法， 第一个参数是mode即颜色空间模式，第二个参数指定了图像的分辨率(宽x高)，第三个参数是颜色。

- 可以直接填入常用颜色的名称。如'red'。
- 也可以填入十六进制表示的颜色，如`#FF0000`表示红色。
- 还能传入元组，比如(255, 0, 0, 255)或者(255， 0， 0)表示红色。 

```python
picture = Image.new('RGB', (200, 100), 'red')
```

#### 保存图像

保存图片的话需要使用`save()`方法：

```python
picture.save('happy.png')
```

保存的时候，如果没有指定图片格式的话，那么`Pillow`会根据输入的后缀名决定保存的文件格式。

------

### 图像的坐标表示

在Pillow中，用的是图像的**左上角**为坐标的原点（0，0），所以这意味着，x轴的数值是从左到右增长的，y轴的数值是从上到下增长的。

我们处理图像时，常常需要去表示一个矩形的图像区域。`Pillow`中很多方法都需要传入一个表示矩形区域的元祖参数。

这个元组参数包含四个值，分别代表矩形四条边的距离X轴或者Y轴的距离。顺序是`(左，顶，右，底)`。其实就相当于，矩形的左上顶点坐标为`(左，顶)`，矩形的右下顶点坐标为`(右，底)`，两个顶点就可以确定一个矩形的位置。

右和底坐标稍微特殊，跟python列表索引规则一样，是左闭又开的。可以理解为`[左, 右)`和`[顶， 底)`这样左闭右开的区间。比如(3, 2, 8, 9)就表示了横坐标范围[3, 7]；纵坐标范围[2, 8]的矩形区域。 

------

### 常用属性

- `PIL.Image.filename`

  图像源文件的文件名或者路径，只有使用`open()`方法创建的对象有这个属性。

  类型：字符串

- `PIL.Image.format`

  图像源文件的文件格式。

- `PIL.Image.mode`

  图像的模式，一般来说是“1”, “L”, “RGB”, 或者“CMYK” 。

- `PIL.Image.size`

  图像的大小

- `PIL.Image.width`

  图像的宽度

- `PIL.Image.height`

  图像的高度

- `PIL.Image.info`

  图像的一些信息，为字典格式

------

### 常用方法

#### 裁剪图片

`Image`使用`crop()`方法来裁剪图像，此方法需要传入一个矩形元祖参数，返回一个新的`Image`对象，对原图没有影响。

```python
croped_im = im.crop((100, 100, 200, 200))
```

#### 复制与粘贴图像

复制图像使用`copy()`方法：

```python
copyed_im = im.copy()
```

粘贴图像使用`paste()`方法：

```python
croped_im = im.crop((100, 100, 200, 200))
im.paste(croped_im, (0, 0))
```

im对象调用了`paste()`方法，第一个参数是被裁剪下来用来粘贴的图像，第二个参数是一个位置参数元祖，这个位置参数是粘贴的图像的左顶点。

#### 调整图像的大小

调整图像大小使用`resize()`方法：

```python
resized_im = im.resize((width, height))
```

`resize()`方法会返回一个重设了大小的`Image`对象。

#### 旋转图像和翻转图像

旋转图像使用`rotate()`方法，此方法按逆时针旋转，并返回一个新的`Image`对象：

```python
# 逆时针旋转90度
im.rotate(90)
im.rotate(180)
im.rotate(20, expand=True)
```

旋转的时候，会将图片超出边界的边角裁剪掉。如果加入`expand=True`参数，就可以将图片边角保存住。

翻转图像使用`transpose()`：

```python
# 水平翻转
im.transpose(Image.FLIP_LEFT_RIGHT)
# 垂直翻转
im.transpose(Image.FLIP_TOP_BOTTOM)
```

#### 获取单个像素的值

使用`getpixel`(*xy*)方法可以获取单个像素位置的值：

```python
im.getpixel((100, 100))
```

传入的xy需要是一个元祖形式的坐标。

如果图片是多通道的，那么返回的是一个元祖。

#### 通过通道分割图片

##### split()

`split()`可以将多通道图片按通道分割为单通道图片：

```python
R, G, B = im.split()
```

`split()`方法返回的是一个元祖，元祖中的元素则是分割后的单个通道的值。

##### getchannel(channel)

`getchannel()`可以获取单个通道的数据：

```python
R = im.getchannel("R")
```

#### 加载图片全部数据

我们可以使用`load()`方法加载图片所有的数据，并比较方便的修改像素的值：

```python
pixdata = im.load()
pixdata[100,200] = 255
```

此方法返回的是一个`PIL.PyAccess`，可以通过这个类的索引来对指定坐标的像素点进行修改。

#### 关闭图片并释放内存

此方法会删除图片对象并释放内存

```python
im.close()
```



转载请注明：[Seven的博客](http://sevenold.github.io)