---
layout: post
title: "计算机视觉-openCV安装及入门"
date: 2018-10-26
description: "opencv安装、计算机视觉"
tag: OpenCV
---

### openCV-python的安装

#### windows安装：下载地址：[opencv](https://www.lfd.uci.edu/~gohlke/pythonlibs/).

```shell
pip install  opencv_python-3.4.3-cp36-cp36m-win_amd64.whl
```

#### ubuntu安装

```shell
pip3 install opencv-python
```



### openCV图片读取与展示

```python
import cv2  # 导入cv库

img = cv2.imread('image0.jpg',1)  # 读取图片文件， 1：彩色， 0：灰色
cv2.imshow('image',img)  # 显示图片

```

![1](/images/cv/1.png)

### openCV图片写入

```python
import cv2 # 导入cv库
img = cv2.imread('image0.jpg',1) # 读取图片文件， 1：彩色， 0：灰色
cv2.imwrite('image1.jpg',img) # 写入文件名字 ， 图片数据 
```

### openCV更改图像质量-有损压缩

```python
import cv2 # 导入cv库
img = cv2.imread('image0.jpg',1) # 读取图片文件， 1：彩色， 0：灰色
cv2.imwrite('imageTest.jpg',img,[cv2.IMWRITE_JPEG_QUALITY,50]) # 写入文件名字 ， 图片数据 ， 当前jpg图片保存的质量（范围0-100）
#1M 100k 10k 0-100 有损压缩
```

![2](/images/cv/2.png)

### openCV更改图像质量-无损压缩

```python
# 1 无损 2 透明度属性
import cv2  # 导入cv库
img = cv2.imread('image0.jpg',1)  # 读取图片文件， 1：彩色， 0：灰色
cv2.imwrite('imageTest.png',img,[cv2.IMWRITE_PNG_COMPRESSION,0])  # 写入文件名字 ， 图片数据 ， 当前jpg图片保存的质量（范围0-100）
# jpg 0 压缩比高0-100 png 0 压缩比低0-9
```

![3](/images/cv/3.png)



### openCV图片像素操作

```python
import cv2  # 导入cv库 
img = cv2.imread('image0.jpg',1)  # 读取图片文件， 1：彩色， 0：灰色
(b,g,r) = img[100,100] # 获取图片的（100,100）坐标的像素值，按照bgr的形式读取
print(b,g,r)# bgr
#10 100 --- 110 100
for i in range(1,100):  # 总共一百个像素点
    img[10+i,100] = (255,0,0)  # 写入标准的蓝色
cv2.imshow('image',img)
```

![4](/images/cv/4.png)

