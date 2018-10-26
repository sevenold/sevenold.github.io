---
layout: post
title: "计算机视觉-openCV图片几何变换"
date: 2018-10-26
description: "opencv安装、计算机视觉"
tag: OpenCV
---

### openCV图片缩放一

```python
import cv2  # 导入cv库
img = cv2.imread('image0.jpg',1)  # 读取图片文件， 1：彩色， 0：灰色
imgInfo = img.shape # 获取图片的维度
print(imgInfo)
height = imgInfo[0] 
width = imgInfo[1]
mode = imgInfo[2]
# 1 放大 缩小 2 等比例 非 2:3 
dstHeight = int(height*0.5)
dstWidth = int(width*0.5)
#最近临域插值 双线性插值 像素关系重采样 立方插值
dst = cv2.resize(img,(dstWidth,dstHeight))
cv2.imshow('image',dst)
```

![1](/images/cv/5.png)

### openCV图片缩放二

```python
import cv2  # 导入cv库
import numpy as np 
img = cv2.imread('image0.jpg',1)  # 读取图片文件， 1：彩色， 0：灰色
imgInfo = img.shape # 获取图片的维度
height = imgInfo[0]
width = imgInfo[1]
# 定义目标图片的高度和宽度
dstHeight = int(height/2)
dstWidth = int(width/2)
dstImage = np.zeros((dstHeight,dstWidth,3),np.uint8) # 0-255 准备好缩放后的图片数据维度
# 计算新的目标图片的坐标
for i in range(0,dstHeight):#行
    for j in range(0,dstWidth):#列 
        iNew = int(i*(height*1.0/dstHeight)) 
        jNew = int(j*(width*1.0/dstWidth))
        dstImage[i,j] = img[iNew,jNew]
cv2.imshow('dst',dstImage)
```

![1](/images/cv/6.png)

### openCV图片剪切

```python
import cv2  # 导入cv库
img = cv2.imread('image0.jpg',1)  # 读取图片文件， 1：彩色， 0：灰色
imgInfo = img.shape
dst = img[100:200,100:300] # 获取宽度100-200， 高度100-300的图像
cv2.imshow('image',dst)
```

![1](/images/cv/7.png)

### openCV图片读取与展示

```python
import cv2  # 导入cv库

img = cv2.imread('image0.jpg',1)  # 读取图片文件， 1：彩色， 0：灰色
cv2.imshow('image',img)  # 显示图片

```

![1](/images/cv/1.png)

### openCV图片移位一

```python
import cv2 # 导入cv库
import numpy as np
img = cv2.imread('image0.jpg',1) # 读取图片文件， 1：彩色， 0：灰色
cv2.imshow('src',img)
imgInfo = img.shape  # 获取图片的维度
height = imgInfo[0]
width = imgInfo[1]
####
matShift = np.float32([[1,0,100],[0,1,200]])# 2*3  设置平移的矩阵
dst = cv2.warpAffine(img,matShift,(height,width))#图片数据 ，移位矩阵 图片的维度信息
# 移位 矩阵
cv2.imshow('dst',dst)
```

![2](/images/cv/8.png)

### openCV图片移位二

```python
import cv2 # 导入cv库
import numpy as np
img = cv2.imread('image0.jpg',1) # 读取图片文件， 1：彩色， 0：灰色
cv2.imshow('src',img)
imgInfo = img.shape  # 获取图片的维度
dst = np.zeros(img.shape,np.uint8)
height = imgInfo[0]
width = imgInfo[1]
for i in range(0,height):
    for j in range(0,width-100):
        dst[i,j+100]=img[i,j]
cv2.imshow('image',dst)
```

![2](/images/cv/9.png)

### openCV图片镜像

```python
import cv2 # 导入cv库
import numpy as np
img = cv2.imread('image0.jpg',1) # 读取图片文件， 1：彩色， 0：灰色
cv2.imshow('src',img)
imgInfo = img.shape  # 获取图片的维度
height = imgInfo[0]
width = imgInfo[1]
deep = imgInfo[2]
newImgInfo = (height*2,width,deep) # 新图片的维度
dst = np.zeros(newImgInfo,np.uint8)#uint8 # 目标图片的数据维度
# 刷新图片的数据
for i in range(0,height):
    for j in range(0,width):
        dst[i,j] = img[i,j]
        #x y = 2*h - y -1
        dst[height*2-i-1,j] = img[i,j]
for i in range(0,width): # 添加分割线
    dst[height,i] = (0,0,255)#BGR
cv2.imshow('dst',dst)
```

![3](/images/cv/10.png)



### openCV图片缩放

```python
import cv2 # 导入cv库
import numpy as np
img = cv2.imread('image0.jpg',1) # 读取图片文件， 1：彩色， 0：灰色
cv2.imshow('src',img)
imgInfo = img.shape  # 获取图片的维度
height = imgInfo[0]
width = imgInfo[1]
matScale = np.float32([[0.5,0,0],[0,0.5,0]]) # 定义缩放矩阵
dst = cv2.warpAffine(img,matScale,(int(width/2),int(height/2))) # 原始数据，缩放矩阵，目标的宽高信息
cv2.imshow('dst',dst)
```

![4](/images/cv/11.png)

### openCV图片仿射变换

```python
import cv2 # 导入cv库
import numpy as np
img = cv2.imread('image0.jpg',1) # 读取图片文件， 1：彩色， 0：灰色
cv2.imshow('src',img)
imgInfo = img.shape  # 获取图片的维度
height = imgInfo[0]
width = imgInfo[1]
#src 3->dst 3 (左上角 左下角 右上角)
matSrc = np.float32([[0,0],[0,height-1],[width-1,0]]) # 获取原图片三个点坐标
matDst = np.float32([[50,50],[300,height-200],[width-300,100]]) # 三个点的新坐标
#把两个矩阵组合
matAffine = cv2.getAffineTransform(matSrc,matDst) # 获取矩阵的组合，
dst = cv2.warpAffine(img,matAffine,(width,height)) # 仿射变换方法
cv2.imshow('dst',dst)
```

![4](/images/cv/12.png)



### openCV图片旋转

```python
import cv2 # 导入cv库
import numpy as np
img = cv2.imread('image0.jpg',1) # 读取图片文件， 1：彩色， 0：灰色
cv2.imshow('src',img)
imgInfo = img.shape  # 获取图片的维度
height = imgInfo[0]
width = imgInfo[1]
# 2*3 定义旋转矩阵--旋转的中心点，旋转的角度， 缩放系数
matRotate = cv2.getRotationMatrix2D((height*0.5,width*0.5),45,1)# mat rotate 1 center 2 angle 3 scale
#100*100 25 
dst = cv2.warpAffine(img,matRotate,(height,width)) # 仿射方法
cv2.imshow('dst',dst)
```

![4](/images/cv/13.png)