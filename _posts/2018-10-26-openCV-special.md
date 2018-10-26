---
layout: post
title: "计算机视觉-openCV图片特效&绘制线段文字"
date: 2018-10-26
description: "opencv安装、计算机视觉"
tag: OpenCV
---

### openCV图片灰度处理一

```python
import cv2  # 导入cv库
img0 = cv2.imread('image0.jpg',0)
img1 = cv2.imread('image0.jpg',1)
print(img0.shape)
print(img1.shape)
cv2.imshow('src',img0)
```

![1](/images/cv/14.png)

### openCV图片灰度处理二

```python
import cv2
img = cv2.imread('image0.jpg',1)
dst = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)# 颜色空间转换 1 data 2 BGR gray
cv2.imshow('dst',dst)
```

![1](/images/cv/15.png)

### openCV图片灰度处理三

```python
import cv2
import numpy as np
img = cv2.imread('image0.jpg',1)
imgInfo = img.shape
height = imgInfo[0]
width = imgInfo[1]
# RGB R=G=B = gray  (R+G+B)/3
dst = np.zeros((height,width,3),np.uint8)
for i in range(0,height):
    for j in range(0,width):
        (b,g,r) = img[i,j]
        gray = (int(b)+int(g)+int(r))/3
        dst[i,j] = np.uint8(gray)
cv2.imshow('dst',dst)
```

![1](/images/cv/16.png)

### openCV图片灰度处理四

```python
#方法4 gray = r*0.299+g*0.587+b*0.114
import cv2
import numpy as np
img = cv2.imread('image0.jpg',1)
imgInfo = img.shape
height = imgInfo[0]
width = imgInfo[1]
dst = np.zeros((height,width,3),np.uint8)
for i in range(0,height):
    for j in range(0,width):
        (b,g,r) = img[i,j]
        b = int(b)
        g = int(g)
        r = int(r)
        gray = r*0.299+g*0.587+b*0.114
        dst[i,j] = np.uint8(gray)
cv2.imshow('dst',dst)

```

![1](/images/cv/17.png)

### openCV图片灰度处理五-算法优化版

```python
# r*0.299+g*0.587+b*0.114
import cv2
import numpy as np
img = cv2.imread('image0.jpg',1)
imgInfo = img.shape
height = imgInfo[0]
width = imgInfo[1]
# RGB R=G=B = gray  (R+G+B)/3
dst = np.zeros((height,width,3),np.uint8)
for i in range(0,height):
    for j in range(0,width):
        (b,g,r) = img[i,j]
        b = int(b)
        g = int(g)
        r = int(r)
        #
        #gray = (r*1+g*2+b*1)/4
        gray = (r+(g<<1)+b)>>2
        dst[i,j] = np.uint8(gray)
cv2.imshow('dst',dst)
```

![2](/images/cv/18.png)

### openCV图片颜色反转-灰色

```python
import cv2
import numpy as np
img = cv2.imread('image0.jpg',1)
imgInfo = img.shape
height = imgInfo[0]
width = imgInfo[1]
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) # 原始图片，灰度api
dst = np.zeros((height,width,1),np.uint8) # 反转矩阵
for i in range(0,height):
    for j in range(0,width):
        grayPixel = gray[i,j]  
        dst[i,j] = 255-grayPixel
cv2.imshow('dst',dst)
```

![2](/images/cv/19.png)

### openCV图片颜色反转-彩色

```python
import cv2
import numpy as np
img = cv2.imread('image0.jpg',1)
imgInfo = img.shape
height = imgInfo[0]
width = imgInfo[1]
dst = np.zeros((height,width,3),np.uint8)
for i in range(0,height):
    for j in range(0,width):
        (b,g,r) = img[i,j]
        dst[i,j] = (255-b,255-g,255-r)
cv2.imshow('dst',dst)
```

![3](/images/cv/20.png)



### openCV图片马赛克

```python
import cv2 # 导入cv库
import numpy as np
img = cv2.imread('image0.jpg',1) # 读取图片文件， 1：彩色， 0：灰色
cv2.imshow('src',img)
imgInfo = img.shape  # 获取图片的维度
height = imgInfo[0]
width = imgInfo[1]
for m in range(100,300):
    for n in range(100,200):
        # pixel ->10*10
        if m%10 == 0 and n%10==0:
            for i in range(0,10):
                for j in range(0,10):
                    (b,g,r) = img[m,n]
                    img[i+m,j+n] = (b,g,r)
cv2.imshow('dst',img)
```

![4](/images/cv/21.png)

### openCV图片毛玻璃

```python
import cv2 # 导入cv库
import numpy as np
img = cv2.imread('image0.jpg',1) # 读取图片文件， 1：彩色， 0：灰色
cv2.imshow('src',img)
imgInfo = img.shape  # 获取图片的维度
height = imgInfo[0]
width = imgInfo[1]
dst = np.zeros((height,width,3),np.uint8)
mm = 8
for m in range(0,height-mm):
    for n in range(0,width-mm):
        index = int(random.random()*8)#0-8
        (b,g,r) = img[m+index,n+index]
        dst[m,n] = (b,g,r)
cv2.imshow('dst',dst)
```

![4](/images/cv/22.png)



### openCV图片融合

```python
# dst  = src1*a+src2*(1-a)
import cv2
import numpy as np
img0 = cv2.imread('image0.jpg',1)
img1 = cv2.imread('image1.jpg',1)
imgInfo = img0.shape
height = imgInfo[0]
width = imgInfo[1]
# ROI
roiH = int(height/2)
roiW = int(width/2)
img0ROI = img0[0:roiH,0:roiW]
img1ROI = img1[0:roiH,0:roiW]
# dst
dst = np.zeros((roiH,roiW,3),np.uint8)
dst = cv2.addWeighted(img0ROI,0.5,img1ROI,0.5,0)#add src1*a+src2*(1-a)
# 1 src1 2 a 3 src2 4 1-a
cv2.imshow('dst',dst)
```

![4](/images/cv/23.png)

### openCV边缘检测一

```python
import cv2
import numpy as np
import random
img = cv2.imread('image0.jpg',1)
imgInfo = img.shape
height = imgInfo[0]
width = imgInfo[1]
cv2.imshow('src',img)
#canny 1 gray 2 高斯 3 canny 
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
imgG = cv2.GaussianBlur(gray,(3,3),0)
dst = cv2.Canny(img,50,50) #图片卷积——》th
cv2.imshow('dst',dst)
```

![4](/images/cv/24.png)

### openCV边缘检测二

```python
import cv2
import numpy as np
import random
import math
img = cv2.imread('image0.jpg',1)
imgInfo = img.shape
height = imgInfo[0]
width = imgInfo[1]
cv2.imshow('src',img)
# sobel 1 算子模版 2 图片卷积 3 阈值判决 
# [1 2 1          [ 1 0 -1
#  0 0 0            2 0 -2
# -1 -2 -1 ]       1 0 -1 ]
              
# [1 2 3 4] [a b c d] a*1+b*2+c*3+d*4 = dst
# sqrt(a*a+b*b) = f>th
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
dst = np.zeros((height,width,1),np.uint8)
for i in range(0,height-2):
    for j in range(0,width-2):
        gy = gray[i,j]*1+gray[i,j+1]*2+gray[i,j+2]*1-gray[i+2,j]*1-gray[i+2,j+1]*2-gray[i+2,j+2]*1
        gx = gray[i,j]+gray[i+1,j]*2+gray[i+2,j]-gray[i,j+2]-gray[i+1,j+2]*2-gray[i+2,j+2]
        grad = math.sqrt(gx*gx+gy*gy)
        if grad>50:
            dst[i,j] = 255
        else:
            dst[i,j] = 0
cv2.imshow('dst',dst)
```

![4](/images/cv/25.png)

### openCV图像浮雕风格