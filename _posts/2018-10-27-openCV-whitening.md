---
layout: post
title: "计算机视觉-openCV图像美化"
date: 2018-10-27
description: "opencv安装、计算机视觉"
tag: OpenCV
---

### openCV彩色图片直方图

```python
import cv2
import numpy as np
def ImageHist(image,type):
    color = (255,255,255)
    windowName = 'Gray'
    if type == 31:
        color = (255,0,0)
        windowName = 'B Hist'
    elif type == 32:
        color = (0,255,0)
        windowName = 'G Hist'
    elif type == 33:
        color = (0,0,255)
        windowName = 'R Hist'
    # 1 image 2 [0] 3 mask None 4 256 5 0-255
    hist = cv2.calcHist([image],[0],None,[256],[0.0,255.0])
    minV,maxV,minL,maxL = cv2.minMaxLoc(hist)  # 获取像素的最大值和最小值以便后面归一化处理
    histImg = np.zeros([256,256,3],np.uint8)
    # 数据归一化处理
    for h in range(256):
        intenNormal = int(hist[h]*256/maxV)
        cv2.line(histImg,(h,256),(h,256-intenNormal),color)
    cv2.imshow(windowName,histImg)
    return histImg
img = cv2.imread('image0.jpg',1)
channels = cv2.split(img)# RGB - R G B
for i in range(0,3):
    ImageHist(channels[i],31+i)
```

![1](/images/cv/33.png)

### openCV直方图均衡化-灰度

```python
import cv2
import numpy as np
img = cv2.imread('image0.jpg',1)
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) # 图片灰度化
cv2.imshow('src',gray)
dst = cv2.equalizeHist(gray) # api 完成直方图均衡化
cv2.imshow('dst',dst)
```

![1](/images/cv/34.png)

### openCV直方图均衡化-彩色

```python
import cv2
import numpy as np
img = cv2.imread('image0.jpg',1)
cv2.imshow('src',img)
(b,g,r) = cv2.split(img) # 通道分解
# 图片单通道处理
bH = cv2.equalizeHist(b)
gH = cv2.equalizeHist(g)
rH = cv2.equalizeHist(r)
result = cv2.merge((bH,gH,rH))# 通道合成
cv2.imshow('dst',result)
```

![1](/images/cv/35.png)

### openCV直方图均衡化-YUV

```python
import cv2
import numpy as np
img = cv2.imread('image0.jpg',1)
imgYUV = cv2.cvtColor(img,cv2.COLOR_BGR2YCrCb) #  
cv2.imshow('src',img)
channelYUV = cv2.split(imgYUV) # 图片分解
channelYUV[0] = cv2.equalizeHist(channelYUV[0]) # 直方图均衡化
channels = cv2.merge(channelYUV) # 合成
result = cv2.cvtColor(channels,cv2.COLOR_YCrCb2BGR)
cv2.imshow('dst',result)
```

![1](/images/cv/36.png)

### openCV图片修补

```python
import cv2 
import numpy as np
img = cv2.imread('damaged.jpg',1)
cv2.imshow('src',img)
imgInfo = img.shape
height = imgInfo[0]
width = imgInfo[1]
paint = np.zeros((height,width,1),np.uint8)
# 描绘图片坏的数组
for i in range(200,300):
    paint[i,200] = 255
    paint[i,200+1] = 255
    paint[i,200-1] = 255
for i in range(150,250):
    paint[250,i] = 255
    paint[250+1,i] = 255
    paint[250-1,i] = 255
cv2.imshow('paint',paint)
#1 src 2 mask
imgDst = cv2.inpaint(img,paint,3,cv2.INPAINT_TELEA)

cv2.imshow('image',imgDst)
```

![2](/images/cv/37.png)

### openCV灰度直方图

```python
# 本质：统计每个像素灰度 出现的概率 0-255 p
import cv2
import numpy as np
import matplotlib.pyplot as plt
img = cv2.imread('image0.jpg',1)
imgInfo = img.shape
height = imgInfo[0]
width = imgInfo[1]
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) # 图片灰度化
count = np.zeros(256,np.float)
# 获取灰度像素
for i in range(0,height):
    for j in range(0,width):
        pixel = gray[i,j]
        index = int(pixel)
        count[index] = count[index]+1
# 灰度等级出现的概率
for i in range(0,255):
    count[i] = count[i]/(height*width)
x = np.linspace(0,255,256)
y = count
plt.bar(x,y,0.9,alpha=1,color='b')
plt.show()
```

![2](/images/cv/38.png)

### openCV彩色直方图

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt
img = cv2.imread('image0.jpg',1)
imgInfo = img.shape
height = imgInfo[0]
width = imgInfo[1]

count_b = np.zeros(256,np.float)
count_g = np.zeros(256,np.float)
count_r = np.zeros(256,np.float)
for i in range(0,height):
    for j in range(0,width):
        (b,g,r) = img[i,j]
        index_b = int(b)
        index_g = int(g)
        index_r = int(r)
        count_b[index_b] = count_b[index_b]+1
        count_g[index_g] = count_g[index_g]+1
        count_r[index_r] = count_r[index_r]+1
for i in range(0,256):
    count_b[i] = count_b[i]/(height*width)
    count_g[i] = count_g[i]/(height*width)
    count_r[i] = count_r[i]/(height*width)
x = np.linspace(0,255,256)
y1 = count_b
plt.figure()
plt.bar(x,y1,0.9,alpha=1,color='b')
y2 = count_g
plt.figure()
plt.bar(x,y2,0.9,alpha=1,color='g')
y3 = count_r
plt.figure()
plt.bar(x,y3,0.9,alpha=1,color='r')
plt.show()
```

![3](/images/cv/39.png)



### openCV灰度直方图均衡化

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt
img = cv2.imread('image0.jpg',1)


imgInfo = img.shape
height = imgInfo[0]
width = imgInfo[1]
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
cv2.imshow('src',gray)
count = np.zeros(256,np.float)
for i in range(0,height):
    for j in range(0,width):
        pixel = gray[i,j]
        index = int(pixel)
        count[index] = count[index]+1
for i in range(0,255):
    count[i] = count[i]/(height*width)
#计算累计概率
sum1 = float(0)
for i in range(0,256):
    sum1 = sum1+count[i]
    count[i] = sum1
#print(count)
# 计算映射表
map1 = np.zeros(256,np.uint16)
for i in range(0,256):
    map1[i] = np.uint16(count[i]*255)
# 映射
for i in range(0,height):
    for j in range(0,width):
        pixel = gray[i,j]
        gray[i,j] = map1[pixel]
cv2.imshow('dst',gray)
```

![4](/images/cv/40.png)

### openCV彩色直方图均衡化

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt
img = cv2.imread('image0.jpg',1)
cv2.imshow('src',img)

imgInfo = img.shape
height = imgInfo[0]
width = imgInfo[1]

count_b = np.zeros(256,np.float)
count_g = np.zeros(256,np.float)
count_r = np.zeros(256,np.float)
for i in range(0,height):
    for j in range(0,width):
        (b,g,r) = img[i,j]
        index_b = int(b)
        index_g = int(g)
        index_r = int(r)
        count_b[index_b] = count_b[index_b]+1
        count_g[index_g] = count_g[index_g]+1
        count_r[index_r] = count_r[index_r]+1
for i in range(0,255):
    count_b[i] = count_b[i]/(height*width)
    count_g[i] = count_g[i]/(height*width)
    count_r[i] = count_r[i]/(height*width)
#计算累计概率
sum_b = float(0)
sum_g = float(0)
sum_r = float(0)
for i in range(0,256):
    sum_b = sum_b+count_b[i]
    sum_g = sum_g+count_g[i]
    sum_r = sum_r+count_r[i]
    count_b[i] = sum_b
    count_g[i] = sum_g
    count_r[i] = sum_r
#print(count)
# 计算映射表
map_b = np.zeros(256,np.uint16)
map_g = np.zeros(256,np.uint16)
map_r = np.zeros(256,np.uint16)
for i in range(0,256):
    map_b[i] = np.uint16(count_b[i]*255)
    map_g[i] = np.uint16(count_g[i]*255)
    map_r[i] = np.uint16(count_r[i]*255)
# 映射
dst = np.zeros((height,width,3),np.uint8)
for i in range(0,height):
    for j in range(0,width):
        (b,g,r) = img[i,j]
        b = map_b[b]
        g = map_g[g]
        r = map_r[r]
        dst[i,j] = (b,g,r)
cv2.imshow('dst',dst)
```

![4](/images/cv/41.png)



### openCV亮度增强

```python
import cv2
import numpy as np
img = cv2.imread('image0.jpg',1)
imgInfo = img.shape
height = imgInfo[0]
width = imgInfo[1]
cv2.imshow('src',img)
dst = np.zeros((height,width,3),np.uint8)
for i in range(0,height):
    for j in range(0,width):
        (b,g,r) = img[i,j]
        bb = int(b)+40
        gg = int(g)+40
        rr = int(r)+40
        if bb>255:
            bb = 255
        if gg>255:
            gg = 255
        if rr>255:
            rr = 255
        dst[i,j] = (bb,gg,rr)
cv2.imshow('dst',dst)
```

![4](/images/cv/42.png)

### openCV磨皮美白-双边滤波

```python
import cv2
img = cv2.imread('1.png',1)
cv2.imshow('src',img)
dst = cv2.bilateralFilter(img,15,35,35) # 滤波函数
cv2.imshow('dst',dst)
```

![4](/images/cv/43.png)

### openCV高斯滤波

```python
import cv2
import numpy as np
img = cv2.imread('image11.jpg',1)
cv2.imshow('src',img)
dst = cv2.GaussianBlur(img,(5,5),1.5)
cv2.imshow('dst',dst)
```

![4](/images/cv/44.png)

### openCV均值滤波

```python
#均值 6*6 1 。 * 【6*6】/36 = mean -》P
import cv2
import numpy as np
img = cv2.imread('image11.jpg',1)
cv2.imshow('src',img)
imgInfo = img.shape
height = imgInfo[0]
width = imgInfo[1]
dst = np.zeros((height,width,3),np.uint8)
for i in range(3,height-3):
    for j in range(3,width-3):
        sum_b = int(0)
        sum_g = int(0)
        sum_r = int(0)
        for m in range(-3,3):#-3 -2 -1 0 1 2
            for n in range(-3,3):
                (b,g,r) = img[i+m,j+n]
                sum_b = sum_b+int(b)
                sum_g = sum_g+int(g)
                sum_r = sum_r+int(r)
            
        b = np.uint8(sum_b/36)
        g = np.uint8(sum_g/36)
        r = np.uint8(sum_r/36)
        dst[i,j] = (b,g,r)
cv2.imshow('dst',dst)
```

![4](/images/cv/45.png)

### openCV中值滤波

```python
import cv2
import numpy as np
img = cv2.imread('image11.jpg',1)
imgInfo = img.shape
height = imgInfo[0]
width = imgInfo[1]
img = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
cv2.imshow('src',img)
dst = np.zeros((height,width,3),np.uint8)
collect = np.zeros(9,np.uint8)
for i in range(1,height-1):
    for j in range(1,width-1):
        k = 0
        for m in range(-1,2):
            for n in range(-1,2):
                gray = img[i+m,j+n]
                collect[k] = gray
                k = k+1
        # 0 1 2 3 4 5 6 7 8
        #   1 
        for k in range(0,9):
            p1 = collect[k]
            for t in range(k+1,9):
                if p1<collect[t]:
                    mid = collect[t]
                    collect[t] = p1
                    p1 = mid
        dst[i,j] = collect[4]
cv2.imshow('dst',dst)
```

![4](/images/cv/46.png)