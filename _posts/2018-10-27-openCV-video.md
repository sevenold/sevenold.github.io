---
layout: post
title: "计算机视觉-openCV视频处理"
date: 2018-10-27
description: "opencv安装、计算机视觉"
tag: OpenCV
---

### openCV视频分解图片

```python
import cv2
cap = cv2.VideoCapture("1.mp4")# 获取一个视频打开cap 1 file name
isOpened = cap.isOpened# 判断是否打开‘
print(isOpened)
fps = cap.get(cv2.CAP_PROP_FPS)#帧率
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))#w h
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(fps,width,height)
i = 0
while(isOpened):
    if i == 10:
        break
    else:
        i = i+1
    (flag,frame) = cap.read()# 读取每一张 flag frame 
    fileName = 'image'+str(i)+'.jpg'
    print(fileName)
    if flag == True:
        cv2.imwrite(fileName,frame,[cv2.IMWRITE_JPEG_QUALITY,100])
print('end!')
```



### openCV图片合成视频

```python
import cv2
img = cv2.imread('image1.jpg')
imgInfo = img.shape
size = (imgInfo[1],imgInfo[0])
print(size)
videoWrite = cv2.VideoWriter('2.mp4',-1,5,size)# 写入对象 1 file name
# 2 编码器 3 帧率 4 size
for i in range(1,11):
    fileName = 'image'+str(i)+'.jpg'
    img = cv2.imread(fileName)
    videoWrite.write(img)# 写入方法 1 jpg data
print('end!')
```

