---
layout: post
title: "【三】数字成像系统"
date: 2019-03-22
description: "数字成像系统"
tag: 机器视觉
---

### 光通量

>指人眼所能感觉到的辐射功率，它等于单位时间内某一波段的辐射能量和该波段的相对视见率的乘积。
>
>符号：Φ
>
>单位：lm(流明)
>
>1lm = 0.00146瓦

#### 常见光源的光通量

![光通量](http://eveseven.oss-cn-shanghai.aliyuncs.com/19-1-11/66671052.jpg)

### 辐照度

>指投射到一平方米表面上的辐射通量密度。也就是说是到达一平方米表面上，单位时间，单位面积上的辐射能。
>
>符号：E
>
>单位：lux(勒克斯)
>
>1 lux = 1 lm/$m^2$

#### 常见照明环境的辐照度

![辐射度](http://eveseven.oss-cn-shanghai.aliyuncs.com/19-1-11/6673147.jpg)

### 光源类别

>按方向
>
>- 直射光
>- 漫射光
>
>按光谱
>
>- 可见光
>- 近可见光
>
>其他
>
>- 偏振
>- 其他

![](https://eveseven.oss-cn-shanghai.aliyuncs.com/20190309202114.png)

#### 特点

【**透视轮廓--背光源**】

![](https://eveseven.oss-cn-shanghai.aliyuncs.com/20190309202131.png)

>发射面是一个漫射面， 均匀性好。可用于镜面反射材料，如晶片或者玻璃基底上的伤痕检测；LCD检测；微小电子元件尺寸、形状、靶标测试

【**表面照明--漫射光源**】

![](https://eveseven.oss-cn-shanghai.aliyuncs.com/20190309202143.png)

>在一定工作距离下，光束集中、亮度高、均匀性好、照射面积相对较小。常用于液晶校正、塑胶容器检查、工作螺孔定位、标签检查、管脚检查、集成电路印字检查等等

【**颜色光源**】

![互补光源](http://eveseven.oss-cn-shanghai.aliyuncs.com/19-1-11/88508873.jpg)

>如果希望更加鲜明地突出某些颜色，则选择色环上相对应的互补颜色光源照明，这样就可以明显的提高图像的对比度。

### RGB

![rgb](http://eveseven.oss-cn-shanghai.aliyuncs.com/19-1-11/81330727.jpg)

>为生活中所最常见的三基色，也是和视觉感受一一致的，很多颜色都是由这三基色产生。
>
>三基色: 红、绿、蓝
>
>三色相交互是白色。

### CYMK

![cymk](http://eveseven.oss-cn-shanghai.aliyuncs.com/19-1-11/84796285.jpg)

> 补色模型
>
> 基色变为基本三基色的补色--红色-->青色、绿色-->黄色、蓝色-->品红
>
> 三基色：青、黄、品红
>
> 三色交互是黑色。

### HSI

![hsi](http://eveseven.oss-cn-shanghai.aliyuncs.com/19-1-11/29068085.jpg)

>- 色调H：描述纯色的属性（红、黄）。
>- 饱和度S：表示的是一种纯色被白光稀释的程度的度量。
>- 亮度I：体现了无色的光照强度的概念，是一个主观的描述。

【与RGB**的换算关系**】

![](https://eveseven.oss-cn-shanghai.aliyuncs.com/20190309202246.png)

`代码演示`

```python
import cv2

# 加载图片
img = cv2.imread('2.jpg')
# 缩放+灰度化
sizeImage = cv2.pyrDown(img)
grayImage = cv2.cvtColor(sizeImage, cv2.COLOR_BGR2GRAY)

cv2.imshow("source image", sizeImage)
cv2.imshow("gray", grayImage)
# RGB通道分离--opencv中，RGB三个通道是反过来的
rgbImage = cv2.split(sizeImage)
cv2.imshow("R", rgbImage[2])
cv2.imshow("G", rgbImage[1])
cv2.imshow("B", rgbImage[0])
# 分离后为单通道，相当于分离通道的同时把其他两个通道填充了相同的数值。
# 比如红色通道，分离出红色通道的同时，绿色和蓝色被填充为和红色相同的数值，这样一来就只有黑白灰了。
# 可以进行观察，会发现原图中颜色越接近红色的地方在红色通道越接近白色。
# 在纯红的地方在红色通道会出现纯白。
# R值为255 -> RGB(255，255，255)，为纯白

# HSI颜色模型+通道分离
hsv = cv2.cvtColor(sizeImage, cv2.COLOR_BGR2HSV)
hsvChannels = cv2.split(hsv)
cv2.imshow("Hue", hsvChannels[0])
cv2.imshow("Saturation", hsvChannels[1])
cv2.imshow("Value", hsvChannels[2])

cv2.waitKey(0)
```

![](https://eveseven.oss-cn-shanghai.aliyuncs.com/20190309210844.png)

### CCD传感器基本原理

![](https://eveseven.oss-cn-shanghai.aliyuncs.com/20190309202350.png)

### 彩色图像传感器的基本原理

![彩色  图像传感器](http://eveseven.oss-cn-shanghai.aliyuncs.com/19-1-11/45875188.jpg)

>最底层的感光区同样是采用CCD图像传感器
>
>彩色滤色片阵列(Color Filter Array),也被称为拜尔滤色镜(Bayer Filter)，排列在感光区上方。
>
>传感器不同，排列方式就不同。

![Bayer](http://eveseven.oss-cn-shanghai.aliyuncs.com/19-1-11/152560.jpg)

>这个单元包含了三原色：红(R),绿(G),蓝(B)。对于单个像素点，只有一种特定的彩色滤色片放置在其上方。彩色图像的像素点不仅包含通常的感光区域还包括了一个彩色滤色片。因为人眼对绿色的敏感度很高，所以我们使用了更多的绿色滤色片，类似棋盘格的形式来摆放绿色滤色片，以期得到有着更高空间分辨率的图像。红色和蓝色滤色片每隔一行与绿色滤色片错落放置。

### γ校正

![校正](http://eveseven.oss-cn-shanghai.aliyuncs.com/19-1-11/21721761.jpg)

>图像传感器输出时经过γ校正，以符合人的视觉习惯；存储时还原回原有的RGB值。

![校正环节](http://eveseven.oss-cn-shanghai.aliyuncs.com/19-1-11/22073630.jpg)

### 图像传输的方式

>- 模拟视频传输：采用同轴电缆等方式，将亮度和色度分离，在不同频带调制后在同一信号线上传输。常用的为同轴电缆，同轴电缆的中心导线用于传输信号，外层是金属屏蔽网。
>- RGB方式：显示器，投影
>- 数字传输（长距离）：光纤高清信号，网线
>- 数字传输（短距离）：USB，火线，HDMI

### 模拟、数字视频传输接口

![](https://eveseven.oss-cn-shanghai.aliyuncs.com/20190309202521.png)

### 常见的图像和视频压缩标准

![标准](http://eveseven.oss-cn-shanghai.aliyuncs.com/19-1-11/87798349.jpg)

### 显示设备及参数（液晶显示）

![](https://eveseven.oss-cn-shanghai.aliyuncs.com/20190309202658.png)

> - 分辨率（最佳分辨率）及显示器尺寸
> - 亮度和对比度：LCD的亮度以流明/平方米（cd/m2）度量，对比度是直接反映LCD显示器能否现丰富的色阶的参数
> - 响应时间：响应时间是LCD显示器的一个重要的参数，它指的是LCD显示器对于输入信号的反应时间。
> - 坏点：如果液晶显示屏中某一个发光单元有问题就会出现总丌透光、总透光、半透光等现象，这就是所谓的“坏点”

转载请注明：[Seven的博客](http://sevenold.github.io)