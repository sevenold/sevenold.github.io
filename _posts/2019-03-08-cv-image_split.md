---
layout: post
title: "【八】图像分割"
date: 2019-03-22
description: "视觉处理算-图像分割"
tag: 机器视觉
---

### **灰度阈值分割**

> - 假设：图像中的目标区和背景区之间或者不同目标区之间，存在不同的灰度或平均灰度。
> - 凡是灰度值包含于z的像素都变成某一灰度值，其他的变成另一个灰度值，则该图像就以z为界被分
>   成两个区域。
>
> ![灰度分割公式](https://eveseven.oss-cn-shanghai.aliyuncs.com/20190115154608.png)
>
> - 如果=1和=0，分割后的图像为二值图像
>
> ![二值](https://eveseven.oss-cn-shanghai.aliyuncs.com/20190115154652.png)
>
> - 确定最佳阈值，使背景和目标之间的差异最大。

#### 大津（Otsu）算法

- 确定最佳阈值，使背景和目标之间的类间方差最大 (因为二者差异最大)。

  ![otsu](https://eveseven.oss-cn-shanghai.aliyuncs.com/20190115154854.png)

  ![图示](https://eveseven.oss-cn-shanghai.aliyuncs.com/1547539080340.png)

- 算法实现：遍历灰度取值。

### **区域生长法分割**

实现步骤：

> 1. 对图像顺序扫描!找到第1个还没有归属的像素, 设该像素为(x0, y0);
> 2. 以(x0, y0)为中心, 考虑(x0, y0)的8邻域像素(x, y)，如果(x, y)满足生长准则, 将(x, y)与 (x0, y0)合并, 同时将(x, y)压入堆栈;
> 3. 从堆栈中取出一个像素, 把它当作(x0, y0)返回到步骤2;
> 4. 当堆栈为空时，返回到步骤1;
> 5. 重复步骤1 - 4直到图像中的每个点都有归属时。生长结束。

### 【总结】

![图像分割](https://eveseven.oss-cn-shanghai.aliyuncs.com/20190116163011.png)

### 【基于阈值的分割】

> 基于阈值的分割算法是最简单直接的分割算法。由于图像中目标位置和其他区域之间具有不同的灰度值，具有这种性质的目标区域通过阈值分割能够取得非常好的效果。通过阈值进行分割通常可以需要一个或多个灰度值作为阈值使图像分成不同的目标区域与背景区域。

> 如何找到合适的阈值进行分割是基于阈值的分割算法中最核心的问题。学者们针对这一问题进行了深入的研究。大津法（OTSU）、最大熵法等算法是其中比较突出的算法，这类算法在图像中适用固定的阈值。但是也有一类算法使用局部阈值，这类算法称为自适应阈值算法。这类算法根据图像的局部特征计算局部的阈值，通常情况下这类算法在有阴影或者图像灰度不均匀的情况下，具有比全局阈值更加好的分割效果。由于基于阈值的分割算法对噪声敏感，通常情况图像在分割之前需要进行图像去噪的操作。       

> OTSU算法通过计算前景和背景的两个类的最大类间方差得到阈值，阈值将整张图像分为前景和背景两个部分。

### 【基于区域的分割】

> ​	基于区域生长的分割算法将具有相似特征的像素集合聚集构成一个区域，这个区域中的相邻像素之间具有相似的性质。算法首先在每个区域中寻找一个像素点作为种子点，然后人工设定合适的生长规则与停止规则，这些规则可以是灰度级别的特征、纹理级别的特征、梯度级别的特征等，生长规则可以根据实际需要具体设置。满足生长规则的像素点视为具有相似特征，将这些像素点划分到种子点所在区域中。将新的像素点作为种子点重复上面的步骤，直到所有的种子点都执行的一遍，生成的区域就是该种子点所在的区域。
>
> ​	区域生长法的优势是整个算法计算简单，对于区域内部较为平滑的连通目标能分割得到很好的结果，同时算法对噪声不那么敏感。而它的缺点也非常明显，需要人为选定合适的区域生长种子点，而且生长规则选取的不合适可能导致区域内有空洞，甚至在复杂的图像中使用区域生长算法可以导致欠分割或者过分割。最后，作为一种串行的算法，当目标区域较大时，目标分割的速度较慢。

### 【基于边缘的分割】

> ​	通过区域的边缘来实现图像的分割是图像分割中常见的一种算法。由于不同区域中通常具有结构突变或者不连续的地方，这些地方往往能够为图像分割提供了有效的依据。这些不连续或者结构突变的地方称为边缘。图像中不同区域通常具有明显的边缘，利用边缘信息能够很好的实现对不同区域的分割。
>
> ​        基于边缘的图像分割算法最重要的是边缘的检测。图像的边缘通常是图像颜色、灰度性质不连续的位置。对于图像边缘的检测，通常使用边缘检测算子计算得出。常用的图像边缘检测算子有：Laplace算子、Sobel算子、Canny算子等。

### 代码演示

```python
import cv2
import matplotlib.pyplot as plt
import numpy as np
import random
# 加载图片
histImage = cv2.imread('lena.jpg')
image_1 = cv2.imread('pic2.png')
image_2 = cv2.imread('pic6.png')
cv2.imshow("source-1", image_1)
cv2.imshow("source-2", image_2)


def GrayHist(img):
    """
    calcHist(images, channels, mask, histSize, ranges, hist=None, accumulate=None)
    images — 源矩阵指针（输入图像，可以多张）。 但都应当有同样的深度（depth）, 比如CV_8U 或者 CV_32F ，
             和同样的大小。每张图片可以拥有任意数量的通道。
    channels — 通道的数量，每个通道都有编号，灰度图为一通道，即channel[1]=0;对应着一维。
                BGR三通道图像编号为channels[3]={0,1,2};分别对应着第一维，第二维，第三维。
    mask — 掩码图像：要对图像处理时，先看看这个像素对应的掩码位是否为屏蔽，如果为屏蔽，就是说该像素不处理（掩码值为0的像素都将被忽略）
    hist — 返回的直方图。
    histSize — 直方图每一维的条目个数的数组，灰度图一维有256个条目。
    ranges — 每一维的像素值的范围，传递数组，与维数相对应，灰度图一维像素值范围为0~255。
    :param name: 命名标志
    :param img: 图片数据
    :return:
    """
    grayHist = cv2.calcHist([img],
                            [0],
                            None,
                            [256],
                            [0.0, 255.0])
    print(grayHist)
    plt.plot(grayHist)
    plt.show()


def ThresholdImage(image, name):
    """
    threshold(src, thresh, maxval, type, dst=None)
    src： 输入图，只能输入单通道图像，通常来说为灰度图
    dst： 输出图
    thresh： 阈值
    maxval： 当像素值超过了阈值（或者小于阈值，根据type来决定），所赋予的值
    type：二值化操作的类型，包含以下5种类型： cv2.THRESH_BINARY； cv2.THRESH_BINARY_INV；
                                        cv2.THRESH_TRUNC； cv2.THRESH_TOZERO；cv2.THRESH_TOZERO_INV
    :param img:
    :return:
    """
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    listType = [cv2.THRESH_BINARY, cv2.THRESH_BINARY_INV, cv2.THRESH_TRUNC, cv2.THRESH_TOZERO, cv2.THRESH_TOZERO_INV]
    for i in listType:
        thresholdImage = cv2.threshold(img, 125, 255, i)
        cv2.imshow("threshold-%s-%s" % (name, i), thresholdImage[1])


def floodFillImage(img, name, lo, up):
    """
    floodFill(image, mask, seedPoint, newVal, loDiff=None, upDiff=None, flags=None)
    image 【输入/输出】 1或者3通道、 8bit或者浮点图像。仅当参数flags的FLOODFILL_MASK_ONLY标志位被设置时image不会被修改，否则会被修改。

    mask 【输入/输出】 操作掩码，必须为单通道、8bit，且比image宽2个像素、高2个像素。使用前必须先初始化。
                        Flood-filling无法跨越mask中的非0像素。例如，一个边缘检测的结果可以作为mask来阻止边缘填充。
                        在输出中，mask中与image中填充像素对应的像素点被设置为1，或者flags标志位中设置的值(详见flags标志位的解释)。
                        此外，该函数还用1填充了mask的边缘来简化内部处理。因此，可以在多个调用中使用同一mask，以确保填充区域不会重叠。
    seedPoint 起始像素点
    newVal    重绘像素区域的新的填充值(颜色)
    loDiff    当前选定像素与其连通区中相邻像素中的一个像素，或者与加入该连通区的一个seedPoint像素，二者之间的最大下行差异值。
    upDiff    当前选定像素与其连通区中相邻像素中的一个像素，或者与加入该连通区的一个seedPoint像素，二者之间的最大上行差异值。
    flags     flags标志位是一个32bit的int类型数据，其由3部分组成： 0-7bit表示邻接性(4邻接、8邻接)；
                    8-15bit表示mask的填充颜色；16-31bit表示填充模式（详见填充模式解释）
    :param img:
    :param name:
    :return:
    """
    seed = (10, 30)  # 起始位置
    # 构建mask,根据mask参数的介绍,其size必须为宽img+2,高img+2
    mask = np.zeros((img.shape[0] + 2, img.shape[1] + 2), dtype=np.uint8)
    newVal = (255, 64, 64)  # img fill的填充颜色值

    # 执行floodFill操作
    # ret, image, mask, rect = cv2.floodFill(img, mask, seed, newVal)
    cv2.floodFill(img, mask, seed, newVal,
                  (lo, )*3, (up, )*3,
                  cv2.FLOODFILL_FIXED_RANGE)
    cv2.imshow('floodfill-%s' % name, img)


# 计算并显示直方图
GrayHist(image_1)
# 大津算法
ThresholdImage(image_1, '1')
ThresholdImage(image_2, '2')
# riceImage = cv2.imread('rice.png')
# ThresholdImage(riceImage, 'rice')

floodFillImage(image_1, '1', 200, 255)
floodFillImage(image_2, '2', 0, 0)
cv2.waitKey(0)
```

![diamante](https://eveseven.oss-cn-shanghai.aliyuncs.com/20190116150300.png)

转载请注明：[Seven的博客](http://sevenold.github.io)