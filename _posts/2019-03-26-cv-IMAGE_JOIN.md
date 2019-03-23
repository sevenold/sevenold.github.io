---
layout: post
title: "全景拼接"
date: 2019-03-23
description: "全景拼接"
tag: 机器视觉
---

```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/2/16 21:29
# @Author  : Seven
# @File    : ImageJoin.py
# @Software: PyCharm
# function : 图片拼接
import cv2
import numpy as np


def get_stitched_image(img1, img2, M):
    """
    使用关键点来缝合图像
    :param img1:
    :param img2:
    :param M: 对应矩阵
    :return: 拼接后的图片数据
    """
    # 获取输入图片的维度
    w1, h1 = img1.shape[:2]
    w2, h2 = img2.shape[:2]

    # 生成对应维度的画布
    img1_dims = np.float32([[0, 0], [0, w1], [h1, w1], [h1, 0]]).reshape(-1, 1, 2)
    img2_dims_temp = np.float32([[0, 0], [0, w2], [h2, w2], [h2, 0]]).reshape(-1, 1, 2)

    # 获取透视变换的交换矩阵
    img2_dims = cv2.perspectiveTransform(img2_dims_temp, M)

    # 最终的维度
    result_dims = np.concatenate((img1_dims, img2_dims), axis=0)

    # 开始拼接图片
    # 通过匹配点来计算维度
    [x_min, y_min] = np.int32(result_dims.min(axis=0).ravel() - 0.5)
    [x_max, y_max] = np.int32(result_dims.max(axis=0).ravel() + 0.5)

    # 变换后创建输出矩阵
    transform_dist = [-x_min, -y_min]
    transform_array = np.array([[1, 0, transform_dist[0]],
                                [0, 1, transform_dist[1]],
                                [0, 0, 1]])

    # 修正图像以获得结果图像
    result_img = cv2.warpPerspective(img2, transform_array.dot(M),
                                     (x_max - x_min, y_max - y_min))
    result_img[transform_dist[1]:w1 + transform_dist[1], transform_dist[0]:h1 + transform_dist[0]] = img1

    # 返回最终结果
    return result_img


def get_sift_homography(img1, img2):
    """
    使用SIFT方法获取两张图片的关键点
    :param img1:
    :param img2:
    :return: 关键点组成的对应矩阵
    """
    # 初始化SIFT方法
    sift = cv2.xfeatures2d_SIFT.create()

    # 获取关键点和描述子
    k1, d1 = sift.detectAndCompute(img1, None)
    k2, d2 = sift.detectAndCompute(img2, None)

    # 初始化Bruteforce匹配器
    bf = cv2.BFMatcher()
    # 通过KNN匹配两张图片的描述子
    matches = bf.knnMatch(d1, d2, k=2)
    # 确定最好的匹配
    verified_matches = []
    for m1, m2 in matches:
        # 把匹配的较好的添加到矩阵中
        if m1.distance < 0.8 * m2.distance:
            verified_matches.append(m1)

    # 最小匹配数
    min_matches = 8
    if len(verified_matches) > min_matches:
        # 存储匹配点
        img1_pts = []
        img2_pts = []
        # 将匹配点加入矩阵中
        for match in verified_matches:
            img1_pts.append(k1[match.queryIdx].pt)
            img2_pts.append(k2[match.trainIdx].pt)
        img1_pts = np.float32(img1_pts).reshape(-1, 1, 2)
        img2_pts = np.float32(img2_pts).reshape(-1, 1, 2)
        # 计算单应矩阵
        M, mask = cv2.findHomography(img1_pts, img2_pts, cv2.RANSAC, 5.0)
        return M
    else:
        print('Error: Not enough matches')
        exit()


def changeImage(img1, img2):
    # 使用SIFT查找关键点并返回关键点组成的矩阵
    M = get_sift_homography(img1, img2)
    # 使用关键点矩阵将图像缝合在一起
    result_image = get_stitched_image(img2, img1, M)
    return result_image


def main():
    # 加载图片
    img1 = cv2.imread("images/csdn_A.jpg")
    img2 = cv2.imread("images/csdn_B.jpg")
    img3 = cv2.imread("images/csdn_C.jpg")
    # 显示加载的图片
    input_images = np.hstack((img1, img2, img3))
    cv2.imshow('Input Images', input_images)

    out1 = changeImage(img1, img2)
    out2 = changeImage(img1, img3)
    result_image = changeImage(out1, out2)

    # 保存文件
    result_image_name = 'results/result_tree.jpg'
    cv2.imwrite(result_image_name, result_image)

    # 显示拼接的结果
    cv2.imshow('Result', result_image)
    cv2.waitKey()


if __name__ == '__main__':
    main()
```

![](https://eveseven.oss-cn-shanghai.aliyuncs.com/20190323145021.png)

转载请注明：[Seven的博客](http://sevenold.github.io)