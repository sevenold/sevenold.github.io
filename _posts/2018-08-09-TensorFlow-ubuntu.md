---
layout: post
title: "TensorFlow-CPU/GPU安装ubuntu16.04版"
date: 2018-08-09
description: "TensorFlow安装，TensorFlow"
tag: TensorFlow
---



### Ubuntu TensorFlow-CPU安装



最简单的方法使用 pip 来安装 

```
# Python 2.7
pip install --upgrade tensorflow
# Python 3.x
pip3 install --upgrade tensorflow
```



安装出错，或者下载速度慢，可以采用离线安装的方式

离线安装包下载地址：https://pypi.org/project/tensorflow/

然后进入安装路径：

```
# Python 2.7
pip install tensorflow-1.10.0-cp27-cp27mu-manylinux1_x86_64.whl 
# Python 3.x
pip3 install tensorflow-1.10.0-cp35-cp35m-manylinux1_x86_64.whl 

```

然后等待，安装成功。



### 验证安装

```
import tensorflow as tf

hello = tf.constant('Hello, TensorFlow!')

sess = tf.Session()

print(sess.run(hello))

```

看到如下的输出，表示安装正确。 

```
Hello, TensorFlow!
```



### Ubuntu TensorFlow-GPU安装

如果你的电脑的GPU支持CUDA，那么你就可以使用GPU加速了 

### 检查自己的GPU是否是CUDA-capable 

```
lspci | grep -i nvidia
```

![image](/images/dl/24.png)

查看你的电脑GPU是否支持CUDA:https://developer.nvidia.com/cuda-gpus 

![image](/images/dl/15.png)



### 安装NVIDIA驱动

下载地址：https://www.geforce.cn/drivers

![image](/images/dl/25.png)

