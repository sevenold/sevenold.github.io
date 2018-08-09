---
layout: post
title: "TensorFlow-CPU/GPU安装windows版"
date: 2018-08-09
description: "TensorFlow安装，TensorFlow"
tag: TensorFlow
---



### Windows TensorFlow-CPU安装

备注：CPU和GPU是不能同时安装的，如果需要的话可以再配置一个环境或者建个虚拟环境。

环境：window7或以上

python版本要求：`3.5.x`以上

打开**window cmd**，直接使用CPU-only命令安装，如下： 

```
pip3 install --upgrade tensorflow
```

![image](/images/dl/11.png)

然后等待，直至安装成功。

> > > 

如果你网速很慢的话，你可以选择离线安装。

下载所需的离线包：https://pypi.org/project/tensorflow/#files

打开**window cmd**，进入你保存文件的目录，使用命令

```
pip install tensorflow-1.10.0-cp36-cp36m-win_amd64.whl
```

然后等待，直至安装成功。

![image](/images/dl/14.png)

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



### Windows TensorFlow-GPU安装

如果你的电脑的GPU支持CUDA，那么你就可以使用GPU加速了

查看你的电脑GPU是否支持CUDA:https://developer.nvidia.com/cuda-gpus

![image](/images/dl/15.png)



### 安装CUDA

当然你的先安装显卡的驱动，这个就自己查询了。我们默认你已经装好了。

查看驱动程序版本号

![image](/images/dl/16.png)

对应版本号去下载对应的CUDA安装包

![image](/images/dl/17.png)

由上图可以看出，我这台演示电脑的

- 版本驱动号：391.24
- 对应版本：CUDA：9.0

所以我们对应的CUDA的下载版本就是9.0，下载网站：https://developer.nvidia.com/cuda-toolkit-archive

![image](/images/dl/18.png)



### 安装说明

- 双击cuda_9.0.176_windows.exe
- 按照屏幕上的提示操作
- 如果出现下面这个界面--说明你还需要安装`[ Visual Studio 2012或以上版本]`
  ![image](/images/dl/21.png)



### 安装cuDNN

同样，根据我们的驱动程序版本号：我们下载的对应版本：

- 版本驱动号：391.24
- CUDA：9.0
- cuDNN：v7.1.4

下载网址：https://developer.nvidia.com/rdp/cudnn-archive

![image](/images/dl/19.png)

![image](/images/dl/20.png)



### 安装说明

说明：把cuDNN的文件复制到CUDA Toolkit 安装目录

解压cudnn5.0

- 生成cuda/include、cuda/lib、cuda/bin三个目录； 
- 分别将`cuda/include、cuda/lib、cuda/bin`三个目录中的内容拷贝到`C:\Program Files\NVIDIAGPU Computing Toolkit\CUDA\v8.0`对应的`include、lib、bin`目录下即可 
- 如果你是自定义的安装路径，需要自己搜索一下NVIDIA GPU Computing Toolkit的目录 



### 安装 TensorFlow-GPU

打开**window cmd**，直接使用命令安装，如下： 

```
pip install --upgrade tensorflow-gpu
```

然后等待，直至安装成功。

> > > 

如果你网速很慢的话，你可以选择离线安装。

下载所需的离线包：https://pypi.org/project/tensorflow-gpu/#files

打开**window cmd**，进入你保存文件的目录，使用命令

```
pip3 install tensorflow_gpu-1.10.0-cp36-cp36m-win_amd64.whl
```

然后等待，直至安装成功。

![image](/images/dl/22.png)



### 验证安装

```
import tensorflow as tf

hello = tf.constant('Hello, TensorFlow!')

sess = tf.Session()

print(sess.run(hello))
```

看到如下的输出，表示安装正确。 

![image](/images/dl/23.png)



