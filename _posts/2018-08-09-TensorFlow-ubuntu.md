---
layout: post
title: "TensorFlow-CPU/GPU安装ubuntu16.04版"
date: 2018-08-10
description: "TensorFlow安装，TensorFlow"
tag: TensorFlow
---

`系统：Ubuntu 16.04。显卡：GTX 1050，独显无集成显卡。 `

### Ubuntu TensorFlow-CPU安装



最简单的方法使用 pip 来安装 

```
# Python 2.7
pip install --upgrade tensorflow
# Python 3.x
pip3 install --upgrade tensorflow
```

![image](/images/dl/26.png)

安装出错，或者下载速度慢，可以采用离线安装的方式

离线安装包下载地址：https://pypi.org/project/tensorflow/

然后进入安装包路径：

```
# Python 2.7
pip install tensorflow-1.10.0-cp27-cp27mu-manylinux1_x86_64.whl 
# Python 3.x
pip3 install tensorflow-1.10.0-cp35-cp35m-manylinux1_x86_64.whl 

```

![image](/images/dl/27.png)

然后等待，安装成功。

![image](/images/dl/28.png)



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

![image](/images/dl/29.png)



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

- `System Setting `-->`software&Updates`-->`Additional Drives`，然后选择`NVIDIA`驱动

  ![image](/images/dl/41.png)

  

- 安装成功--重启电脑

#### 查看是否安装成功

```
nvidia-smi
```

![image](/images/dl/30.png)

出现这个页面就表示安装成功了。



### 安装CUDA

查询电脑的版本号：

```
nvidia-smi
```

![image](/images/dl/31.png)



对应版本号去下载对应的CUDA安装包 

![image](/images/dl/17.png)

由上图可以看出，我这台演示电脑的

- 版本驱动号：`384.130`
- 对应版本：`CUDA：8.0`

所以我们对应的CUDA的下载版本就是8.0，注意我下载的是`runfile`，下载网站：https://developer.nvidia.com/cuda-toolkit-archive 

![image](/images/dl/32.png)

### 安装说明

进入安装包的路径执行以下命令（注意版本号）

- ```
  sudo sh cuda_8.0.61_375.26_linux.run
  ```

按照命令行提示 在安装过程中会询问是否安装显卡驱动，由于我们在第一步中已经安装，所以我们选择否（不安装） 

![image](/images/dl/38.png)

然后等待，安装完成。

安装完成后可能会有警告，提示samplees缺少必要的包： 

![image](/images/dl/43.png)



原因是缺少相关的依赖库,安装相应库就解决了： 

```
sudo apt-get install freeglut3-dev build-essential libx11-dev libxmu-dev libxi-dev libgl1-mesa-glx libglu1-mesa libglu1-mesa-dev 
```

![image](/images/dl/39.png)



### 配置环境变量

打开shell运行： `gedit ~/.bashrc `

加入如下内容： 

```
export PATH=/usr/local/cuda-8.0/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64:$LD_LIBRARY_PATH
```

立即生效，运行`source ~/.bashrc` 

### 测试是否安装成功

- 查看CUDA版本:` nvcc -V`

- ```
  seven@seven:~$ nvcc -V
  nvcc: NVIDIA (R) Cuda compiler driver
  Copyright (c) 2005-2016 NVIDIA Corporation
  Built on Tue_Jan_10_13:22:03_CST_2017
  Cuda compilation tools, release 8.0, V8.0.61
  ```

  1. 编译 CUDA Samples
     进入samples的安装目录
     我们选择其中一个进行编译验证下如：

![image](/images/dl/40.png)

如果没有报错，则安装完成 



### 安装cuDNN

同样，根据我们的驱动程序版本号：我们下载的对应版本： 

- 版本驱动号：`384.130`
- 对应版本：`CUDA：8.0`
- 对应版本：`cuDNN: v6.0`

下载网址：https://developer.nvidia.com/rdp/cudnn-archive 

![image](/images/dl/34.png)



### 安装说明

- 解压下载好的安装包以后会出现cuda的目录，进入该目录 执行以下命令

- ```
  cd ~
  tar -zxf cudnn-8.0-linux-x64-v5.1.tgz
  cd cuda
  sudo cp lib64/* /usr/local/cuda/lib64/
  sudo cp include/* /usr/local/cuda/include/
  ```




### 安装完成

至此，CUDA与cuDNN已经安装完成



### 安装 TensorFlow-GPU

备注：我用的是cuda 8.0和cudnn6.0 所以TensorFlow的版本应该是1.4。

最简单的方式是使用pip安装：

```
# Python 2.7
pip install --upgrade tensorflow-gpu==1.4
# Python 3.x
pip3 install --upgrade tensorflow-gpu==1.4
```

如果你网速很慢的话，你可以选择离线安装。

下载所需的离线包：https://github.com/tensorflow/tensorflow/tags

打开终端，进入你保存文件的目录，使用命令

```
pip3 install tensorflow_gpu-1.4.0-cp35-cp35m-manylinux1_x86_64.whl 
```

然后等待，直至安装成功。 

![image](/images/dl/37.png)

### 验证安装

```
import tensorflow as tf

hello = tf.constant('Hello, TensorFlow!')

sess = tf.Session()

print(sess.run(hello))

```

看到如下的输出，表示安装正确。 

![image](/images/dl/42.png)

```
Hello, TensorFlow!
```





