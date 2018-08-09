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

下载地址：https://www.geforce.cn/drivers

![image](/images/dl/25.png)

#### **准备工作**

在待安装驱动的主机上打开一个终端(Ctrl+Alt+T)，或者直接切换到终端界面(Ctrl+Alt+F1),进行如下操作 

- 卸载可能存在的旧版本 nvidia 驱动（对没有安装过 nvidia 驱动的主机，这步可以省略，但推荐执行，以防万一） 

- ```
  sudo apt-get remove --purge nvidia*
  ```

  安装驱动可能需要的依赖(可选) 

  ```
    sudo apt-get update
  
    sudo apt-get install dkms build-essential linux-headers-generic
  ```

- 把 nouveau 驱动加入黑名单 

- ```
  sudo nano /etc/modprobe.d/blacklist-nouveau.conf
  
  在文件 blacklist-nouveau.conf 中加入如下内容：
    blacklist nouveau
    blacklist lbm-nouveau
    options nouveau modeset=0
    alias nouveau off
    alias lbm-nouveau off
  
  ```

  禁用 nouveau 内核模块 

  ```
  echo options nouveau modeset=0 | sudo tee -a /etc/modprobe.d/nouveau-kms.conf
  
  sudo update-initramfs -u
  ```

- 重启电脑，然后终端输入

- ```
  lsmod | grep nouveau
  ```

  如果没有内容输出，那就禁用成功了。

#### **开始安装**

- 禁用X server

- ```
  Ctrl + Alt + F1   ===> 进入命令行模式
  
  sudo service lightdm stop   ===> 禁用X server
  ```

  安装驱动

  ```
  sudo chmod u+x NVIDIA-Linux-x86_64-361.45.11.run
  
  sudo ./NVIDIA-Linux-x86_64-361.45.11.run
  ```

- 如果出现`The distribution-provided pre-install script failed! `不必理会，继续安装

- 最重要的：第一个选`NO`，在选择`would you like to run the nvidia-xconfig ...............when you restart X?`--选择`YES`;否则在启动X-windows的时候不会使用NVIDIA驱动。

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

- 版本驱动号：`391.24`
- 对应版本：`CUDA：9.1`

所以我们对应的CUDA的下载版本就是9.1，下载网站：https://developer.nvidia.com/cuda-toolkit-archive 

![image](/images/dl/32.png)

### 安装说明

进入安装包的路径执行以下命令（注意版本号）

- `sudo dpkg -i cuda-repo-ubuntu1604-9-1-local_9.1.85-1_amd64.deb`
- `sudo apt-key add /var/cuda-repo-9-1-local/7fa2af80.pub`
- `sudo apt-get update`
- `sudo apt-get install cuda`

然后等待，安装完成。

![image](/images/dl/33.png)

### 安装cuDNN

同样，根据我们的驱动程序版本号：我们下载的对应版本： 

- 版本驱动号：`391.24`
- 对应版本：`CUDA：9.1`
- 对应版本：`cuDNN: v7.13`

下载网址：https://developer.nvidia.com/rdp/cudnn-archive 

![image](/images/dl/34.png)

![image](/images/dl/35.png)

### 安装说明

- 解压下载好的安装包以后会出现cuda的目录，进入该目录 执行以下命令

- ```
  cd cuda/include/   
  sudo cp cudnn.h /usr/local/cuda-9.1/include/   
  cd ../lib64   
  sudo cp lib* /usr/local/cuda-9.1/lib64/   
  sudo chmod a+r /usr/local/cuda-9.1/include/cudnn.h/ usr/local/cuda-9.1/lib64/libcudnn*
  ```

  接下来执行以下命令： 

  ```
  cd /usr/local/cuda-9.1/lib64/   
  sudo rm -rf libcudnn.so libcudnn.so.5   
  sudo ln -s libcudnn.so.5.1.5 libcudnn.so.5   
  sudo ln -s libcudnn.so.5 libcudnn.so
  ```

- 在终端中输入以下命令进行环境变量的配置： 

- ```
  sudo gedit /etc/profile
  ```

  在末尾加上： 

  ```
  PATH=/usr/local/cuda-9.1/bin:$PATH   
  export PATH
  ```

- 创建链接文件 

- ```
  sudo gedit /etc/ld.so.conf.d/cuda.conf
  ```

  在该文件末尾加入 

  ```
  /usr/local/cuda/lib64
  ```

- 然后使用ldconfig使之生效 

- ```
  sudo ldconfig
  ```

  

### 安装 TensorFlow-GPU

最简单的方式是使用pip安装：

```
# Python 2.7
pip install --upgrade tensorflow-gpu
# Python 3.x
pip3 install --upgrade tensorflow-gpu
```

如果你网速很慢的话，你可以选择离线安装。

下载所需的离线包：https://pypi.org/project/tensorflow-gpu/#files

打开终端，进入你保存文件的目录，使用命令

```
pip3 install tensorflow_gpu-1.10.0-cp36-cp36m-win_amd64.whl
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

```
Hello, TensorFlow!
```

