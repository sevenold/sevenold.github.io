---
layout: post
title: "使用 Virtualenv 进行安装TensorFlow"
date: 2018-08-11
description: "TensorFlow安装，TensorFlow，Virtualenv "
tag: TensorFlow
---



### 使用 Virtualenv 进行安装TensorFlow

环境：`ubuntu 16.04`

- 安装 pip 和 Virtualenv 

  ```
  $  sudo apt-get install python-pip python-dev python-virtualenv # for Python 2.7
  $  sudo apt-get install python3-pip python3-dev python-virtualenv # for Python 3.n
  ```

- 创建 Virtualenv 环境

- ```
  $  virtualenv --system-site-packages targetDirectory # for Python 2.7
  $  virtualenv --system-site-packages -p python3 targetDirectory # for Python 3.n
  ```

  `targetDirectory` 用于指定 Virtualenv 目录。我们假定创建的虚拟环境为`tensorflow`，即`targetDirectory`为`tensorflow`。 

- 激活 Virtualenv 环境 

- ```
  $  source ~/tensorflow/bin/activate 
  ```

  执行上述 `source` 命令后，您的提示符应该会变成如下内容： 

  ```
  (tensorflow)$ 
  ```

- 安装或更新`pip`

- ```
  (tensorflow)$ easy_install -U pip
  ```

  安装 `TensorFlow `

  ```
  (tensorflow)$ pip install --upgrade tensorflow      # for Python 2.7
  (tensorflow)$ pip3 install --upgrade tensorflow     # for Python 3.n
  (tensorflow)$ pip install --upgrade tensorflow-gpu  # for Python 2.7 and GPU
  (tensorflow)$ pip3 install --upgrade tensorflow-gpu # for Python 3.n and GPU
  ```

- 备注：如果安装失败或者网速慢，可参考[TensorFlow-CPU/GPU安装ubuntu16.04版](https://sevenold.github.io/2018/08/TensorFlow-ubuntu/) ,离线安装`tensorflow`。

- 退出 Virtualenv 环境 

- ```
  (tensorflow)$ deactivate 
  ```

  



### 验证环境

- 激活 Virtualenv 环境 

- ```
   $ source ~/tensorflow/bin/activate 
  ```

  在 Virtualenv 环境激活后，您就可以从这个 shell 运行 TensorFlow 程序。您的提示符将变成如下所示，这表示您的 Tensorflow 环境已处于活动状态： 

- ```
  (tensorflow)$ 
  ```

  输入代码验证

  ```
  import tensorflow as tf
  
  hello = tf.constant('Hello, TensorFlow!')
  
  sess = tf.Session()
  
  print(sess.run(hello))
  
  ```

- 看到如下输出，表示安装正确

- ```
  Hello, TensorFlow!
  ```

  用完 TensorFlow 后，可以通过发出以下命令调用 `deactivate` 函数来停用环境： 

  ```
  (tensorflow)$ deactivate 
  ```

- 提示符将恢复为您的默认提示符 ，就表示退出了。

- ```
  $
  ```

  



### 卸载 Virtualenv 环境 `targetDirectory`

要卸载你刚刚新建的Virtualenv 环境 `TensorFlow`的话，只需要删除该文件夹就可以了。

```
$ rm -r targetDirectory  # 在这里 targetDirectory改为你创建的环境名字
```

