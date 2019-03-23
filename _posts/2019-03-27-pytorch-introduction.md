---
layout: post
title: "PyTorch概述、环境部署及简单使用"
date: 2019-03-23
description: "PyTorch概述、环境部署及简单使用"
tag: Pytorch
---

### PyTorch概述

【**官网**】

![](https://eveseven.oss-cn-shanghai.aliyuncs.com/20190305150727.png)

> 网址[pytorch](https://pytorch.org/) ：https://pytorch.org/

【**简介**】

​	PyTorch是一个非常有可能改变深度学习领域前景的Python库。[PyTorch](https://github.com/pytorch/pytorch)是一个较新的深度学习框架。从名字可以看出，其和Torch不同之处在于PyTorch使用了Python作为开发语言，所谓“Python first”。一方面，使用者可以将其作为加入了GPU支持的`numpy`，另一方面，PyTorch也是强大的深度学习框架。

​	目前有很多深度学习框架，PyTorch主推的功能是`动态网络模型`。例如在TensorFlow中，使用者通过预先编写网络结构，网络结构是不能变化的（但是TensorFlow2.0加入了动态网络的支持）。而PyTorch中的网络结构不再只能是一成不变的。同时PyTorch实现了多达上百种op的自动求导（AutoGrad）。

​	在我使用过的各种深度学习库中，PyTorch是最灵活、最容易掌握的。

【**发展路线**】

- 002年发布`Torch`
- 2011年发布`Torch7`

> 最开始的torch框架是使用`lua`语言开发的，代码的可读性和实现起来都挺麻烦，所以热度一直就不高。
>
> ```lua
> torch.Tensor{1,2,3}
> 
> function gradUpdate(mlp,x,y,learningRate)
>   local criterion = nn.ClassNLLCriterion()
>   pred = mlp:forward(x)
>   local err = criterion:forward(pred, y); 
>   mlp:zeroGradParameters();
>   local t = criterion:backward(pred, y);
>   mlp:backward(x, t);
>   mlp:updateParameters(learningRate);
> end
> ```

- 2016年10月发布0.1，后端还是基于原来的`torch`的Python接口发布的的`PyTorch`
- 2018年12月发布1.0，后端改为了`caffe2`的pytorch稳定版1.0

【**框架**】

  ![](https://eveseven.oss-cn-shanghai.aliyuncs.com/20190221140942.png)

【**评分**】

![](https://eveseven.oss-cn-shanghai.aliyuncs.com/20190305152746.png)

【**招聘描述**】

![](https://eveseven.oss-cn-shanghai.aliyuncs.com/20190305152857.png)

【**github情况**】

![](https://eveseven.oss-cn-shanghai.aliyuncs.com/20190305153024.png)

> 综合来说，pytorch也是一个非常热门的趋势。

### PyTorch环境配置

[GPU环境配置](https://sevenold.github.io/2018/08/TensorFlow-windows/)

[Pycharm-Python环境配置](https://sevenold.github.io/2018/11/anaconda-windows/)

【[Python版本：3.6.x](https://www.python.org/)】

![](https://eveseven.oss-cn-shanghai.aliyuncs.com/20190221154911.png)

【[cuda版本：9.0](https://developer.nvidia.com/cuda-downloads)】

![](https://eveseven.oss-cn-shanghai.aliyuncs.com/20190221154939.png)

【[torch版本：1.0](https://pytorch.org/)】

![](https://eveseven.oss-cn-shanghai.aliyuncs.com/20190221154743.png)

【[pycharm：commit](https://www.jetbrains.com/pycharm/)】

![](https://eveseven.oss-cn-shanghai.aliyuncs.com/20190221155001.png)

【**测试**】

```python
>>> import torch
>>> torch.__version__
'1.0.1'
>>> torch.cuda.is_available()
True
```

> 输出为true，则表示环境配置成功并可以支持GPU运算。

### PyTorch功能简介

【**PyTorch动态图**】

> 每一步都是构建图的过程
>
> - 适合debug

![](https://eveseven.oss-cn-shanghai.aliyuncs.com/dynamic_graph.gif)

【**tensorflow静态图**】

> 先建好图，然后计算图
>
> - 不需要中间修改
> - 自建命名体系
> - 自建时序控制
> - 很难debug
> - 2.0支持动态图优先

![](https://eveseven.oss-cn-shanghai.aliyuncs.com/20190305153526.png)

【**深度学习计算库**】

- **`GPU加速`**

  > 代码实例：
  >
  > ```python
  > import torch
  > import time
  > 
  > print(torch.__version__)
  > print(torch.cuda.is_available())  # 判断是否支持GPU运算
  > 
  > # 初始化计算变量
  > a = torch.randn(10000, 1000)
  > b = torch.randn(1000, 2000)
  > 
  > start = time.time()
  > # 计算矩阵的乘法
  > c = torch.matmul(a, b)
  > 
  > end = time.time()
  > 
  > print("cpu计算时间及结果", a.device, end-start, c.norm(2))
  > 
  > device = torch.device('cuda')
  > a = a.to(device)
  > b = b.to(device)
  > 
  > 
  > start = time.time()
  > c = torch.matmul(a, b)
  > 
  > end = time.time()
  > print("GPU初始化", a.device, end-start, c.norm(2))
  > 
  > start = time.time()
  > c = torch.matmul(a, b)
  > 
  > end = time.time()
  > print("GPU计算时间及结果", a.device, end-start, c.norm(2))
  > ```
  >
  > 运行结果：
  >
  > ```python
  > 1.0.1
  > True
  > cpu计算时间及结果 cpu 0.21739435195922852 tensor(141500.7031)
  > GPU初始化 cuda:0 1.1588668823242188 tensor(141500.7188, device='cuda:0')
  > GPU计算时间及结果 cuda:0 0.009351253509521484 tensor(141500.7188, device='cuda:0')
  > ```
  >
  > 使用GPU运算，在我的电脑上运算速度快了20多倍。

- **`自动求导`**

  > 代码实例：
  >
  > 公式：$y = a^2x+bx+c$
  >
  > 数学计算结果：2a、1、1
  >
  > ```python
  > import torch
  > import time
  > from torch import autograd
  > print(torch.__version__)
  > print(torch.cuda.is_available())  # 判断是否支持GPU运算
  > x = torch.tensor(1.)
  > a = torch.tensor(1., requires_grad=True)
  > b = torch.tensor(2., requires_grad=True)
  > c = torch.tensor(3., requires_grad=True)
  > 
  > y = a**2 * x + b * x + c
  > 
  > print('before:', a.grad, b.grad, c.grad)
  > grads = autograd.grad(y, [a, b, c])
  > print('after :', grads[0], grads[1], grads[2])
  > ```
  >
  > 运行结果
  >
  > ```python
  > 1.0.1
  > True
  > before: None None None
  > after : tensor(2.) tensor(1.) tensor(1.)
  > ```

- 常用网络层API

  ![](https://eveseven.oss-cn-shanghai.aliyuncs.com/20190221152531.png)

  转载请注明：[Seven的博客](http://sevenold.github.io)

