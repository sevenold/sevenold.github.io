---
layout: post
title: "pytorch自动求导、numpy的转换、模型的存取"
date: 2019-03-23
description: "pytorch自动求导、numpy的转换、模型的存取"
tag: Pytorch
---

### autograd自动求导一

```python
import torch
import torchvision
import torch.nn as nn
import numpy as np
# 创建张量
x = torch.tensor(1., requires_grad=True)
w = torch.tensor(2., requires_grad=True)
b = torch.tensor(3., requires_grad=True)
# 构建计算图。
y = w * x + b    # y = 2 * x + 3
# # 计算渐变。
# y.backward()
# 
# # 打印渐变。
# print(x.grad)    # x.grad = 2
# print(w.grad)    # w.grad = 1
# print(b.grad)    # b.grad = 1

grads = torch.autograd.grad(y, [x, b, w])
print(grads)
```

> (tensor(2.), tensor(1.), tensor(1.))

### autograd自动求导二

```python
# Create tensors of shape (10, 3) and (10, 2).
x = torch.randn(10, 3)
y = torch.randn(10, 2)

# Build a fully connected layer.
linear = nn.Linear(3, 2)
print('w: ', linear.weight)
print('b: ', linear.bias)

# Build loss function and optimizer.
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(linear.parameters(), lr=0.01)

# Forward pass.
pred = linear(x)

# Compute loss.
loss = criterion(pred, y)
print('loss: ', loss.item())

# Backward pass.
loss.backward()

# Print out the gradients.
print('dL/dw: ', linear.weight.grad)
print('dL/db: ', linear.bias.grad)

# 1-step gradient descent.
optimizer.step()

# Print out the loss after 1-step gradient descent.
pred = linear(x)
loss = criterion(pred, y)
print('loss after 1 step optimization: ', loss.item())
```

> w:  Parameter containing:
> tensor([[ 0.1648,  0.3432,  0.2555],
>         [ 0.5305, -0.2595,  0.5334]], requires_grad=True)
> b:  Parameter containing:
> tensor([-0.2244,  0.5731], requires_grad=True)
> loss:  0.8838511109352112
> dL/dw:  tensor([[ 0.3066,  0.4282,  0.3613],
>         [ 0.0505, -0.4339,  0.6579]])
> dL/db:  tensor([-0.1423,  0.3835])
> loss after 1 step optimization:  0.8719408512115479

###  从numpy加载数据 

```python
# Create a numpy array.
x = np.array([[1, 2], [3, 4]])

# Convert the numpy array to a torch tensor.
y = torch.from_numpy(x)

# Convert the torch tensor to a numpy array.
z = y.numpy()
print(z)
```

> [[1 2]
>  [3 4]]

### 预训练模型

```python
# Download and load the pretrained ResNet-18.
resnet = torchvision.models.resnet18(pretrained=True)

# If you want to finetune only the top layer of the model, set as below.
for param in resnet.parameters():
    param.requires_grad = False

# Replace the top layer for finetuning.
resnet.fc = nn.Linear(resnet.fc.in_features, 100)  # 100 is an example.

# Forward pass.
images = torch.randn(64, 3, 224, 224)
outputs = resnet(images)
print(outputs.size())     # (64, 100)
```

### 保存和读取模型

```python
# Save and load the entire model.
torch.save(resnet, './model.ckpt')
model = torch.load('model.ckpt')

# Save and load only the model parameters (recommended).
torch.save(resnet.state_dict(), 'params.ckpt')
resnet.load_state_dict(torch.load('params.ckpt'))
```

  转载请注明：[Seven的博客](http://sevenold.github.io)