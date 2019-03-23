---
layout: post
title: "pytorch基本数据类型及常用计算API的使用"
date: 2019-03-23
description: "pytorch基本数据类型及常用计算API的使用"
tag: Pytorch
---

### Tensor的数据类型

#### **torch.FloatTensor ：**

>  用于生成数据类型为浮点型的Tensor，传递给torch.FloatTensor的参数可以是一个列表，也可以是一个维度值

```python
a = torch.FloatTensor(2, 3)  # 随机生成
b = torch.FloatTensor([2, 3, 4, 5])  # 指定生成
print(a)
print(b)
```

> tensor([[0, 0, 0],[0, 0, 0]], dtype=torch.int32)
> tensor([2, 3, 4, 5], dtype=torch.int32)

#### **torch.IntTensor ：**

> 用于生成数据类型为整型的 Tensor，传递给 torch.IntTensor的参数可以是一个列表，也可以是一个维度值

```python
a = torch.IntTensor(2, 3)  # 随机生成
b = torch.IntTensor([2, 3, 4, 5])  # 指定生成
print(a)
print(b)
```

> tensor([[0, 0, 0],
>         [0, 0, 0]], dtype=torch.int32)
> tensor([2, 3, 4, 5], dtype=torch.int32)

#### **torch.rand ：**

>  用于生成数据类型为浮点型且维度指定的随机Tensor，和在NumPy中使用numpy.rand生成随机数的方法类似，随机生成的浮点数据在0～1区间均匀分布

```python
a = torch.rand(2, 3)
print(a)
```

> tensor([[0.0584, 0.8775, 0.6964],
>         [0.4844, 0.4144, 0.8594]])

#### **torch.randn ：**

>  用于生成数据类型为浮点型且维度指定的随机Tensor，和在NumPy中使用numpy.randn生成随机数的方法类似，随机生成的浮点数的取值满足均值为0、方差为1的正太分布

```python
a = torch.randn(2, 3)
print(a)
```

> tensor([[-0.8214, -0.0331,  0.7489],
>         [-0.3789, -1.0754, -2.1049]])

#### **torch.range** ：

>  用于生成数据类型为浮点型且自定义起始范围和结束范围的Tensor，所以传递给torch.range的参数有三个，分别是范围的起始值、范围的结束值和步长，其中，步长用于指定从起始值到结束值的每步的数据间隔

```python
a = torch.arange(1, 20)
print(a)
```

> tensor([ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
>         19])

#### **torch.zeros ：**

>  用于生成数据类型为浮点型且维度指定的Tensor，不过这个浮点型的Tensor中的元素值全部为0

```python
a = torch.zeros(2, 3)
print(a)
```

> tensor([[0., 0., 0.],
>         [0., 0., 0.]])

### Tensor的运算

#### **torch.abs ：**

>  将参数传递到torch.abs后返回输入参数的绝对值作为输出，输入参数必须是一个Tensor数据类型的变量

```python
a = torch.randn(2, 3)
print(a)
b = torch.abs(a)
print(b)
```

> tensor([[ 1.0926,  1.1767, -1.2057],
>         [-0.5718,  0.8411,  0.4633]])
> tensor([[1.0926, 1.1767, 1.2057],
>         [0.5718, 0.8411, 0.4633]])

#### **torch.add ：**

> 将参数传递到 torch.add后返回输入参数的求和结果作为输出，输入参数既可以全部是Tensor数据类型的变量，也可以一个是Tensor数据类型的变量，另一个是标量

```python
a = torch.randn(2, 3)
print(a)
b = torch.randn(2, 3)
print(b)
c = torch.add(a, b)
print(c)
d = torch.randn(2, 3)
print(d)
e = torch.add(c, d)
print(e)
```

> tensor([[ 0.5688, -1.0098,  0.0875],
>         [-1.1604,  0.4908, -0.0372]])
> tensor([[-0.9039,  1.5682,  0.4096],
>         [ 1.2196,  0.9986, -0.2263]])
> tensor([[-0.3351,  0.5584,  0.4971],
>         [ 0.0593,  1.4895, -0.2635]])
> tensor([[-1.3956, -0.3117,  0.2534],
>         [-0.1104,  0.1590, -1.3836]])
> tensor([[-1.7307,  0.2467,  0.7505],
>         [-0.0512,  1.6484, -1.6471]])

#### **torch.clamp** ：

> 对输入参数按照自定义的范围进行裁剪，最后将参数裁剪的结果作为输出。
>
> - 所以输入参数一共有三个，分别是需要进行裁剪的Tensor数据类型的变量、裁剪的上边界和裁剪的下边界，
> - 具体的裁剪过程是：使用变量中的每个元素分别和裁剪的上边界及裁剪的下边界的值进行比较，如果元素的值小于裁剪的下边界的值，该元素就被重写成裁剪的下边界的值；
> - 同理，如果元素的值大于裁剪的上边界的值，该元素就被重写成裁剪的上边界的值。

```python
a = torch.randn(2, 3)
print(a)
b = torch.clamp(a, -0.1, 0.1)
print(b)
```

> tensor([[ 0.2453, -2.7730, -0.1356],
>         [-0.1726, -0.7038, -0.8896]])
> tensor([[ 0.1000, -0.1000, -0.1000],
>         [-0.1000, -0.1000, -0.1000]])

#### **torch.div ：**

> 将参数传递到torch.div后返回输入参数的求商结果作为输出，同样，参与运算的参数可以全部是Tensor数据类型的变量，也可以是Tensor数据类型的变量和标量的组合

```python
a = torch.randn(2, 3)
print(a)
b = torch.randn(2, 3)
print(b)
c = torch.div(a, b)
print(c)
d = torch.randn(2, 3)
print(d)
e = torch.div(d, 10)
print(e)
```

> tensor([[-1.7551,  1.1126, -0.4721],
>         [-1.3093, -0.6008,  0.8263]])
> tensor([[ 0.3355, -1.0047,  0.5603],
>         [-0.4188,  1.9017,  0.7473]])
> tensor([[-5.2315, -1.1074, -0.8427],
>         [ 3.1263, -0.3159,  1.1058]])
> tensor([[ 1.5984, -0.5883,  0.1418],
>         [-0.8639, -0.7141, -1.2127]])
> tensor([[ 0.1598, -0.0588,  0.0142],
>         [-0.0864, -0.0714, -0.1213]])

#### **torch.mul ：**

> 将参数传递到 torch.mul后返回输入参数求积的结果作为输出，参与运算的参数可以全部是Tensor数据类型的变量，也可以是Tensor数据类型的变量和标量的组合

```python
a = torch.randn(2, 3)
print(a)
b = torch.randn(2, 3)
print(b)
c = torch.mul(a, b)
print(c)
d = torch.randn(2, 3)
print(d)
e = torch.mul(d, 10)
print(e)
```

> tensor([[ 1.8059, -0.1371,  0.3777],
>         [-0.4319,  1.4183,  2.4060]])
> tensor([[ 0.9212, -1.1527,  0.1132],
>         [-0.9209, -0.3348,  0.7959]])
> tensor([[ 1.6636,  0.1581,  0.0428],
>         [ 0.3977, -0.4748,  1.9149]])
> tensor([[ 0.6092,  0.0987,  1.0230],
>         [ 0.5443,  1.9293, -0.0953]])
> tensor([[ 6.0925,  0.9873, 10.2297],
>         [ 5.4433, 19.2932, -0.9529]])

#### **torch.pow ：**

> 将参数传递到torch.pow后返回输入参数的求幂结果作为输出，参与运算的参数可以全部是Tensor数据类型的变量，也可以是Tensor数据类型的变量和标量的组合

```python
a = torch.randn(2, 3)
print(a)
b = torch.pow(a, 2)
print(a)
```

> tensor([[0.4679, 0.7738, 0.0807],
>         [0.7510, 1.6373, 0.1373]])
> tensor([[0.4679, 0.7738, 0.0807],
>         [0.7510, 1.6373, 0.1373]])

**torch.mm ：**

> #### 将参数传递到 torch.mm后返回输入参数的求积结果作为输出，不过这个求积的方式和之前的torch.mul运算方式不太一样，torch.mm运用矩阵之间的乘法规则进行计算，所以被传入的参数会被当作矩阵进行处理，参数的维度自然也要满足矩阵乘法的前提条件，即前一个矩阵的行数必须和后一个矩阵的列数相等，否则不能进行计算。

```python
a = torch.randn(2, 3)
print(a)
b = torch.randn(3, 2)
print(b)
c = torch.mm(a, b)
print(c)
```

> tensor([[ 0.1439, -0.0668, -1.4954],
>         [-0.4558,  0.1897,  0.6573]])
> tensor([[ 0.4398,  1.0332],
>         [ 2.2942, -1.1032],
>         [-0.1976, -0.3176]])
> tensor([[ 0.2056,  0.6973],
>         [ 0.1049, -0.8890]])

#### **torch.mv ：**

>  将参数传递到torch.mv后返回输入参数的求积结果作为输出，torch.mv运用矩阵与向量之间的乘法规则进行计算，被传入的参数中的第1个参数代表矩阵，第2个参数代表向量，顺序不能颠倒。

```python
a = torch.randn(2, 3)
print(a)
b = torch.randn(3)
print(b)
c = torch.mv(a, b)
print(c)
```

> tensor([[ 0.8297, -0.2407, -0.2188],
>         [-1.0946, -1.5037, -0.0340]])
> tensor([2.6157, 0.4282, 1.2846])
> tensor([ 1.7861, -3.5507])

  转载请注明：[Seven的博客](http://sevenold.github.io)