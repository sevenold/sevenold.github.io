---
layout: post
title: "python第一话之数值类型和序列类型"
date: 2018-12-01
description: "python数值类型，序列类型"
tag: python
---



[TOC]

### Python的数值类型

Python中的基本数据类型有数值类型、字符串型、列表、元组、字典、集合等。本节介绍数值类型。数值类型包括整型、布尔型、浮点型和复数类型。

#### 基本整形四则运算

用Python实现简单的加减乘除

我们先进入`Python的交互模式`

![python](http://eveseven.oss-cn-shanghai.aliyuncs.com/18-12-1/36226595.jpg)

或者执行`ipython`

![ipython](http://eveseven.oss-cn-shanghai.aliyuncs.com/18-12-1/65669846.jpg)

##### `加法`

```python
In [1]: 1+1
Out[1]: 2

In [2]: 12+2
Out[2]: 14
```

##### `减法`

```python
In [3]: 21-1
Out[3]: 20

In [4]: 20-22
Out[4]: -2
```

##### `乘法`

```python
In [5]: 2*2
Out[5]: 4

In [6]: 10*10
Out[6]: 100
```

##### `除法`

```python
In [9]: 4/2
Out[9]: 2.0

In [10]: 2/2
Out[10]: 1.0
```

> 以上就是我们所遇到的一些基本的加减乘除运算，接下来我们再看看其他形式的扩展。

#### 保存计算结果-基本的赋值运算

在我们后面的学习中，经常会遇到需要保存计算结果的情况，下面我们来看一下如何保存我们的计算结果。

```python
In [11]: a = 1+1

In [12]: 1+1
Out[12]: 2

In [13]: a
Out[13]: 2
```

#### 基本整数与小数四则运算

计算机在计算的时候，除了整数运算，还有小数运算，还有小数和整数的混合运算。

##### `加法`

```python
In [14]: 2+1.2
Out[14]: 3.2
```

##### `减法`

```python
In [16]: 2.2-2
Out[16]: 0.20000000000000018
```

##### `乘法`

```python
In [17]: 3.3*2
Out[17]: 6.6
```

##### `除法`

```python
In [18]: 2.4/4
Out[18]: 0.6
```

> 小数一般是`float`类型
>
> ```python
> In [19]: b = 3+2.2
> 
> In [20]: type(b)
> Out[20]: float
> ```
>
> 但是`浮点数`不是我们真正看到的数，比如`1.2`实际是`1.1999999999`
>
> 所以小数计算都是不精确的，那么我们如何进行精确计算呢？

#### 精确计算-decimal的运算

在`Python`中如果我们要实现精确计算，我们是使用`decimal`这个库函数

```python
In [27]: import decimal 

In [28]: a = decimal.Decimal('2.2')

In [29]: b = decimal.Decimal('2')

In [30]: a-b
Out[30]: Decimal('0.2')
```

#### 布尔型的计算

`布尔型变量`只有`True`和`False`两种情况，`True`就是`1`，`False`就是`0`。

##### `基本情况`

```python
In [34]: a = True

In [35]: b = False

In [36]: a
Out[36]: True

In [37]: b
Out[37]: False

In [38]: type(a)
Out[38]: bool
```

##### `运算`

```python
In [39]: a + 1
Out[39]: 2

In [40]: a + a 
Out[40]: 2

In [41]: a + b
Out[41]: 1
```

#### 复数类型

```python
In [42]: 1 + 2j
Out[42]: (1+2j)

In [43]: a = 1 + 2j

In [44]: type(a)
Out[44]: complex
```

#### 总结

整型、布尔型、浮点型和复数类型

```python
In [45]: type(2)
Out[45]: int

In [46]: type(True)
Out[46]: bool

In [47]: type(1.2)
Out[47]: float

In [48]: type(1+2j)
Out[48]: complex
```

#### 扩展

##### `整除-向下取整`

`//`在`Python`中是指的`向下取整`：1 < 1.n < 2  ==> 1

```python
In [49]: 2.3 / 2
Out[49]: 1.15

In [50]: 2.3 // 2
Out[50]: 1.0
```

##### `整除-向上取整`

`向上取整`：使用`math`库函数实现

```python
In [52]: import math

In [53]: math.ceil(2.3/2)
Out[53]: 2
```

##### `幂运算`

```python
In [54]: 2 * 2 *2
Out[54]: 8

In [55]: 2 **3
Out[55]: 8
```

##### `取余`

```python
In [56]: 6 % 4
Out[56]: 2
```

这就是我们常见的数值类型的相关使用和运算。

### Python的序列类型

前面我们遇到的都是一些数值类型的使用与运算，但是如果我想在Python中表示字母怎么办呢？这就是我们接下来看一下序列类型。

#### 字符串-`str`

```python
In [57]: 'abc'
Out[57]: 'abc'

In [58]: type('abc')
Out[58]: str

In [59]: "I'm seven"
Out[59]: "I'm seven"

In [60]: """abcde
    ...: fghijk"""
Out[60]: 'abcde\nfghijk'
```

##### `字符串的使用`

> 字符串是通过下标索引值来获取对应值的。

```python
In [74]: a = 'abc'

In [75]: a[0]
Out[75]: 'a'

In [76]: a[2]
Out[76]: 'c'

In [77]: a[1]
Out[77]: 'b'
```



#### 列表-`list`

> 数值和字符串的混合使用

```python
In [61]: ['abc', 123, 'dde']
Out[61]: ['abc', 123, 'dde']

In [62]: type(['abc', 123, 'dde'])
Out[62]: list
```

##### `列表的使用-简单取值`

> 根据下标取出对应数据，下标从0开始计数：0,1,2,3...

```python
In [65]: a = ['abc', 123, 'dde']

In [66]: a[0]
Out[66]: 'abc'

In [67]: a[1]
Out[67]: 123

In [68]: a[2]
Out[68]: 'dde'
```

##### `列表的使用-切片`

> 通过`list[start_index : end_index : stride ]`来进行`切片`，`切片`方式类似数学中的`左闭右开区间`
>
> `start_index:` 开始的索引值
>
> `end_index:`  结束的索引值
>
> `stride:` 步长

```python
In [78]: a = ['abc', 123, 'dde']

In [79]: a[:1]
Out[79]: ['abc']

In [80]: a[:2]
Out[80]: ['abc', 123]

In [81]: a[1:]
Out[81]: [123, 'dde']

In [82]: a[1:3]
Out[82]: [123, 'dde']
    
In [84]: a = [1,2,3,4,5,6,'a','b','c']

In [85]: a[1:6:2]
Out[85]: [2, 4, 6]

```

其中：`[-1]:` 表示倒着计数

```python
In [92]: a = [1,2,3,4,5,6,'a','b','c']

In [93]: a[1:]
Out[93]: [2, 3, 4, 5, 6, 'a', 'b', 'c']

In [94]: a[-1:]
Out[94]: ['c']

In [95]: a[2:-1]
Out[95]: [3, 4, 5, 6, 'a', 'b']

In [96]: a[2:8]
Out[96]: [3, 4, 5, 6, 'a', 'b']
```



#### 元组-`tuple`

```python
In [63]: (123, 'abc', 'seven')
Out[63]: (123, 'abc', 'seven')

In [64]: type((123, 'abc', 'seven'))
Out[64]: tuple
```

##### `元组的使用`

> 元组的使用是和列表的类似的。

```python
In [70]: b = (123, 'abc', 'seven')

In [71]: b[1]
Out[71]: 'abc'

In [72]: b[0]
Out[72]: 123

In [73]: b[2]
Out[73]: 'seven'
```



#### 类型转换

> 我们经常会使用到这几种类型， 所以这几种类型间的转换又尤为关键

##### `字符串转列表`

```python
In [111]: a = 'abcd'

In [112]: b = list(a)

In [113]: type(a)
Out[113]: str

In [114]: type(b)
Out[114]: list

In [115]: b
Out[115]: ['a', 'b', 'c', 'd']
```

##### `列表转字符串`

> 列表变成字符串会把列表里的中括号和空格也变成字符串

```python
In [123]: a = ['a', 'b', 'c', 'd']

In [124]: b = str(a)

In [125]: type(a)
Out[125]: list

In [126]: type(b)
Out[126]: str

In [127]: b
Out[127]: "['a', 'b', 'c', 'd']"
```



##### `字符串转元组`

```python
In [116]: a = 'abcd'

In [117]: b = tuple(a)

In [118]: type(a)
Out[118]: str

In [119]: type(b)
Out[119]: tuple

In [120]: b
Out[120]: ('a', 'b', 'c', 'd')
```

##### `元组转字符串`

> 列表变成字符串会把列表里的小括号和空格也变成字符串

```python
In [128]: a = ('a', 'b', 'c', 'd')

In [129]: b = str(a)

In [130]: type(a)
Out[130]: tuple

In [131]: type(b)
Out[131]: str

In [132]: b
Out[132]: "('a', 'b', 'c', 'd')"
```



#### 元组和列表的区别

> 在我们前面的接触过程中，列表和元组基本的功能是一样的，那列表和元组都存在，是为什么呢？

```python
In [133]: a = ('a', 'b', 'c', 'd')

In [134]: b = ['a', 'b', 'c', 'd']

In [135]: a[1]
Out[135]: 'b'

In [136]: b[1]
Out[136]: 'b'

In [137]: b[1] = 3

In [138]: b
Out[138]: ['a', 3, 'c', 'd']

In [139]: a[1] = 3
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
<ipython-input-139-23f2cf2bdf70> in <module>()
----> 1 a[1] = 3

TypeError: 'tuple' object does not support item assignment
```

`扩展：`

```python
In [140]: a = 'abcdefg'

In [141]: type(a)
Out[141]: str

In [142]: a[1]
Out[142]: 'b'

In [143]: a[1] = 's'
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
<ipython-input-143-55e6e4038777> in <module>()
----> 1 a[1] = 's'

TypeError: 'str' object does not support item assignment
```

> 总结：列表：可变
>
> ​		元组： 不可变
>
> ​             字符串： 不可变
>
> 所以，在序列类型中，只有列表才是可变的类型。



#### 更改字符串和元组的元素

> 前面我们讲了在序列类型中，只有列表才是可变的类型。那我们如何来更改不可变数据类型的元素呢？

```python
In [151]: a = 'abcdefg'

In [152]: a = a[1:]

In [153]: a
Out[153]: 'bcdefg'

In [154]: a = ('a', 'b', 'c', 'd')

In [155]: a = a[1:3]

In [156]: a
Out[156]: ('b', 'c')
```



#### 拆包

> 元组拆包可以应用到任何迭代对象上， 唯一的要求是， 被可迭代对象中的元素数量必须要和这些元素的元组的空档数一致， 除非我们用* 来表示忽略多余的元素。

```python
In [157]: a = ('a', 'b', 'c', 'd')

In [158]: x,*y,z = a

In [159]: x
Out[159]: 'a'

In [160]: z
Out[160]: 'd'

In [161]: y
Out[161]: ['b', 'c']
In [162]: x,y,z = a
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
<ipython-input-162-57ae45ef0060> in <module>()
----> 1 x,y,z = a

ValueError: too many values to unpack (expected 3)
```

> `x` 接收第一个元素，`z` 接收最后一个元素，由于`y`前面有`*`号，所以剩余的元素由`y`接收
>
> 总结：有多少个元素就需要多少个变量来接收，除非有`*`号，不然就会报错。

```python
In [163]: x,y,z = a,a,a

In [164]: x
Out[164]: ('a', 'b', 'c', 'd')

In [165]: y
Out[165]: ('a', 'b', 'c', 'd')

In [166]: z
Out[166]: ('a', 'b', 'c', 'd')
```



#### 变量的赋值

> 变量的保存都是保存在内存中
>
> 注意：`变量`是没有类型的，有类型的是他所`指向的数据`

```python
In [167]: a = 123

In [168]: id(a)
Out[168]: 10923232

In [169]: b = 'abc'

In [170]: id(b)
Out[170]: 140343125492152

In [171]: a = '111'

In [172]: id(a)
Out[172]: 140342939172344
```

> `id()`：查看数据的地址
>
> 总结：赋值给变量是保存在内存中，重新赋值后，变量指向新的地址



#### 变量的引用-成员运算

> 通过`in`或者`not in` 来进行成员运算

```python
In [173]: a = 'abcd123'

In [174]: 'c' in a
Out[174]: True

In [175]: '8' in a
Out[175]: False

In [176]: 'a' not in a
Out[176]: False

In [177]: 'q' not in a
Out[177]: True
```

转载请注明：[Seven的博客](http://sevenold.github.io)