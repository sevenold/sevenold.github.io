---
layout: post
title: "python第六话之函数基础和函数参数"
date: 2018-12-13
description: "python函数基础和函数参数"
tag: python
---


[TOC]


### 函数基础和函数参数

函数是组织好的，可重复使用的，用来实现单一，或相关联功能的代码段。

函数能提高应用的模块性，和代码的重复利用率。你已经知道Python提供了许多内建函数，比如print()。但你也可以自己创建函数，这被叫做用户自定义函数。

### 函数基础

#### 定义一个函数

> 你可以定义一个由自己想要功能的函数，以下是简单的规则：
>
> - 函数代码块以 **def** 关键词开头，后接函数标识符名称和圆括号 **()**。
> - 任何传入参数和自变量必须放在圆括号中间，圆括号之间可以用于定义参数。
> - 函数的第一行语句可以选择性地使用文档字符串—用于存放函数说明。
> - 函数内容以冒号起始，并且缩进。
> - **return [表达式]** 结束函数，选择性地返回一个值给调用方。不带表达式的return相当于返回 None。

**演示**：

> 我们上节课实现了打印列表，如果我们打印几个列表呢？

```python
li = [1, 0, 5, 7, 9]
for i in li:
    print(i)

print('---------')

li = [1, 'A', 5, 7, 9]
for i in li:
    print(i)

print('---------')
li = [1, 3, 's', 7, 9]
for i in li:
    print(i)
```

**输出结果：**

```python
1
0
5
7
9
---------
1
A
5
7
9
---------
1
3
s
7
9
```

> 以我们上节所讲的知识点，如果要打印三个列表的话，就是上述这种方法，那还有没有更简单的呢？

**演示**：

```
l1 = [1, 0, 5, 7, 9]
l2 = [1, 'A', 5, 7, 9]
l3 = [1, 0, 'S', 7, 9]

def demo(li):
	for i in li:
		print(i)

demo(l1)
print('---------')
demo(l2)
print('---------')
demo(l3)
```

**输出结果**

```python
1
0
5
7
9
---------
1
A
5
7
9
---------
1
0
S
7
9
```

> 上述就是使用函数的形式来实现多个列表的打印，是不是比前面的更简单。

#### 函数的定义

> def  函数名(参数)：
>
> ​        pass
>
> ​        return    表达式
>
> 函数名命名规则： 字母、数字和下划线组成，和变量命名规则一致
>
> return 后面可以返回任意表达式，但不能是赋值语句
>
> 注意：函数名定义和变量名的定义是一样的，只能使用字母、数字和下划线定义，不能以数字开头。

#### 关键字

> 关键字是不能拿来做变量定义的。

**演示：**

```python
In [3]: a
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
<ipython-input-3-3f786850e387> in <module>()
----> 1 a

NameError: name 'a' is not defined

In [4]: def
  File "<ipython-input-4-7b18d017f89f>", line 1
    def
       ^
SyntaxError: invalid syntax
```

> 如果把关键字拿来定义，是会报语法错误的。

```python
In [1]: import keyword

In [2]: print(keyword.kwlist)
['False', 'None', 'True', 'and', 'as', 'assert', 'break', 'class', 'continue', 'def', 'del', 'elif', 'else', 'except', 'finally', 'for', 'from', 'global', 'if', 'import', 'in', 'is', 'lambda', 'nonlocal', 'not', 'or', 'pass', 'raise', 'return', 'try', 'while', 'with', 'yield']
```

> 上述就是整个Python编程语言的全部关键字，在基础阶段都会提到的。

#### 函数调用

```python
l1 = [1, 0, 'S', 7, 9]

def demo(li):
    for i in li:
        print(i)

demo(l1)
```

> 调用方式：函数名（参数）

#### 函数返回

```python
l1 = [1, 0, 'S', 7, 9]

def demo(li):
    for i in li:
        print(i)
    return 'ok'

print(demo(l1))
```

**输出：**

```python
1
0
S
7
9
ok
```

> return：
>
> 注意 return 和 print 的区别，return是函数的返回值，返回值可以赋值给变量，而print只是打印出来

### 函数参数

> 刚才讲到了函数的定义，那函数里面可以传入哪些对象呢？

```python
def demo(x):
    print(x)

demo('demo')
```

**输出**

```python
demo
```

> 如果我们不传值呢？

```python
def demo(x):
    print(x)

demo()
```

**输出**

```python
TypeError: demo() missing 1 required positional argument: 'x'
```

> TypeError：demo()缺少一个必需的位置参数：'x'。

> 传入几个参数呢？

#### 必备参数

> def  func(x):
>
>   pass

```python
def demo(x):
    print(x)

demo(1, 2)
```

**输出**

```python
TypeError: demo() takes 1 positional argument but 2 were given
```

> 一个参数对应一个数值

#### 默认参数

> def  func(x, y=None):
>
>   pass

```python
def demo(x, y=1):
    print(x, y)

demo(1, 2)
demo(3)
```

**输出**

```python
1 2
3 1
```

> y=1.就是默认参数，没有传入新参数的时候，就使用默认参数。

#### 关键字参数

```python
def demo(x, y=1):
    print(x, y)

demo(1, 2)
demo(y="q", x='s')
```

**输出**

```python
1 2
s q
```

> 关键字参数，调用的时候带上参数名。

#### 不定长参数

> def  func(*args, **kwargs):
>
>   pass
>
> 注意：*+参数名

```python
def demo(*args):
    print(args)

demo(1, 2, 3, 4)
demo(1)
```

**输出**

```python
(1, 2, 3, 4)
(1,)
```

> 参数名前面加`*号`是不定长参数，输出是一个元组。

```python
def demo(*a):
    print(*a)  # 加*：去除括号
    print(a)

demo(1, 2, 3, 4)
print('-------')
demo((1, 2, 3, 4))
print('-------')
demo(*(1, 2, 3, 4))
```

**输出**

```python
1 2 3 4
(1, 2, 3, 4)
-------
(1, 2, 3, 4)
((1, 2, 3, 4),)
-------
1 2 3 4
(1, 2, 3, 4)
```

> 加*：去除括号



```python
def demo(**a):
    print(a)

demo(x=1, y=2, s=2)
```

**输出**

```python
{'x': 1, 'y': 2, 's': 2}
```

> 参数名前面加`**号`是不定长参数，输出是一个字典。
>
> 注意：传入的参数是键值对。

#### **演示：**

```python
def demo(*args, **kwargs):
    print(args)
    print(kwargs)

demo(1, 2, 3, x=1, y=2, s=2)
```

**输出**

```python
(1, 2, 3)
{'x': 1, 'y': 2, 's': 2}
```

> 传入的键值对，只能放在最后。

#### 总结：

> 必备参数：在函数调用的时候，必备参数必须要传入
>
> 默认参数： 在函数调用的时候，默认参数可以不传入值，不传入值时，会使用默认参数
>
> 不定长参数：在函数调用的时候，不定长参数可以不传入，也可以传入任意长度。其中定义时，元组形式可以放到参数最前面，字典形式只能放到最后面



### 常见的内置函数

  常见内置函数提供了一些处理的数据的方法，可以帮助我们提高开发速度

#### 常见函数

##### `len`

> 求长度

```python
li = [2,8,5]
In [6]: len(li)
Out[6]: 3
```

##### `min`

> 求最小值

```python
li = [2,8,5]
In [6]: len(li)
Out[6]: 3
```

##### `max`

> 求最大值

```python
li = [2,8,5]
In [8]: max(li)
Out[8]: 8
```

##### `sorted`

> 排序

```python
li = [2,8,5]
In [9]: sorted(li)
Out[9]: [2, 5, 8]
```

##### `reversed`

> 反向

```python
li = [2,8,5]
In [10]: reversed(li)
Out[10]: <list_reverseiterator at 0x7f68aa81af98>

In [11]: list(reversed(li))
Out[11]: [5, 8, 2]
```

##### `sum`

> 求和

```python
li = [2,8,5]
In [12]: sum(li)
Out[12]: 15
```



#### 进制转换函数

##### `bin`

> 二进制

```python
In [13]: bin(12)
Out[13]: '0b1100'
```

##### `oct`

> 八进制

```python
In [16]: oct(18)
Out[16]: '0o22
```

##### `hex`

> 十六进制

```python
In [17]: hex(12)
Out[17]: '0xc'
```

##### `ord`

> 字符转ASCII码

```python
In [19]: ord('a')
Out[19]: 97
```

##### `chr`

> ASCII码转字符

```python
In [20]: chr(97)
Out[20]: 'a'
```

#### 扩展

##### `enumerate`

> 返回一个可以枚举的对象

```python
In [21]: li = ['a','b','c','d']

In [22]: enumerate(li)
Out[22]: <enumerate at 0x7f68aa877d80>

In [23]: list(enumerate(li))
Out[23]: [(0, 'a'), (1, 'b'), (2, 'c'), (3, 'd')]

In [24]: dict(enumerate(li))
Out[24]: {0: 'a', 1: 'b', 2: 'c', 3: 'd'}

```

##### `eval`

> 取出字符串中内容
>
> 将字符串str当成有效的表达式来求值并返回计算结果

```python
In [25]: a = "{'a':1}"

In [26]: eval(a)
Out[26]: {'a': 1}

In [27]: b = '1 + 2 + 3'

In [28]: eval(b)
Out[28]: 6
```

##### `exec`

>  执行字符串或complie方法编译过的字符串，没有返回值

```python
In [29]: s = '''
    ...: z = 10
    ...: su = x + y + z
    ...: print(su)
    ...: print('OK')
    ...: '''

In [30]: x = 1

In [31]: y = 2

In [32]: exec(s)
13
OK

In [33]: exec(s,{'x':0,'y':0})
10
OK

In [34]: exec(s,{'x':0,'y':0},{'y':10,'z':0})  #以字符串为主,以最后的为主
20
OK
```

>  注意：eval 和 exec 是炸弹 能不能就不用，就好像你从不知道这东西一样，除非你足够的熟悉

##### `filter`

> 过滤器

```python
In [38]: def test1(x):
    ...:     return x>10
    ...: l1 = [10,2,20,13,5]

In [39]: filter(test1, l1)
Out[39]: <filter at 0x7f68aa7ecb70>

In [40]: list(filter(test1, l1))
Out[40]: [20, 13]
```

##### `map`

> 对于参数iterable中的每个元素都应用fuction函数，并将结果作为列表返回

```python
In [41]: l2 = [1,2,3]

In [42]: map(str,l2)
Out[42]: <map at 0x7f68aa7ecba8>

In [43]: list(map(str,l2))
Out[43]: ['1', '2', '3']
```

##### `zip`

> 将对象逐一配对

```python
In [44]: l3 = [1,2,3]

In [45]: t1 = ('a','b','c')

In [46]: zip(t1,l3)
Out[46]: <zip at 0x7f68abb3ec48>

In [47]: list(zip(t1,l3))
Out[47]: [('a', 1), ('b', 2), ('c', 3)]

In [48]: dict(zip(t1,l3))
Out[48]: {'a': 1, 'b': 2, 'c': 3}
```

转载请注明：[Seven的博客](http://sevenold.github.io)