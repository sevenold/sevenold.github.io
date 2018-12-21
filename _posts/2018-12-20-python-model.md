---
layout: post
title: "python第十三话之列表推导式、迭代器生成器，模块和包"
date: 2018-12-21
description: "python列表推导式、迭代器生成器，模块和包"
tag: python
---

### 推导表达式

> 得到一个元素为1到10的列表，可以怎么做？

#### 列表

##### **循环添加**

**方法一：**

```python
x = list(range(1,11))
```

```python
x
Out[3]: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
```

**方法二：**

```python
li = []
for i in range(1,11):
    li.append(i)
```

```python
li
Out[3]: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
```

> 这是我们前面所总结过的生成一个列表的方法。

##### **列表推导**

```python
li = [i for i in range(1,11)]
```

```python
li
Out[3]: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
```

> 通过列表推导，精简了代码。

##### 运行时间对比

```python
import time
s = time.time()
for i in range(1000000):
    x = list(range(1,11))
e = time.time()
print('list: ', e-s)


f = time.time()
for i in range(1000000):
    li = []
    for i in range(1,11):
        li.append(i)
end = time.time()

print('append: ', end-f)

ff = time.time()
for i in range(1000000):
    li = [i for i in range(1,11)]
end1 = time.time()

print('[]: ', end1-ff)
```

```python
list:  0.7910020351409912
append:  2.110013246536255
[]:  1.0232622623443604
```

> 可以看出列表推导的方式是比append方法更高效的，而list强制转换的是可能出现问题，所以总的来说就是推荐使用列表推导的方法。

#### **列表推导+条件判断**

```python
l2 = [i for i in range(1,11) if i % 2 == 0]
```

```python
l2
Out[4]: [2, 4, 6, 8, 10]
```

> 单独的if条件判断只能放在后面

#### **列表推导+三目运算**

```python
l2 = [i if i % 2 == 0 else 0 for i in range(1,11)]
```

```python
l2
Out[3]: [0, 2, 0, 4, 0, 6, 0, 8, 0, 10]
```

> 如果是if+else的判断条件，就需要放前面。

#### 集合

```python
se = {i for i in range(1,11)}
```

```python
se
Out[4]: {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
```

#### 字典

```python
li = list(range(1,11))
di = {i:j for i,j in enumerate(li)}
```

```python
di
Out[3]: {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7, 7: 8, 8: 9, 9: 10}
```

#### 总结

> 推导表达式相对于for循环来处理数据，要更加的方便。
>
> 列表推导表达式使用更加的广泛。

### 迭代器和生成器

#### 迭代器

列表推导是往列表中一个一个地放入数据，那如果一个一个地取出数据呢？

```python
li = list(range(1, 11))
it = iter(li)
it2 = li.__iter__()
```

```python
it
Out[3]: <list_iterator at 0x14c69f4b208>
it2
Out[4]: <list_iterator at 0x14c69f4b6a0>
```

> 生成迭代器

##### 迭代器对象

```python
dir(list)
sir(str)
dir(tuple)
```

> 迭代器对象本身需要支持以下两种方法，它们一起构成迭代器协议：

```python
iterator.__iter__()
iterator.__next__()
```

##### 取值

```python
next(it)
Out[12]: 1
next(it)
Out[13]: 2
next(it)
Out[14]: 3
it.__next__()
Out[15]: 4
it.__next__()
Out[16]: 5
it.__next__()
Out[17]: 6

next(it)
Traceback (most recent call last):
  File "C:\Program Files\Python36\lib\site-packages\IPython\core\interactiveshell.py", line 3265, in run_code
    exec(code_obj, self.user_global_ns, self.user_ns)
  File "<ipython-input-13-bc1ab118995a>", line 1, in <module>
    next(it)
StopIteration
```

> 通过
>
> next(iterator)
>
> iterator.__next__()
>
> 来进行取值
>
> 注意：如果迭代器值取完之后，会返回 StopIteration 错误

##### 自定义迭代器

```python
class TupleIter:
    def __init__(self, li):
        self.li = li
        self._index = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self._index < len(self.li):
            index = self.li[self._index]
            self._index += 1
            return index
        else:
            raise StopIteration


t1 = (1, 2, 3, 4, 5)
a = TupleIter(t1)
```

```python
a
Out[3]: <__main__.TupleIter at 0x1376ba4b358>
next(a)
Out[4]: 1
a.__next__()
Out[5]: 2
```

> 可以自己定义iter和next方法来自定义迭代器。

#### 生成器

```python
def func(n):
    i = 0
    while i < n:
        yield i
        i += 1


s = func(10)
```

```python
s
Out[3]: <generator object func at 0x000001442E4FE990>
[i for i in s]
Out[4]: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
```

> 迭代器提供了一个实现迭代器协议的简便方法
>
> yield 表达式只能在函数中使用,在函数体中使用 yield 表达式可以使函数成为一个生成器



```python
def func(end):
    n, a, b = 0, 0, 1
    while n < end:
        yield b
        a, b = b, a + b
        n += 1


g = func(10)
```

```python
g
Out[3]: <generator object func at 0x00000238904BD990>
[i for i in g]
Out[4]: [1, 1, 2, 3, 5, 8, 13, 21, 34, 55]
```

> 通过生成器就生成了一个迭代器，通过dir(g)就能查出iter和next的方法。就比我们自定义迭代器更简单。

```python
def func(end):
    n, a, b = 0, 0, 1
    while n < end:
        print('*'*5, a, b)
        yield b
        print('-' * 5, a, b)
        a, b = b, a + b
        n += 1


g = func(10)
```

```python
next(g)
***** 0 1
Out[3]: 1
next(g)
----- 0 1
***** 1 1
Out[4]: 1
next(g)
----- 1 1
***** 1 2
Out[5]: 2
```

> yield 可以返回表达式结果，并且暂定函数执行

> 通过迭代器去生成斐波拉契数列要比直接得到更加节省内存。

>    总结：yield只能在函数里面使用。

### 模块

在另一个py文件中的对象如何导入到当前的py文件中呢？

#### 模块

> 在python中，模块就是一个py文件，可以使用下面两种方法导入

```python
import 
from import 
from import as
```

```python
import datetime
from datetime import datetime (as this_datetime)
```

> 在同一目录下，可直接使用上面两种方法去导入；在不同目录下，需要使用  sys.path  添加路径
>
> sys.path.append('path')

![路径](http://eveseven.oss-cn-shanghai.aliyuncs.com/18-12-21/19249391.jpg)

```python
from pages import a
a.demo()
```

> 不同的文件夹导入模块可以用这个两种方式。

> 在python3中导入后，会在当前路径下生成一个\_\_pycache\_\_  文件夹



#### sys模块

> sys 模块提供了与python解释器交互的函数，在python中它是始终可以导入使用的.

##### [sys.argv](https://docs.python.org/3/library/sys.html)   

```python
import sys

print(sys.argv)
```

```python
$ python b.py 123 456
['b.py', '123', '456']
```

> 获取终端命令行输入

##### [sys.path](https://docs.python.org/3/library/sys.html)   

```python
import sys

print(sys.path)
```

>  解释器模块导入查找路径

#### if\_\_name\_\_ == '\_\_main\_\_':

##### [\_\_name\_\_](https://docs.python.org/3/reference/import.html?highlight=__name__)

> python会自动的给模块加上这个属性
>
> 如果模块是被直接调用的，则 __name__ 的值是 \_\_main\_\_否则就是该模块的模块名 

```python
pages.a a
__main__ b
```

##### 注意

> if \_\_name\_\_ == '\_\_main\_\_': 该语句可以控制代码在被其他模块导入时不被执行

```python
__main__ b
```



### 包和包管理

如果模块太多了，怎么方便的去管理呢？

####  包概念

> 把很多模块放到一个文件夹里面，就可以形成一个包.

#### 包管理

> 当把很多模块放在文件中时，为了方便引用包中的模块，引入了包管理

#### \_\_init\_\_.py

> 在包管理中，加入此模块，则包名可以直接通过属性访问的方式，访问此模块内的对象，此模块不加上可能不会报错，但是规范是要加上，文件内容可以为空

#### 相对路径导入

> 在包管理中，可以通过. (一个点) 和 .. (两个点)分别来导入同层和上一层的模块

![包管理](http://eveseven.oss-cn-shanghai.aliyuncs.com/18-12-21/51214573.jpg)



##### 引入作用

> 在包中，如果包中模块要导入同一包中的其他模块，就必须使用此方法导入.



##### 使用方法

```python
from  .module(..module)  import obj   (as  new_name)
```



##### 引入之后的影响

> 当一个模块中出现此导入方式，则该模块不能被直接运行，只能被导入

转载请注明：[Seven的博客](http://sevenold.github.io)