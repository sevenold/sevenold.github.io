---
layout: post
title: "python第十一话之文件"
date: 2018-12-19
description: "python文件"
tag: python
---


### 私有属性和私有方法

我们python3中有没有私有属性这种说法？如果有的话有是怎么使用的？

#### ”私有“变量、方法

> 1、封装类的实例上面的“私有”数据，但是Python语言并没有访问控制。 
>
> 2、Python程序员不去依赖语言特性去封装数据，而是通过遵循一定的属性和方法命名规约来达到这个效果。

#### 单下滑线(_)

> 第一个约定是任何以单下划线_开头的名字都应该是内部实现。

```python
class A:
    def __init__(self):
        self._internal = 0 # An internal attribute
        self.public = 1 # A public attribute

    def public_method(self):
        '''
        A public method
        '''
        pass

    def _internal_method(self):
        print('_internal_method')
```

> Python并不会真的阻止别人访问内部名称。但是如果你这么做肯定是不好的，可能会导致脆弱的代码。 同时还要注意到，使用下划线开头的约定同样适用于模块名和模块级别函数。 

```python
a = A()
a._internal_method()
a._internal
```

```python
_internal_method
0
```

#### 双下滑线（__）

你还可能会遇到在类定义中使用两个下划线(__)开头的命名。

```python
class B:
    def __init__(self):
        self.__private = 0

    def __private_method(self):
        print('_B__private_method')

    def public_method(self):
        pass
        self.__private_method()
```

> 使用双下划线开始会导致访问名称变成其他形式。 比如，在前面的类B中，私有属性会被分别重命名为 `_B__private` 和 `_B__private_method` 。 这时候你可能会问这样重命名的目的是什么，答案就是继承——这种属性通过继承是无法被覆盖的。 

```python
b = B()
b._B__private
b._B__private_method()
b.public_method()
```

```python
_B__private_method
_B__private_method
```



> 私有名称 `__private` 和 `__private_method` 被重命名为 `_C__private` 和 `_C__private_method` ，这个跟父类B中的名称是完全不同的。 

```python
class C(B):
    def __init__(self):
        super().__init__()
        self.__private = 1 # Does not override B.__private

    # Does not override B.__private_method()
    def __private_method(self):
        print('_C__private_method')
```

```python
_B__private_method
_B__private_method
_C__private_method
```

### 文件基本操作

我们的程序都是运行在内存中的，内存是不可持久化存储的，那怎样才能持久存储呢？

#### 打开文件

```python
path = 'text.txt'  # 相对路径
path = 'home/seven/text.txt'  # 绝对路径
file = open(path, mode='w+')
```

> 以w+模式打开文件，是为写入和读取的模式，没有文件会新建文件，有文件会清空文件。

#### 文件打开模式

![文件打开模式](http://eveseven.oss-cn-shanghai.aliyuncs.com/18-12-19/31221654.jpg)

> 不同的文件打开模式，对文件的操作有不同，大家一定要注意。



#### 写入文件

```python
file.write('python')
Out[4]: 6
file.write('python2')
Out[5]: 7
```

> 写单个字符串

```python
file.writelines(['1', '2', '3'])
```

> 写一行数据

```python
file.flush()
```

> 本来写入的数据是存在内存里的，使用flush方法，把数据保存到硬盘中。



#### 读取与关闭

```python
file.seek(0) # 把光标移到首位
file.read()
Out[18]: 'python\n\npython3\n\nc++\n\nc\n\njava\n\nmachine learning\n\ndeep learning\n'
```

> 读取全部数据

```python
file.readline()
Out[21]: 'python\n'
file.readline()
Out[22]: '\n'
file.readline()
Out[23]: 'python3\n'
file.readline()
Out[24]: '\n'
file.readline()
Out[25]: 'c++\n'
```

> 一行一行的读取数据

```python
file.readlines()
Out[26]: 
['\n',
 'c\n',
 '\n',
 'java\n',
 '\n',
 'machine learning\n',
 '\n',
 'deep learning\n']

```

> 读取所有行并以列表形式返回

```python
file.flush() # 把内存中的数据保存到硬盘中
file.close() # 关闭并保存文件
file.closed  # 判断文件是否关闭
```

```python
file.close()
file.closed
Out[28]: True
```



#### 查看与移动指针

```python
file.tell()
Out[9]: 50
file.seek(0, 0)  #0代表从文件开头开始算起，1代表从当前位置开始算起，2代表从文件末尾算起。
Out[10]: 0
file.tell()
Out[11]: 0
```

> tell 查看光标位置，seek移动光标的位置。

#### 总结

> 持久存储：保存内存中数据都是易丢失的，只有保存在硬盘中才能持久的存储，保存在硬盘中的基本方法就是把数据写入文件中。
>
> 打开与关闭：在python中文件的打开与关闭变得十分简单快捷，文件在关闭的时候就会自动保存
>
> 写入与读取：文件的写入和读取是必须要十分熟练的内容



### 上下文管理

文件能够自动关闭吗？

```python
with open('test.txt','r') as file:
    st = file.read()
    print(st)
```

```python
c
java
machine learning
deep learning
file.closed
Out[3]: True
```

> with能够自动关闭文件，不需要执行close方法

```python
import time
class RunTime:
    def __enter__(self):
        self.start_time = time.time()
        return self.start_time

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        self.run_time = self.end_time - self.start_time
        print('Time consuming %s ' % self.run_time)

with RunTime():
    for i in range(100000):
        pass
```

```python
Time consuming 0.005983591079711914 
```

> 通过这两个方法可以方便的实现上下文管理
>
> with会把 __enter__ 的返回值赋值给 as 后的变量

#### 总结

> with: 使用with打开文件，则文件不需要自己关闭，会自动的关闭
>
> __enter__: 进入时需要执行的代码，相当于准备工作
>
> __exit__ : 退出时需要执行的代码，相当于收尾工作

### IO流

文件可以持久存储，但是现在类似于临时的一些文件，不需要持久存储，如一些临时的二维码等，这个不需要持久存储，但是却需要短时间内大量读取，这是时候还是只能保存在文件里面吗？

#### [StringIO](https://docs.python.org/3/library/io.html)

```python
In [4]: import io

In [5]: sio = io.StringIO()  # 创建io

In [6]: sio.write('abc')  # 写入数据
Out[6]: 3

In [7]: sio
Out[7]: <_io.StringIO at 0x7f0b775ddaf8>

In [8]: sio.read()
Out[8]: ''

In [9]: sio.seek(0)
Out[9]: 0

In [10]: sio.read()
Out[10]: 'abc'

In [11]: sio.getvalue()  # 读取数据，全部的，不管光标位置
Out[11]: 'abc'

In [12]: sio.close()

In [13]: sio
Out[13]: <_io.StringIO at 0x7f0b775ddaf8>

In [14]: sio.getvalue()
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
<ipython-input-14-2c8cd5e6194b> in <module>()
----> 1 sio.getvalue()

ValueError: I/O operation on closed file

```

> StringIO在内存中如同打开文件一样操作字符串，因此也有文件的很多方法
>
> 当创建的StringIO调用 close() 方法时，在内存中的数据会被丢失 

#### [BytesIO](https://docs.python.org/3/library/io.html)

```python
In [17]: bio = io.BytesIO()  # 创建IO

In [18]: bio
Out[18]: <_io.BytesIO at 0x7f0b775b9150>

In [19]: bio.write(b'abc')  # 写入数据
Out[19]: 3

In [20]: bio.read()
Out[20]: b''

In [21]: bio.seek(0)
Out[21]: 0

In [22]: bio.read()
Out[22]: b'abc'

In [23]: bio.getvalue()  # 读取数据 
Out[23]: b'abc'

In [24]: bio.close()

In [25]: bio.read()
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
<ipython-input-25-dca3ff2736f4> in <module>()
----> 1 bio.read()

ValueError: I/O operation on closed file.

```

> BytesIO和 StringIO 类似，但是BytesIO操作的是 Bytes数据



### 使用工具

文件可以直接新建，但是现在如果需要创建文件夹和移动文件夹怎么办呢？

#### [os](https://docs.python.org/3/library/os.html)   操作系统交互

>  os模块提供python和操作系统交互的接口

##### 直接调用吸引命令

```python
In [1]: import os

In [2]: os.system('ls')
Data  PythonClassEnv  ReadMe.md
Out[2]: 0
```

##### 通用路径操作

```python
In [5]: os.path
Out[5]: <module 'posixpath' from '/usr/lib/python3.5/posixpath.py'>

In [6]: os.path.join(r'Data', r'a')
Out[6]: 'Data/a'
```



##### 文件目录操作

```python
In [7]: os.mkdir('text')

In [8]: os.system('ls')
Data  PythonClassEnv  ReadMe.md  text
Out[8]: 0
    
In [9]: os.rename('text', 'text1')

In [10]: os.system('ls')
Data  PythonClassEnv  ReadMe.md  text1
Out[10]: 0

```

> os 提供了Python和操作系统交互方式，只要是和操作系统相关，就可以尝试在os模块中找方法

#### [shutil](https://docs.python.org/3/library/shutil.html)   高级文件操作

> shutil 模块提供了许多关于文件和文件集合的高级操作

##### 移动文件

```python
In [11]: import shutil

In [12]: shutil.move('text1', 'text')
Out[12]: 'text'

In [13]: os.system('ls')
Data  PythonClassEnv  ReadMe.md  text
Out[13]: 0

```

##### 复制文件夹

```python
In [11]: import shutil

In [12]: shutil.move('text1', 'text')
Out[12]: 'text'

In [13]: os.system('ls')
Data  PythonClassEnv  ReadMe.md  text
Out[13]: 0

```



##### 删除文件

```python
In [11]: import shutil

In [12]: shutil.move('text1', 'text')
Out[12]: 'text'

In [13]: os.system('ls')
Data  PythonClassEnv  ReadMe.md  text
Out[13]: 0

```



转载请注明：[Seven的博客](http://sevenold.github.io)