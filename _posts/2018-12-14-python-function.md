---
layout: post
title: "python第六话之函数作用域和匿名函数"
date: 2018-12-13
description: "python函数作用域和匿名函数"
tag: python
---


### 函数作用域和匿名函数

本节知识点：`匿名函数`、`闭包`、`函数作用域`、`递归函数`。

### 匿名函数

上节我们讲过的filter函数，可以过滤出列表中大于10的数据，但是使用都需要提前定义一个函数，有没有更加简便的方式呢？

我们先来复习下`filter`函数：

```python
In [1]: s = [3, 20, 4, 6, 5, 9]

In [2]: s
Out[2]: [3, 20, 4, 6, 5, 9]

In [3]: def fun1(x):
   ...:     return x>5

In [4]: filter(fun1, s)
Out[4]: <filter at 0x7f4a2038f080>

In [5]: list(filter(fun1, s))
Out[5]: [20, 6, 9]
```

> 那如果有大量的操作是不是很麻烦，那有没有更简单的方法呢？

#### lambda

```python
In [6]: s
Out[6]: [3, 20, 4, 6, 5, 9]

In [7]: list(filter(lambda x: x>5, s))
Out[7]: [20, 6, 9]
```

> Python中，lambda函数也叫匿名函数，及即没有具体名称的函数，它允许快速定义单行函数，类似于C语言的宏，可以用在任何需要函数的地方。这区别于def定义的函数。 

#### 语法规则

> lambda  参数： 表达式

### 应用场景

> 简单函数： 简单的函数，可以不用使用def定义一个函数，使用匿名函数。
>
> 函数调用：类似于`filter`、`map`等函数里面，可以使用匿名函数来处理
>
> 提高开发效率：匿名函数的合理使用能够代码更加简洁。

```python
li = [3, 4, 5, 6, 98, 'a', 5.2]
print(sorted(li, key=lambda x: x if isinstance(x, (int, float)) else ord(str(x))))
```

**输出**

```python
[3, 4, 5, 5.2, 6, 'a', 98]
```

#### 总结

> lambda与def的区别： 
> 1）def创建的方法是有名称的，而lambda没有。 
> 2）lambda会返回一个函数对象，但这个对象不会赋给一个标识符，而def则会把函数对象赋值给一个变量（函数名）。 
> 3）lambda只是一个表达式，而def则是一个语句。 
> 4）lambda表达式” : “后面，只能有一个表达式，def则可以有多个。 
> 5）像if（三元运算符可以）或for（列表推导式可以）或print等语句不能用于lambda中，def可以。 
> 6）lambda一般用来定义简单的函数，而def可以定义复杂的函数。 
> 6）lambda函数不能共享给别的程序调用，def可以。 



### 函数作用域

在函数里面也有可以定义变量，那函数里面的变量名如果和函数外面的变量名重名，会相互影响吗？

#### 外部不能访问函数内部变量

```python
def fun1():
	x = 1
	return x
print(x) #不能访问函数里面
```

**输出**

```python
NameError: name 'x' is not defined
```



#### 函数内部能够访问函数外部变量

```python
x = 123
def fun2():
	print(x)
	return x + 1
print(fun2())    #函数里面可以使用全局变量
```

**输出**

```python
123
124
```



#### 函数里面不能修改函数外部变量

```python
x = 123
def fun3():
	x = x + 1   #直接报错，可以访问但是不能修改 
	print(x)
	return x + 1
print(fun3()) 
```

**输出**

```python
UnboundLocalError: local variable 'x' referenced before assignment
```



#### 函数里面和函数外部变量名相同

```python
x = 123
print(x, id(x))
def  fun4():
	x = 456
	print(x, id(x))
	x += 1
	return x
print(fun4())
```

**输出**

```python
123 1722514256
456 1982594811888
457
```

#### 扩展

##### `global`

```python
x = 123
def fun5():
	global x   #如果非要改变全局的变量
	x += 1
	return x
print(fun5())
```

**输出**

```python
124
```

> 函数内部如果需要修改全局变量，就需要使用global修饰变量。

##### `nonlocal`

```python
def  fun6():
	x = 123
	def fun7():
		nonlocal x  #python2没有，python3独有的
		x += 1
		return  x
	return fun7()

print(fun6())
```

**输出**

```python
124
```

> 在函数嵌套的情况下，同样也有函数作用域的问题，但是在Python3中提供了方便，只需要使用nonlocal就可以在里层函数内部修改外部函数的变量。

#### 总结

![1](http://eveseven.oss-cn-shanghai.aliyuncs.com/18-12-13/95671711.jpg)

> 函数内部： 函数内部的变量，作业域只在函数内部，函数内部不可以直接更改函数外部的变量。
>
> global：函数内部如果需要修改全局变量，就需要使用global修饰变量。
>
> nonlocal:  在函数嵌套的情况下，同样也有函数作用域的问题，但是在Python3中提供了方便，只需要使用nonlocal就可以在里层函数内部修改外部函数的变量。

### 闭包

函数里面可以再定义函数，那函数里面定义的函数可以在外面调用吗？

#### 函数调用

```python
g = lambda x: x>10
g
Out[3]: <function __main__.<lambda>(x)>
g(100)
Out[4]: True
h = g
h
Out[6]: <function __main__.<lambda>(x)>
h(111)
Out[7]: True
```

> 在进行函数调用的时候需要加上`()`，才能进行使用。

#### 内嵌函数

```python
def fun6():
    x = 123
    def fun7():
        nonlocal x
        x += 1
        return  x
    return fun7

f7=fun6()
print(f7())
```

**输出**

```python
124
```

#### 总结

> 闭包是函数里面嵌套函数，外层函数返回里层函数，这种情况称之为闭包。
>
> 闭包是概念，不是某种函数类型，和递归的概念类似，就是种特殊的函数调用。
>
> 闭包可以得到外层函数的局部变量，是函数内部和函数外部沟通的桥梁。

### 递归函数

函数里面可以自身调用自身吗

```python
def fun8(n):
    if n == 1:
        return 1
    return fun8(n - 1)* n
print(fun8(8))
```

**输出**

```python
40320
```

> 递归中可以函数自身调用自身，但是使用时类似于条件循环一样，要有递归的终止条件

#### 总结

> 使用递归时，常常可以让代码更加简洁
>
> 递归会占用比较多的内存，当递归次数比较多时，性能就会降低，因此不建议多使用递归

转载请注明：[Seven的博客](http://sevenold.github.io)