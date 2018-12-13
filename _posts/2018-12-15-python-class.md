---
layout: post
title: "python第八话之类定义、属性和继承"
date: 2018-12-14
description: "python类定义、属性和继承"
tag: python
---


### 类定义、属性和继承

面向对象是一种编程思想，所以这一章内容会比较抽象，大家可以先学会怎么去写，后面用的多了写的多了自然就理解了。在第一章中大概阐述了python中的类与类型，前面学过的基本数据类型就是类，这里就来自定义类。

### 类定义

之前我们在数据类型里面学习到了列表的方法，那是怎么做的可以让列表里面放下这么多方法呢？

```python
class Abc:
    def fun1(self):
        print('this is fun1')

    def fun2(self):
        print('this is fun2')

a=Abc()
print(a)
print(a.fun1())
print(a.fun2())
```

**输出**

```python
<__main__.Abc object at 0x0000024ED872A908>
this is fun1
None
this is fun2
None
```

> cla  = ClassName()
>
> cla.fun1()
>
> cla.fun2()
>
> 实例化之后，可以实现类似于列表中方法的定义形式

#### 总结

> 定义：累得定义使用关键字 class
>
> 封装：类可以把各种对象组织在一起，通过.(点)运算符来调用类中封装好的对象。
>
> 概念：类就像是我们平时说的名词，一个称呼，但是却不是一个具体的实例，比如说：我们都是人，但是人这个名词，不能具体指代你我，我们会用一个人的名字去指代一个具体的人，这个过程就类似于实例化。

### 属性

1.类数据属性。类属性是可以直接通过“类名.属性名”来访问和修改。类属性是这个类的所有实例对象所共有的属性，任意一个实例对象都可以访问并修改这个属性（私有隐藏除外）。

2.实例数据属性。在属性前面加了self标识的属性为实例的属性，在定义的时候用的self加属性名字的形式，在查看实例的属性时就是通过实例的名称+‘.’+属性名来访问实例属性。

3.方法属性。定义属性方法的内容是函数，函数的第一个参数是self，代表实例本身。

#### **举个栗子**

```python
class Animal:
    eye = 2  # 类属性

    def __init__(self, name, food):
        self.name = name  # 实例属性
        self.food = food

    def play(self):
        print('hahaha')
```

#### 实例化

```python
cat = Animal('cat','fish') #先不传值
cat.play()
```

**输出**

```python
hahaha
```

> 类的实例化，实例化后会自动执行__init__这个初始化函数。

#### 实例属性访问

```python
print(cat.name)
print(cat.food)
```

**输出**

```python
hahaha
cat
fish
```

> 实例的属性，实例自己可以访问，定义时有加self，不可以 ClassName. attribute（类名.属性）

#### 类属性

```python
print(Animal.eye)
print(cat.eye)
```

**输出**

```python
2
2
```

> 直接定义在类中，类和实例都可以访问，没有加self
>
> ​       可以  ClassName. attribute（类名.属性）

#### 扩展

```python
print(Animal.name)
```

**输出**

```python
AttributeError: type object 'Animal' has no attribute 'name'
```

> 类只能访问类属性，不能访问实例属性。

#### 总结

> 类属性：类的属性，类名和实例都可以调用，相当于类和实例公用的变量
>
> 实例属性：实例自己的属性，类不能访问，其他的实例也不能访问
>
> 属性调用： 通过属性调用可以直接得到属性的属性值

### 方法

类中的方法，就是函数，但是被称之为方法，在类中的方法，在被实例调用的时候会自动传入实例本身，因此，在一般情况下，需要在参数中加入self。

```python
class Animal:
    eye = 2  # 类属性

    def __init__(self, name, food):
        self.name = name  # 实例属性
        self.food = food

    def play(self):
        print('hahaha')
```

#### 方法调用

```python
cat = Animal('cat','fish') 
cat.play()
```

**输出**

```python
hahaha
```

> 类中的self指代的就是实例本身

#### 扩展

```python
cat = Animal('cat','fish') 
Animal.play(cat)
```

**输出**

```python
hahaha
```



### 继承

如果在B类中定义一个方法，但是这个方法已经在A类中被定义过了，那怎样在B类中使用A类中的方法呢？

```python
class Animal:
    eye = 2

    def __init__(self, name, food):
        self.name = name
        self.food = food

    def play(self):
        print('hahaha')


class Dog(Animal):
     def wangwang(self):
         print('汪汪汪！！%s' %self.name)


demo = Dog('旺财', '骨头')
print(demo.name)
print(demo.food)
print(demo.wangwang())
```

**输出**

```python
旺财
骨头
汪汪汪！！旺财
```

#### 语法规则

> class    A:
>
>   def  play(slef):
>
>   print('hahaha ')
>
>
>
> class  B(A):
>
>   pass



#### 总结

> ​      类的继承可以让子类将父类的全部方法和属性继承过来.
>
> ​      在python3中，默认继承object类

转载请注明：[Seven的博客](http://sevenold.github.io)