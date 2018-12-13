---
layout: post
title: "python第九话之多继承和魔法方法"
date: 2018-12-13
description: "python多继承和魔法方法"
tag: python
---

### 多继承和魔法方法

在上一节中，我们讲了类的定义，属性和方法，那么我们这个节课来看看`多继承`和`魔法方法`。

### 多继承

在上节我们讲到了继承，一个类可以继承一个类，继承之后可以把父类所有的方法和属性都直接继承过来，那一个类可以继承多个类呢？

如果可以继承多个类的话，那如果两个父类中有一样的方法的情况下，子类继承哪一个呢？

#### 演示

```python
class Base:
    def play(self):
        return 'this is Base'


class A(Base):
    def play(self):
        return 'this is A'


class B(Base):
    def play(self):
        return 'this is B'
 

class C(A,B):
    pass

c = C()
print(c.play())
```

**输出**

```python
this is A
```

> 首先类是可以多继承的
>
> 优先使用第一个类里面的方法。

**总结**



![1](http://eveseven.oss-cn-shanghai.aliyuncs.com/18-12-13/99683112.jpg)



> 通过C类实例的方法调用来看
>
> 当继承多个父类时，如果父类中有相同的方法，那么子类会优先使用最先被继承的方法.

#### 重构

在上面的例子中，如果不想继承父类的方法怎么办呢？

```python
class Base:
    def play(self):
        return 'this is Base'


class A(Base):
    def play(self):
        return 'this is A'


class B(Base):
    def play(self):
        return 'this is B'


class C(A, B):
    def play(self):
        return 'this is C'


c = C()
print(c.play())
```

**输出**

```python
this is C
```

> 当子类继承父类之后，如果子类不想使用父类的方法，可以通过重写来覆盖父类的方法.

#### 扩展

重写父类方法之后，如果又需要使用父类的方法呢？

##### 方法一

```python
class Base:
    def play(self):
        print('this is Base')


class A(Base):
    def play(self):
        print('this is A')


class B(Base):
    def play(self):
        print('this is B')


class C(A, B):

    def play(self):
        A.play(self)
        print('这是C')


demo = C()
demo.play()
```

**输出**

```python
this is A
这是C
```



##### 方法二

```python
class Base:
    def play(self):
        print('this is Base')


class A(Base):
    def play(self):
        print('this is A')


class B(Base):
    def play(self):
        print('this is B')


class C(A, B):

    def play(self):
        super().play()
        print('这是C')


demo = C()
demo.play()
```

**输出**

```python
this is A
这是C
```

#### super

>  super函数可以调用父类的方法

```python
class Base:
    def play(self):
        print('this is Base')


class A(Base):
    def play(self):
        super().play()
        print('this is A')


class B(Base):
    def play(self):
        super().play()
        print('this is B')


class C(A, B):

    def play(self):
        super().play()
        print('这是C')


demo = C()
demo.play()
```

**输出**

```python
this is Base
this is B
this is A
这是C
```

那为什么是这个顺序输出呢？

```python
print(C.mro())
```

**输出**

```python
[<class '__main__.C'>, <class '__main__.A'>, <class '__main__.B'>, <class '__main__.Base'>, <class 'object'>]
```

#### 继承顺序：

![2](http://eveseven.oss-cn-shanghai.aliyuncs.com/18-12-13/86231356.jpg)



> 在python3中，类被创建时会自动创建方法解析顺序mro
>
> object是所有类的父类

#### Mixin开发模式

![3](http://eveseven.oss-cn-shanghai.aliyuncs.com/18-12-13/85222178.jpg)

> Mixin是一种开发模式，一般规范上，Mixin类是继承的终点，即不再被继承
>
> Mixin的优点就是不需要过多考虑继承关系，不会出现各父类之间有相同方法的情况

#### 总结

> mro: 类在生成时会自动生成方法解析顺序，可以通过  类名.mro()来查看
>
> super: super函数可以来调用父类的方法，使用super的好处在于即使父类改变了，那么也不需要更改类中的代码
>
> Mixin: Mixin是一种开发模式，给大家在今后的开发中提供一种思路.

### 魔法方法

在讲字符串拼接的时候，字符串可以直接相加，那我们自定义的类可以实现吗？

```python
class Rectangle:

    # 传入长和宽
    def __init__(self, length, width):
        self.length = length
        self.width = width

    def area(self):
        area = self.width * self.length
        return area

    def __add__(self, other):
        add_length = self.length + other.length
        add_width = self.width + other.width
        return add_length,add_width

a = Rectangle(3,4)
b = Rectangle(5,6)
print(a+b)
```

**输出**

> (8, 10)

#### 运算方法

![4](http://eveseven.oss-cn-shanghai.aliyuncs.com/18-12-13/78302936.jpg)

> 运算方法大家了解下就行，在实际运用中用不并不多。

##### add和radd原理

```python
class A:
    pass


class B:


    def __add__(self, other):
        print('__add__')

    def __radd__(self, other):
        print('__radd__')

a = A()
b = B()
a+b
```

**输出**

```python
__radd__
```

> 优先在两类里找add方法，没有就自动调用radd方法。

**演示**

```python
class A:
    def __init__(self,name,age):
        self.name =  name
        self.age = age


class B:

    def __init__(self,age):
        self.age = age

    def __add__(self, other):
        print('__add__')

    def __radd__(self, other):
        return other.age + self.age

a = A('age',123)
b = B(123)
print(a+b)
```

**输出**

```python
246
```



#### str和repr原理

```python
class Rectangle:

    # 传入长和宽
    def __init__(self, length, width):
        self.length = length
        self.width = width

    def area(self):
        area = self.width * self.length
        return area

    def __add__(self, other):
        add_length = self.length + other.length
        add_width = self.width + other.width
        return add_length, add_width

    def __radd__(self, other):
        return "Rectangle radd"

    def __str__(self):
        return 'length is %s, width is %s ' % (self.length, self.width)

    def __repr__(self):
        return 'area  is %s' % self.area()


a = Rectangle(3,4)
b = Rectangle(5,6)
print(a+b)
print(a.__add__(b))

print(a)
```

**输出**：

```python
# 有str
(8, 10)
(8, 10)
length is 3, width is 4 
# 无str
(8, 10)
(8, 10)
area  is 12
```

> 优先在两类里找str方法，没有就自动调用repr方法。
>
> 在python中，str和repr方法在处理对象的时候，分别调用的是对象的__str__和__repr__方法
>
> print也是如此，调用str函数来处理输出的对象，如果对象没有定义__str__方法，则调用repr处理

#### call方法

```python
class Rectangle:

    # 传入长和宽
    def __init__(self, length, width):
        self.length = length
        self.width = width

    def area(self):
        area = self.width * self.length
        return area

    def __add__(self, other):
        add_length = self.length + other.length
        add_width = self.width + other.width
        return add_length, add_width

    def __radd__(self, other):
        return "Rectangle radd"

    def __str__(self):
        return 'length is %s, width is %s ' % (self.length, self.width)

    def __repr__(self):
        return 'area  is %s' % self.area()

    def __call__(self):
        return 'Rectangle called'


a = Rectangle(3,4)
b = Rectangle(5,6)
print(a())
```

**输出**

```python
Rectangle called
```

> 正常情况下，实例是不能像函数一样被调用的，要想实例能够被调用，就需要定义 __call__  方法

#### 其他魔法方法

![5](http://eveseven.oss-cn-shanghai.aliyuncs.com/18-12-13/42006401.jpg)

**演示**：

```python
class Rectangle:

    # 传入长和宽
    def __init__(self, length, width):
        self.length = length
        self.width = width

    def area(self):
        area = self.width * self.length
        return area

    def __add__(self, other):
        add_length = self.length + other.length
        add_width = self.width + other.width
        return add_length, add_width

    def __radd__(self, other):
        return "Rectangle radd"

    def __str__(self):
        return 'length is %s, width is %s ' % (self.length, self.width)

    def __repr__(self):
        return 'area  is %s' % self.area()

    def __call__(self):
        return 'Rectangle called'


a = Rectangle(3,4)

print(a.__class__)
print(a.__class__.__base__)
print(a.__class__.__bases__)
print(a.__dict__)   # 所有属性，键值对返回
print(a.__doc__)
print(a.__dir__())
```

**输出**

```python
<class '__main__.Rectangle'>
<class 'object'>
(<class 'object'>,)
{'length': 3, 'width': 4}
None
['length', 'width', '__module__', '__init__', 'area', '__add__', '__radd__', '__str__', '__repr__', '__call__', '__dict__', '__weakref__', '__doc__', '__hash__', '__getattribute__', '__setattr__', '__delattr__', '__lt__', '__le__', '__eq__', '__ne__', '__gt__', '__ge__', '__new__', '__reduce_ex__', '__reduce__', '__subclasshook__', '__init_subclass__', '__format__', '__sizeof__', '__dir__', '__class__']
```

> 简单了解。

#### **魔法方法应用场景**

> __str__和__repr__: str和repr都是分别调用这两个魔术方法来实现的
>
> 原理：在类中，很多事情其实调用的魔术方法来实现的
>
> 作用：通过合理的利用魔术方法，可以让我们更加方便的展示我们的数据

转载请注明：[Seven的博客](http://sevenold.github.io)