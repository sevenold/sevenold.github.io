---
layout: post
title: "python第十二话之异常"
date: 2018-12-19
description: "python异常"
tag: python
---


### 异常

> 异常是什么？
>
> 程序运行过程中出现异常，程序还能正常运行吗？
>
> 如果出现异常该如何让程序正常运行下去呢？

异常即是一个事件，该事件会在程序执行过程中发生，影响了程序的正常执行。

一般情况下，在Python无法正常处理程序时就会发生一个异常。

异常是Python对象，表示一个错误。

当Python脚本发生异常时我们需要捕获处理它，否则程序会终止执行。

#### 语法规则

![异常处理](http://eveseven.oss-cn-shanghai.aliyuncs.com/18-12-19/20180386.jpg)

#### try-except

> 异常是我们敲代码的过程中遇到最多的，那么我们有什么办法来捕获异常呢？

```python
a = 1
b
print('ok')  #ok能够打印出来吗？
```

```python
NameError: name 'b' is not defined
```

> 捕获异常 ，让代码正常执行

```python
try:
    a = 1
    b
except NameError as e:
    print(e)
print('ok')  #ok能够打印出来吗？
```

```python
name 'b' is not defined
ok
```

> 通过捕获异常，代码不仅把错误打印了，后面的代码也正常执行了。

> 那么其他异常怎么办呢？是一样的吗？

```python
try:
    a = 1
    file = open('test.txt', 'r')
except NameError as e:
    print(e)
print('ok')  #ok能够打印出来吗？
```

```python
FileNotFoundError: [Errno 2] No such file or directory: 'test.txt'
```

> 不同类型的异常，就要用不同的状态去捕获

```python
try:
    a = 1
    file = open('test.txt', 'r')
except NameError as e:
    print(e)

except FileNotFoundError as e:
    print(e)
print('ok')  #ok能够打印出来吗？
```

```python
[Errno 2] No such file or directory: 'test.txt'
ok
```

> 异常那么多，我们需要每一个都写吗？

```python
try:
    a = 1
    c = 1 + 'a'
    file = open('test.txt', 'r')
except NameError as e:
    print(e)

except FileNotFoundError as e:
    print(e)

except Exception as e:
    print(e)

print('ok')  #ok能够打印出来吗？
```

```python
unsupported operand type(s) for +: 'int' and 'str'
ok
```

> 同时出现两个异常，会都捕获吗？

```python
try:
    a = 1
    c
    file = open('test.txt', 'r')
except NameError as e:
    print(e)

except FileNotFoundError as e:
    print(e)
print('ok')  #ok能够打印出来吗？
```

```python
name 'c' is not defined
ok
```

> 出现异常后，报异常后的代码就不会执行了，就会跳到except去执行。



#### try-except-else

```python
try:
    a = 1
    c = 1
    file = open('tests.txt', 'r')

except NameError as e:
    print(e)

except FileNotFoundError as e:
    print(e)

except Exception as e:
    print(e)

else:
    file.close()

print('ok')  #ok能够打印出来吗？
```

```python
[Errno 2] No such file or directory: 'tests.txt'
ok
file
Traceback (most recent call last):
  File "C:\Program Files\Python36\lib\site-packages\IPython\core\interactiveshell.py", line 3265, in run_code
    exec(code_obj, self.user_global_ns, self.user_ns)
  File "<ipython-input-3-046c168df224>", line 1, in <module>
    file
NameError: name 'file' is not defined
```

> 这是有抛出异常的情况。



```python
try:
    a = 1
    c = 1
    file = open('test.txt', 'x')

except NameError as e:
    print(e)

except FileNotFoundError as e:
    print(e)

except Exception as e:
    print(e)

else:
    file.close()
    
print('ok')  #ok能够打印出来吗？
```

```python
ok
file
Out[3]: <_io.TextIOWrapper name='test.txt' mode='w' encoding='cp936'>
file.closed
Out[4]: True
```

> 这是没有异常的情况

> else是在不抛出异常的情况下执行。

#### try-except-finally

```python
try:
    a = 1
    c = 1
    file = open('tests.txt', 'r')

except NameError as e:
    print(e)

except FileNotFoundError as e:
    print(e)

except Exception as e:
    print(e)

else:
    file.close()

finally:
    print('end')
print('ok')  #ok能够打印出来吗？
```

```python
[Errno 2] No such file or directory: 'tests.txt'
end
ok
```

> 这是抛出异常的情况

```python
try:
    a = 1
    c = 1
    file = open('test.txt', 'r')

except NameError as e:
    print(e)

except FileNotFoundError as e:
    print(e)

except Exception as e:
    print(e)

else:
    file.close()

finally:
    print('end')
print('ok')  #ok能够打印出来吗？
```

```python
end
ok
```

> 这是不抛出异常的情况

> 不管会不会抛出异常，finally都会在最后执行。

#### 返回错误

```python
def func():
    try:
        c = 1
        file = open('tx.txt', 'r')
        print('aaaaaa')
    except FileNotFoundError as e:
        return e

    except NameError as e:
        print(e)

x = func()
print(x)
```

```python
[Errno 2] No such file or directory: 'tx.txt'
```

#### 直接抛出异常

```python
def func():
    try:
        c = 1
        file = open('tx.txt', 'r')
        print('aaaaaa')
    except FileNotFoundError as e:
        raise e

    except NameError as e:
        print(e)

x = func()
print(x)
```

```python
FileNotFoundError: [Errno 2] No such file or directory: 'tx.txt'
```

> raise是直接抛出异常--和不使用try是一样的。

#### 自定义类

```python
class MyError(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)


try:
    raise MyError(2 * 2)
except MyError as e:
    print('My exception occurred, value:', e.value)
    print(e)
```

```python
My exception occurred, value: 4
4
```

#### 自定义异常错误

```python
class MyError(ValueError):
    ERROR = ("-1", "没有该用户！")


# 抛出异常测试函数
def raiseTest():
    # 抛出异常
    raise MyError(MyError.ERROR[0],  # 异常错误参数1
                  MyError.ERROR[1])  # 异常错误参数2


# 主函数
if __name__ == '__main__':
    try:
        raiseTest()
    except MyError as msg:
        print("errCode:", msg.args[0])  # 获取异常错误参数1
        print("errMsg:", msg.args[1])  # 获取异常错误参数2

```

```python
errCode: -1
errMsg: 没有该用户！
```



#### 注意

> 注意事项：
>
> 1. try 后面必须跟上 except
>
> 2. except 只有在函数中才能使用 return
>
> 3. finally 不管是否发生异常，始终都会执行

#### 总结

> try: 将可能会发生异常的代码放在try中，就可以得到异常，并做相应处理
>
> except: except用来接受异常，并且可以抛出或者返回异常
>
> else和finally:  else在没有异常的时候会执行；finally不管是否有异常，都会执行



### 异常处理

> python中有哪些异常？
>
> 怎样查看所有的异常？
>
> 如何通过程序的报错来找到有问题的代码

#### [异常层次结构](https://docs.python.org/3/library/exceptions.html)

> 在 Python 中所有的异常都是继承 BaseException
>
> 代码中会出现的异常都是 Exception 的子类， 因此在 except 中只需要在最后加上 Exception 即可
>
> 在抛出异常的过程中，会从上倒下依次对比异常，找到之后就不会再往后查找

![异常](http://eveseven.oss-cn-shanghai.aliyuncs.com/18-12-19/719558.jpg)



#### 错误回溯

![错误](http://eveseven.oss-cn-shanghai.aliyuncs.com/18-12-19/1559957.jpg)



#### 总结

> 在今后的学习和工作过程中，会遇到大量的报错，每个开发人员都必须掌握查找和解决报错的能力
>
> 在自己无法解决需要寻求帮助时，也要掌握如何描述问题，把问题描述清楚的能力，这会大大节省双方的时间和精力

### 断言

在调试代码过程中，对于不知道的值可以使用print输出查看一下，但是有些时候，我们清楚某个值应该是怎样的，比如应该是int类型的数据，这个时候需要在类型不对的情况下终止代码，再来调试代码，该怎么做呢？

```python
assert  1==1 
assert  1==2  #报错


assert len(in_s) == 4, 'input size rank 4 required!'
assert len(f_s) == 4, 'filter size rank 4 required!'
assert f_s[2] == in_s[3], 'intput channels not match filter channels.'
assert f_s[0] >= stride and f_s[1] >= stride, 'filter should not be less than stride!'
assert padding in ['SAME', 'VALID'], 'padding value[{0}] not allowded!!'.format(padding)
```

> 断言语句是将调试断言插入程序的一种便捷方式
>
> assert 的语法规则是：
>
> 表达式返回 True  不报错
>
> 表达式返回 False  报错  报 AssertionError

转载请注明：[Seven的博客](http://sevenold.github.io)