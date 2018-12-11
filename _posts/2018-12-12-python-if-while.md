---
layout: post
title: "python第五话之控制流程"
date: 2018-12-12
description: "python控制流程，循环，条件"
tag: python
---

[TOC]



### 控制流程

逻辑值包含了两个值： 
`True`：表示非空的量(比如：string,tuple.list.set,dictonary等) ，所有非零数 。
`False`：表示0,None,空的量等 
`作用`：主要用于判断语句中，用来判断

- 一个字符串是否为空
- 一个运算结果是否为零
- 一个表达式是否可用

### 条件判断

条件语句是根据条件来设置程序接下来的走向。

条件语句的关键字有`if，elif，else`。

#### 基本形式：

> if 判断条件:
>
> 执行语句
>
> else:
>
> 执行语句

判断条件后面和else这个关键字后面都必须加冒号，冒号后面缩进的语句是子语句，多个子语句组成了语句块，如果是单个语句可以与条件写在同一行直接跟在冒号的后面，如果是语句块则一行一条语句，每一行都必须缩进。注意冒号和缩进都是语法的一部分，缩进一般为四个空格。

#### 单个条件

这个是针对只有一个判断条件时的，条件满足时就执行缩进的子语句，else就是表示其余的情况，只要条件不满足则执行else后面子语句。判断语句一般是返回值为bool类型的表达式，值为True则是条件满足，值为False则是条件不满足。

#### **演示**

```python
>>> a = '天晴'
>>> if a=='天晴':
	print('天气好，出去玩吧！')
else:
	print('天气不好，呆在家吧。。')

天气好，出去玩吧！
```

```python
>>> a = '下雨'
>>> if a=='天晴':
	print('天气好，出去玩吧！')
else:
	print('天气不好，呆在家吧。。')

天气不好，呆在家吧。。
```



#### 多个条件

如果判断需要多个条件需同时判断时，可以使用 or （或），表示两个条件有一个成立时判断条件成功；使用 and （与）时，表示只有两个条件同时成立的情况下，判断条件才成功。

#### 演示：

```python
>>> a = "天晴"
>>> t = "有空"
>>> if a=="天晴" and t=="有空":
...     print("天气真好，咱们出去玩！！")
... else:
...     print("天气不好，呆在家吧！！")
... 
天气真好，咱们出去玩！！
```

对于多条件分支的判断使用elif关键字来用来条件分支的.

#### 基本形式

> if 判断条件1:
>
> 执行语句1
>
> elif 判断条件2:
>
> 执行语句2
>
> elif 判断条件n:
>
> 执行语句n
>
> else:
>
> 执行语句x

写多条件分支时，同一个条件中只能有一个if一个else，对elif的个数没有限制但必须是写在if后面，else放在最后表示以上条件都不满足的情况。满足哪个判断条件就执行这个判断条件对应的执行语句，如果列出的条件都不满足则执行else的子语句，语句的执行顺序是从上到下，遇到满足的条件则直接进入它的子语句块，其他剩余判断条件和子语句将不再进行判断和执行。

#### 演示

```python
print('分数等级测试')
score = input('请输入你的分数')
if 90<=int(score)<=100:
    print('你的等级是A')
elif 75<=int(score)<90:
    print('你的等级是B')
elif 60<=int(score)<75:
    print('你的等级是C')
elif 0<=int(score)<60:
    print('你的等级是D')
else:
    print('输入有误!')
```

```python
运行结果（python shell中显示）：
分数等级测试
请输入你的分数98
你的等级是A
```

这里使用了内置的函数input()获取键盘的输入，这里会把键盘的输入以字符串的形式赋值给score这个名字，同类型的才可以进行比较，所以后面在进行条件判断时要把score转换成int类型再进行比较。

### 三目运算

#### 演示：

```python
>>> a = 3 
>>> if a>5:
...     print(True)
... else:
...     print(False)
... 
False
```

更简单的写法呢？

```python
>>> print(True) if a>5 else print(False) 
False
```

#### 

### 条件循环

#### `while`

> 语法规则：
>
>  while  判断语句：
>
> ​    循环体
>
> 注意：注意缩进

#### 演示：

```python
>>> li = [1, 3, 5, 7, 9]
>>> i = 0
>>> while i < len(li):
...     print(True) if li[i]>5 else print(False) 
...     i += 1
... 
False
False
False
True
True
```



对于刚才值大于5的三目运算，如果是判断一个列表中数字该怎么做呢？

#### break

> 跳出循环

**演示**

```python
li = [1, 3, 5, 7, 9]
i = 0
while i < len(li):
	if li[i] > 5:
	    break
	print(li[i])
	i += 1
```

**输出**

```python
1
3
5
```

#### continue

> 跳过此次循环

**演示**

```python
li = [1, 3, 5, 7, 9]
i = 0
while i < len(li):
    print(li[i])
	if li[i] == 5:
	    continue
	i += 1
```

大家猜测下执行结果。。。。。

解决上面的问题

```python
li = [1, 3, 5, 7, 9]
i = -1
while i < len(li)-1:
    print(li[i])
    i += 1
	if li[i] == 5:
	    continue
```

**输出**

```python
1
3
7
9
```

#### else

> 当while的条件不满足时，运行。
>
> 注意：break时，不运行

```python
li = [1, 3, 5, 7, 9]
i = -1
while i < len(li)-1:
	i += 1
	if li[i] == 5:
	    continue
	print(li[i])
else:
    print('ok')
```

**输出**

```python
1
3
7
9
ok
```

**演示**

```python
li = [1, 3, 5, 7, 9]
i = -1
while i < len(li)-1:
	i += 1
	if li[i] == 5:
        # continue
		break
	print(li[i])
else:
    print('ok')
```

**输出**

```python
1
3
```

#### **总结**

> 循环可以被终止：
>
> - 判断语句可以返回  False
> - 通过break终止循环 
>
> else的执行条件：
>
> 只有在循环不是被break终止的情况下才会执行else中的内容

### 迭代循环

#### **for**

> 只要是可迭代对象，都可以使用for循环遍历。

#### 语法规则

> for  i  in  obj：
>
> ​     循环体
>
> 注意：注意缩进

**演示**

```python
li = [1, 3, 5, 7, 9]
for i in li:
    print(i)
```

**输出**

```python
1
3
5
7
9
```



#### **range**

**演示**

```python
for i in range(1, 21):
	print(i)
```

**输出**

```python
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
18
19
20
```

> 内置函数，表示一个范围，不包含结尾值。

```python
In [3]: list(range(21))
Out[3]: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    
In [4]: list(range(2, 21))
Out[4]: [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]

In [5]: list(range(1, 21, 2))
Out[5]: [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]

```



#### **continue**

> 跳出当前循环

**演示**

```python
for i in range(1, 21):
	if i%5 == 0:
	    continue
	print(i)
```

**输出**

```python
1
2
3
4
6
7
8
9
11
12
13
14
16
17
18
19
```

#### break

> 跳出循环

**演示**

```python
for i in range(1, 21):
	if i%5 == 0:
	    break
	print(i)
```

**输出**

```python
1
2
3
4
```

#### **else**

> 当for循环结束时，运行。
>
> 注意：break时，不运行

**演示**：

```python
for i in range(1, 21):
	if i%5 == 0:
	    continue
	print(i)
else:
    print('end...')
```

**输出**

```python
1
2
3
4
6
7
8
9
11
12
13
14
16
17
18
19
end...
```



#### **总结**

> for 后面需要接上可迭代对象
>
> for会依次取出可迭代对象中的元素
>
>
>
> continue的用法：
>
> continue和break类似，但是continue不会终止循环，而是结束本次循环，跳到下次循环

转载请注明：[Seven的博客](http://sevenold.github.io)