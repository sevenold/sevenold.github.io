---
layout: post
title: "datetime-Python时间处理模块"
date: 2018-11-21
description: "datetime-Python时间处理模块"
tag: python
---

转载自：简书-[王南北](https://www.jianshu.com/u/948da055a416)

## 简介

在Python的时间处理模块中，`time`这个模块主要侧重于时间戳格式的处理，而`datetime`则相当于`time`模块的高级封装，提供了更多关于日期处理的方法。

并且`datetime`的接口使用起来更加的直观，方便。

------

## 结构

`datetime`主要由五个模块组成：

- **datetime.date**：表示日期的类。常用的属性有year, month, day。
- **datetime.time**：表示时间的类。常用的属性有hour, minute, second, microsecond。
- **datetime.datetime**：表示日期+时间。
- **datetime.timedelta**：表示时间间隔，即两个时间点之间的长度，常常用来做时间的加减。
- **datetime.tzinfo**：与时区有关的相关信息。

在`datetime`中，使用的最多的就是`datetime.datetime`模块，而`datetime.timedelta`常常被用来修改时间。

最后，`datetime`的时间显示是与时区有关系的，所以还有一个处理时区信息的模块`datetime.tzinfo`。

------

## datetime.datetime

```python
class datetime.datetime(year, month, day, hour=0, minute=0, second=0, microsecond=0, tzinfo=None, *, fold=0)
```

- MINYEAR <= year <= MAXYEAR
- 1 <= month <= 12
- 1 <= day <= number of days in the given month and year
- 0 <= hour < 24
- 0 <= minute < 60
- 0 <= second < 60
- 0 <= microsecond < 1000000
- fold in [0, 1]

这是`datetime.datetime`参数的取值范围，如果设定的值超过这个范围，那么就会抛出`ValueError`异常。

其中`year`，`month`，`day`是必须参数。

```python
In [1]: import datetime

In [2]: datetime.datetime(year=2000, month=1, day=1, hour=12)
Out[2]: datetime.datetime(2000, 1, 1, 12, 0)
```

### 类方法 Classmethod

这些方法大多数用来生成一个`datetime`对象。

- classmethod datetime.**today()**

  获取今天的时间。

  ```python
  In [1]: datetime.datetime.today()
  Out[1]: datetime.datetime(2018, 7, 2, 15, 5, 17, 127663)
  ```

- classmethod datetime.**now(tz=None)**

  获取当前的时间，可以传入一个`tzinfo`对象来指定某一个时区。

  ```python
  In [2]: datetime.datetime.now()
  Out[2]: datetime.datetime(2018, 7, 2, 15, 8, 30, 593801)
  ```

- classmethod datetime.**fromtimestamp(timestamp, tz=None)**

  用一个时间戳来生成`datetime`对象，同样可以指定时区。

  **注：时间戳需要为10位的**

  ```python
  In [3]: datetime.datetime.fromtimestamp(1530515475.18224)
  Out[3]: datetime.datetime(2018, 7, 2, 15, 11, 15, 182240)
  ```

### 实例方法 instance method

这些方法大多是一个`datetime`对象能进行的操作。

- datetime.**date()** 和 datetime.**time()**

  获取`datetime`对象的日期或者时间部分。

  ```python
  In [1]: datetime.datetime.now().date()
  Out[1]: datetime.date(2018, 7, 2)
  
  In [1]: datetime.datetime.now().time()
  Out[1]: datetime.time(15, 24, 37, 355514)
  ```

- datetime.**replace(year=self.year, month=self.month, day=self.day, hour=self.hour, minute=self.minute, second=self.second, microsecond=self.microsecond, tzinfo=self.tzinfo, * fold=0)**

  替换`datetime`对象的指定数据。

  ```python
  In [1]: now = datetime.datetime.now()
  
  In [2]: now
  Out[2]: datetime.datetime(2018, 7, 2, 15, 26, 45, 116239)
  
  In [3]: now.replace(year=2000)
  Out[3]: datetime.datetime(2000, 7, 2, 15, 26, 45, 116239)
  ```

- datetime.`timestamp()`

  转换成时间戳。

  ```python
  In [1]: datetime.datetime.now().timestamp()
  Out[1]: 1530515994.798248
  ```

- datetime.**weekday()**

  返回一个值，表示日期为星期几。0为星期一，6为星期天。

  ```python
  In [1]: datetime.now().weekday()
  Out[1]: 1
  ```

------

## datetime.timedelta

在实际的使用中，我们常常会遇到这样的需求：需要给某个时间增加或减少一天，甚至是增加或减少一天三小时二十分钟。

那么在遇到这样的需求时，去计算时间戳是非常的麻烦的，所以`datetime.timedelta`这个模块使我们能够非常方便的对时间做加减。

```python
class datetime.timedelta(days=0, seconds=0, microseconds=0, milliseconds=0, minutes=0, hours=0, weeks=0)
```

`datetime`是对某些运算符进行了重载的，所以我们可以如下操作。

```python
In [1]: from datetime import timedelta

In [2]: now
Out[2]: datetime.datetime(2018, 7, 2, 15, 26, 45, 116239)

In [3]: now - timedelta(days=1)
Out[3]: datetime.datetime(2018, 7, 1, 15, 26, 45, 116239)

In [4]: now + timedelta(days=1)
Out[4]: datetime.datetime(2018, 7, 3, 15, 26, 45, 116239)

In [5]: now + timedelta(days=-1)
Out[5]: datetime.datetime(2018, 7, 1, 15, 26, 45, 116239)
```

------

## strftime() 和 strptime()

`datetime`中提供了两个方法，可以方便的把`datetime`对象转换成格式化的字符串或者把字符串转换成`datetime`对象。

- 由`datetime`转换成字符串：**datetime.strftime()**

  `strftime()`是datetime对象的实例方法：

  ```python
  In [1]: datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S %f')
  Out[1]: '2018-07-02 15:26:45 116239'
  ```

- 由字符串转换成`datetime`：**datetime.datetime.strptime()**

  `strptime()`则是一个类方法：

  ```python
  In [1]: newsTime='Sun, 23 Apr 2017 05:15:05 GMT'
  
  In [2]: GMT_FORMAT = '%a, %d %b %Y %H:%M:%S GMT'
  
  In [3]: datetime.datetime.strptime(newsTime,GMT_FORMAT)
  Out[3]: datetime.datetime(2017, 4, 23, 5, 15, 5)
  ```

以下是格式化的符号：

- %y 两位数的年份表示（00-99）
- %Y 四位数的年份表示（000-9999）
- %m 月份（01-12）
- %d 月内中的一天（0-31）
- %H 24小时制小时数（0-23）
- %I 12小时制小时数（01-12）
- %M 分钟数（00=59）
- %S 秒（00-59）
- %a 本地简化星期名称
- %A 本地完整星期名称
- %b 本地简化的月份名称
- %B 本地完整的月份名称
- %c 本地相应的日期表示和时间表示
- %j 年内的一天（001-366）
- %p 本地A.M.或P.M.的等价符
- %U 一年中的星期数（00-53）星期天为星期的开始
- %w 星期（0-6），星期天为星期的开始
- %W 一年中的星期数（00-53）星期一为星期的开始
- %x 本地相应的日期表示
- %X 本地相应的时间表示
- %Z 当前时区的名称
- %% %号本身

转载请注明：[Seven的博客](http://sevenold.github.io)