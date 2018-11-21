---
layout: post
title: "logging-Python多层级日志输出"
date: 2018-11-21
description: "logging-Python多层级日志输出"
tag: python
---

转载自：简书-[王南北](https://www.jianshu.com/u/948da055a416)

本文地址：https://www.jianshu.com/p/3be28b5d2ff8

### 简介

在应用的开发过程中，我们常常需要去记录应用的状态，事件，结果。而Python最基础的Print很难满足我们的需求，这种情况下我们就需要使用python的另一个标准库：`logging`。

这是一个专门用于记录日志的模块。相对于Print来说，`logging`提供了日志信息的分级，格式化，过滤等功能。如果在程序中定义了丰富而有条理的log信息，那么可以非常方便的去分析程序的运行状况，在有问题时也能够方便的去定位问题，分析问题。

以下是具体的一些应用场景。

| 执行的任务                       | 这项任务的最佳工具                                         |
| :------------------------------- | :--------------------------------------------------------- |
| 显示控制台输出                   | print()                                                    |
| 报告在程序正常运行期间发生的事件 | logging.info()或 logging.debug()                           |
| 发出有关特定运行时事件的警告     | logging.warning()                                          |
| 报告有关特定运行时事件的错误     | 抛出异常                                                   |
| 报告错误但不抛出异常             | logging.error()， logging.exception()或 logging.critical() |

------

### 基础用法

以下是一些`logging`最基础的使用方法，如果不需要深入的去定制log的话，那么只需要使用最基础的部分即可。

```python
In [1]: import logging

In [2]: logging.info('hello world')

In [3]: logging.warning('good luck')
WARNING:root:good luck
```

可以看到，`logging.info()`的日志信息没有被输出，而`logging.warning()`的日志信息被输出了，这就是因为`logging`的日志信息分为几个不同的重要性级别，而默认输出的级别则是`warning`，也就是说，重要性大于等于`warning`的信息才会被输出。

以下是`logging`模块中信息的五个级别，重要性从上往下递增。

| 等级       | 什么时候使用                                                 |
| ---------- | ------------------------------------------------------------ |
| `DEBUG`    | 详细信息，通常仅在Debug时使用。                              |
| `INFO`     | 程序正常运行时输出的信息。                                   |
| `WARNING`  | 表示有些预期之外的情况发生，或者在将来可能发生什么情况。程序依然能按照预期运行。 |
| `ERROR`    | 因为一些严重的问题，程序的某些功能无法使用了。               |
| `CRITICAL` | 发生了严重的错误，程序已经无法运行。                         |

我们也可以通过设置来设定输出日志的级别：

```python
In [1]: import logging

In [2]: logging.basicConfig(level=logging.DEBUG)

In [3]: logging.info('hello world')
INFO:root:hello world
```

可以看到，在设定了`level`参数为`logging.DEBUG`后，`logging.info()`的日志信息就正常输出了。

------

### logging.basicConfig(**\*kwargs*)

通过`basicConfig()`方法可以为`logging`做一些简单的配置。此方法可以传递一些关键字参数。

- **filename**

  文件名参数，如果指定了这个参数，那么`logging`会把日志信息输入到指定的文件之中。

  ```python
  import logging
  logging.basicConfig(filename='example.log')
  logging.warning('Hello world')
  ```

- **filemode**

  如果指定了`filename`来输出日志到文件，那么`filemode`就是打开文件的模式，默认为'a'，追加模式。当然也可以设置为'w'，则每一次输入都会丢弃掉之前日志文件中的内容。

- **format**

  指定输出的log信息的格式。

  ```python
  In [1]: import logging
  
  In [2]: logging.basicConfig(format='%(asctime)s %(message)s')
  
  In [3]: logging.warning('hello world')
  2018-07-06 16:28:12,074 hello world
  ```

- **datefmt**

  如果在`format`中使用了`asctime`输出时间，那么可以使用此参数控制输出日期的格式，使用方式与`time.strftime()`相同。

- **level**

  设置输出的日志的级别，只有高出此级别的日志信息才会被输出。

  ```python
  In [1]: import logging
  
  In [2]: logging.basicConfig(level=logging.INFO)
  
  In [3]: logging.info('hi')
  INFO:root:hi
  
  In [4]: logging.debug('byebye')
  ```

**注：需要注意的是，`basicConfig()`方法是一个一次性的方法，只能用来做简单的配置，多次的调用`basicConfig()`是无效的。**

------

### 深度定制logging日志信息

在深度使用`logging`来定制日志信息之前，我们需要先来了解一下`logging`的结构。`logging`的主要逻辑结构主要由以下几个组件构成：

- **Logger**：提供应用程序直接使用的接口。
- **Handler**：将log信息发送到目标位置。
- **Filter**：提供更加细粒度的log信息过滤。
- **Formatter**：格式化log信息。

这四个组件是`logging`模块的基础，在基础用法中的使用方式，其实也是这四大组件的封装结果。

这四个组件的关系如下所示：

![logging-结构](/images/py/68.png)

`logger`主要为外部提供使用的api接口，而每个`logger`下可以设置多个`Handler`，来将log信息输出到多个位置，而每一个`Handler`下又可以设置一个`Formatter`和多个`Filter`来定制输出的信息。

------

### Logger

`Logger`这个对象主要有三个任务要做：

- 向外部提供使用接口。
- 基于日志严重等级（默认的过滤组件）或filter对象来决定要对哪些日志进行后续处理。
- 将日志消息传送给所有符合输出级别的`Handlers`。

#### logging.getLogger(name=None)

首先，我们需要通过`getLogger()`方法来生成一个`Logger`，这个方法中有一个参数name，则是生成的`Logger`的名称，如果不传或者传入一个空值的话，`Logger`的名称默认为root。

```python
In [1]: import logging

In [2]: logger = logging.getLogger('nanbei')
```

需要注意的是，只要在同一个解释器的进程中，那么相同的`Logger`名称，使用`getLogger()`方法将会指向同一个`Logger`对象。

而使用`logger`的一个好习惯，是生成一个模块级别的`Logger`对象：

```python
In [1]: logger = logging.getLogger(__name__)
```

通过这种方式，我们可以让`logger`清楚的记录下事件发生的模块位置。

除此之外，`logger`对象是有层级结构的：

- `Logger`的名称可以是一个以`.`分割的层级结构，每个`.`后面的`Logger`都是`.`前面的`logger`的子辈。

  例如，有一个名称为**nanbei**的`logger`，其它名称分别为**nanbei.a**，**nanbei.b**和**nanbei.a.c**都是**nanbei**的后代。

- 子`Logger`在完成对日志消息的处理后，默认会将log日志消息传递给它们的父辈`Logger`相关的`Handler`。

  因此，我们不不需要去配置每一个的`Logger`，只需要将程序中一个顶层的`Logger`配置好，然后按照需要创建子`Logger`就好了。也可以通过将一个`logger`的`propagate`属性设置为False来关闭这种传递机制。

例如：

```python
In [1]: import logging
# 生成一个名称为nanbei的Logger
In [2]: logger = logging.getLogger('nanbei')
# 生成一个StreamHandler，这个Handler可以将日志输出到console中
In [3]: sh = logging.StreamHandler()
# 生成一个Formatter对象，使输出日志时只显示Logger名称和日志信息
In [4]: fmt = logging.Formatter(fmt='%(name)s - %(message)s')
# 设置Formatter到StreamHandler中
In [5]: sh.setFormatter(fmt)
# 将Handler添加到Logger中
In [6]: logger.addHandler(sh)
# 生成一个nanbei的子Logger：nanbei.child
In [7]: child_logger = logging.getLogger('nanbei.child')
# 可以看到两个Logger输出的日志信息都使用了相同的日志格式
In [8]: logger.warning('hello')
nanbei - hello

In [9]: child_logger.warning('hello')
nanbei.child - hello
```

### 方法

在`Logger`对象中，主要提供了以下方法：

| 方法                                                         | 描述                                          |
| ------------------------------------------------------------ | --------------------------------------------- |
| Logger.setLevel()                                            | 设置日志器将会处理的日志消息的最低输出级别    |
| Logger.addHandler() 和 Logger.removeHandler()                | 为该logger对象添加、移除一个handler对象       |
| Logger.addFilter() 和 Logger.removeFilter()                  | 为该logger对象添加、移除一个filter对象        |
| Logger.debug()，Logger.info()，Logger.warning()，Logger.error()，Logger.critical() | 输出一条与方法名对应等级的日志                |
| Logger.exception()                                           | 输出一条与Logger.error()类似的日志            |
| Logger.log()                                                 | 可以传入一个明确的日志level参数来输出一条日志 |

------

### Handler

`Handler`的作用主要是把log信息输出到我们希望的目标位置，其提供了如下的方法以供使用：

| 方法                                          | 描述                              |
| --------------------------------------------- | --------------------------------- |
| Handler.setLevel()                            | 设置handler处理日志消息的最低级别 |
| Handler.setFormatter()                        | 为handler设置一个格式器对象       |
| Handler.addFilter() 和 Handler.removeFilter() | 为handler添加、删除一个过滤器对象 |

我们可以通过这几个方法，给每一个`Handler`设置一个`Formatter`和多个`Filter`，来定制不同的输出log信息的策略。

而`Handler`本身是一个基类，不应该直接实例化使用，我们应该使用的是其多种多样的子类，每一个不同的子类可以将日志信息输出到不同的目标位置，以下是一些常用的`Handler`。

| Handler                                   | 描述                                                         |
| ----------------------------------------- | ------------------------------------------------------------ |
| logging.StreamHandler                     | 将日志消息发送到输出到Stream，如std.out, std.err或任何file-like对象。 |
| logging.FileHandler                       | 将日志消息发送到磁盘文件，默认情况下文件大小会无限增长       |
| logging.handlers.RotatingFileHandler      | 将日志消息发送到磁盘文件，并支持日志文件按大小切割           |
| logging.hanlders.TimedRotatingFileHandler | 将日志消息发送到磁盘文件，并支持日志文件按时间切割           |
| logging.handlers.HTTPHandler              | 将日志消息以GET或POST的方式发送给一个HTTP服务器              |
| logging.handlers.SMTPHandler              | 将日志消息发送给一个指定的email地址                          |
| logging.NullHandler                       | 该Handler实例会忽略error messages，通常被想使用logging的library开发者使用来避免'No handlers could be found for logger XXX'信息的出现。 |

------

### Filter

`Filter`可以被`Handler`和`Logger`用来做比level分级更细粒度的、更复杂的过滤功能。

`Filter`是一个过滤器基类，它可以通过name参数，来使这个`logger`下的日志通过过滤。

```python
class logging.Filter(name='')
```

比如，一个`Filter`实例化时传递的name参数值为`A.B`，那么该`Filter`实例将只允许名称为类似如下规则的`Loggers`产生的日志通过过滤：`A.B`，`A.B.C`，`A.B.C.D`，`A.B.D`。

而名称为`A.BB`，`B.A.B`的`Loggers`产生的日志则会被过滤掉。如果name的值为空字符串，则允许所有的日志事件通过过滤。

```python
In [1]: import logging

In [2]: logger = logging.getLogger('nanbei')

In [3]: filt = logging.Filter(name='nanbei.a')

In [4]: sh = logging.StreamHandler()

In [5]: sh.setLevel(logging.DEBUG)

In [6]: sh.addFilter(filt)

In [7]: logger.addHandler(sh)

In [8]: logging.getLogger('nanbei.a.b').warning('i am nanbei.a.b')
i am nanbei.a.b

In [9]: logging.getLogger('nanbei.b.b').warning('i am nanbei.a.b')
```

可以看到，名称为**nanbei.b.b**的`Logger`的日志没有被输出。

------

### Formatter

`Formater`对象用于配置日志信息的最终顺序、结构和内容。

`Formatter`类的构造方法定义如下：

```python
logging.Formatter.__init__(fmt=None, datefmt=None, style='%')
```

- **fmt**

  这个参数主要用于格式化log信息整体的输出。

  以下是可以用来格式化的字段：

  | 字段/属性名称   | 使用格式            |                             描述                             |
  | :-------------- | :------------------ | :----------------------------------------------------------: |
  | asctime         | %(asctime)s         | 日志事件发生的时间--人类可读时间，如：2003-07-08 16:49:45,896 |
  | created         | %(created)f         | 日志事件发生的时间--时间戳，就是当时调用time.time()函数返回的值 |
  | relativeCreated | %(relativeCreated)d | 日志事件发生的时间相对于logging模块加载时间的相对毫秒数（目前还不知道干嘛用的） |
  | msecs           | %(msecs)d           |                  日志事件发生事件的毫秒部分                  |
  | levelname       | %(levelname)s       | 该日志记录的文字形式的日志级别（'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'） |
  | levelno         | %(levelno)s         |     该日志记录的数字形式的日志级别（10, 20, 30, 40, 50）     |
  | name            | %(name)s            | 所使用的日志器名称，默认是'root'，因为默认使用的是 rootLogger |
  | message         | %(message)s         |       日志记录的文本内容，通过 `msg % args`计算得到的        |
  | pathname        | %(pathname)s        |              调用日志记录函数的源码文件的全路径              |
  | filename        | %(filename)s        |              pathname的文件名部分，包含文件后缀              |
  | module          | %(module)s          |                filename的名称部分，不包含后缀                |
  | lineno          | %(lineno)d          |              调用日志记录函数的源代码所在的行号              |
  | funcName        | %(funcName)s        |                   调用日志记录函数的函数名                   |
  | process         | %(process)d         |                            进程ID                            |
  | processName     | %(processName)s     |                   进程名称，Python 3.1新增                   |
  | thread          | %(thread)d          |                            线程ID                            |
  | threadName      | %(thread)s          |                           线程名称                           |

- **datefmt**

  如果在dmt中指定了asctime，那么这个参数可以用来格式化asctime的输出，使用方式与time.strftime()相同。

- **style**

  Python 3.2新增的参数，可取值为 '%', '{'和 '$'，如果不指定该参数则默认使用'%'。



转载请注明：[Seven的博客](http://sevenold.github.io)