---
layout: post
title: "HTML解析库BeautifulSoup4"
date: 2018-11-21
description: "HTML解析库BeautifulSoup4"
tag: 爬虫
---

转载自：简书-[王南北](https://www.jianshu.com/u/948da055a416)

### HTML解析库BeautifulSoup4

`BeautifulSoup` 是一个可以从HTML或XML文件中提取数据的Python库，它的使用方式相对于正则来说更加的简单方便，常常能够节省我们大量的时间。

`BeautifulSoup`也是有官方中文文档的：[https://www.crummy.com/software/BeautifulSoup/bs4/doc/index.zh.html](https://www.crummy.com/software/BeautifulSoup/bs4/doc/index.zh.html)

------

### 安装

`BeautifulSoup`的安装也是非常方便的，pip安装即可。

```python
pip install beautifulsoup4
```

------

### 简单例子

以下是一段HTML代码，作为例子被多次用到，这是 *爱丽丝梦游仙境* 中的一段内容。

```python
html_doc = """
<html><head><title>The Dormouse's story</title></head>
<body>
<p class="title"><b>The Dormouse's story</b></p>

<p class="story">Once upon a time there were three little sisters; and their names were
<a href="http://example.com/elsie" class="sister" id="link1">Elsie</a>,
<a href="http://example.com/lacie" class="sister" id="link2">Lacie</a> and
<a href="http://example.com/tillie" class="sister" id="link3">Tillie</a>;
and they lived at the bottom of a well.</p>

<p class="story">...</p>
"""
```

我们获取的网页数据通常会像上面这样是完全的字符串格式，所以我们首先需要使用`BeautifulSoup`来解析这段字符串。然后会获得一个`BeautifulSoup`对象，通过这个对象我们就可以进行一系列操作了。

```python
In [1]: from bs4 import BeautifulSoup

In [2]: soup = BeautifulSoup(html_doc)

In [3]: soup.title
Out[3]: <title>The Dormouse's story</title>

In [4]: soup.title.name
Out[4]: 'title'

In [5]: soup.title.string
Out[5]: "The Dormouse's story"

In [6]: soup.title.parent.name
Out[6]: 'head'

In [7]: soup.p
Out[7]: <p class="title"><b>The Dormouse's story</b></p>

In [8]: soup.p['class']
Out[8]: ['title']

In [9]: soup.a
Out[9]: <a class="sister" href="http://example.com/elsie" id="link1">Elsie</a>

In [10]: soup.find_all('a')
Out[10]:
[<a class="sister" href="http://example.com/elsie" id="link1">Elsie</a>,
 <a class="sister" href="http://example.com/lacie" id="link2">Lacie</a>,
 <a class="sister" href="http://example.com/tillie" id="link3">Tillie</a>]

In [11]: soup.find(id="link3")
Out[11]: <a class="sister" href="http://example.com/tillie" id="link3">Tillie</a>
```

可以看到，相对于正则来说，操作简单了不止一个量级。

------

### 使用

### 指定解析器

在上面的例子中，我们可以看到在查找数据之前，是有一个解析网页的过程的：

```python
soup = BeautifulSoup(html_doc)
```

`BeautifulSoup`会自动的在系统中选定一个可用的解析器，以下是主要的几种解析器：

| 解析器           | 使用方法                                                     | 优势                                                  | 劣势                                            |
| ---------------- | ------------------------------------------------------------ | ----------------------------------------------------- | ----------------------------------------------- |
| Python标准库     | `BeautifulSoup(markup, "html.parser")`                       | Python的内置标准库执行速度适中文档容错能力强          | Python 2.7.3 or 3.2.2)前 的版本中文档容错能力差 |
| lxml HTML 解析器 | `BeautifulSoup(markup, "lxml")`                              | 速度快文档容错能力强                                  | 需要安装C语言库                                 |
| lxml XML 解析器  | `BeautifulSoup(markup, ["lxml", "xml"])``BeautifulSoup(markup, "xml")` | 速度快唯一支持XML的解析器                             | 需要安装C语言库                                 |
| html5lib         | `BeautifulSoup(markup, "html5lib")`                          | 最好的容错性以浏览器的方式解析文档生成HTML5格式的文档 | 速度慢不依赖外部扩展                            |

由于这个解析的过程在大规模的爬取中是会影响到整个爬虫系统的速度的，所以推荐使用的是`lxml`，速度会快很多，而`lxml`需要单独安装：

```python
pip install lxml
```

安装成功后，在解析网页的时候，指定为lxml即可。

```python
soup = BeautifulSoup(html_doc, 'lxml')
```

注意：**如果一段HTML或XML文档格式不正确的话,那么在不同的解析器中返回的结果可能是不一样的，所以要指定某一个解析器。**

------

### 节点对象

BeautifulSoup将复杂HTML文档转换成一个复杂的树形结构，每个节点都是Python对象，所有对象可以归纳为4种：`Tag`，`NavigableString`，`BeautifulSoup`，`Comment`。

### tag

`tag`就是标签的意思，`tag`还有许多的方法和属性。

```python
>>> soup = BeautifulSoup('<b class="boldest">Extremely bold</b>')
>>> tag = soup.b
>>> type(tag)
<class 'bs4.element.Tag'>
```

- name

  每一个`tag`对象都有`name`属性，为标签的名字。

  ```python
  >>> tag.name
  'b'
  ```

- Attributes

  在HTML中，`tag`可能有多个属性，所以`tag`属性的取值跟字典相同。

  ```python
  >>> tag['class']
  'boldest'
  ```

  如果某个`tag`属性有多个值，那么返回的则是列表格式。

  ```python
  >>> soup = BeautifulSoup('<p class="body strikeout"></p>')
  >>> soup.p['class']
  ["body", "strikeout"]
  ```

- get_text()

  通过`get_text()`方法我们可以获取某个`tag`下所有的文本内容。

  ```python
  In [1]: soup.body.get_text()
  Out[1]: "The Dormouse's story\nOnce upon a time there were three little sisters; and their names were\nElsie,\nLacie and\nTillie;\nand they lived at the bottom of a well.\n...\n"
  ```

### NavigableString

`NavigableString`的意思是可以遍历的字符串，一般被标签包裹在其中的的文本就是`NavigableString`格式。

```python
In [1]: soup = BeautifulSoup('<p>No longer bold</p>')

In [2]: soup.p.string
Out[2]: 'No longer bold'

In [3]: type(soup.p.string)
Out[3]: bs4.element.NavigableString
```

### BeautifulSoup

`BeautifulSoup`对象就是解析网页获得的对象。

### Comment

`Comment`指的是在网页中的注释以及特殊字符串。

------

### Tag与遍历文档树

`tag`对象可以说是`BeautifulSoup`中最为重要的对象，通过`BeautifulSoup`来提取数据基本都围绕着这个对象来进行操作。

首先，一个节点中是可以包含多个子节点和多个字符串的。例如`html`节点中包含着`head`和`body`节点。所以`BeautifulSoup`就可以将一个HTML的网页用这样一层层嵌套的节点来进行表示。

以上方的爱丽丝梦游仙境为例：

### contents和children

通过`contents`可以获取某个节点所有的子节点，包括里面的`NavigableString`对象。获取的子节点是列表格式。

```python
In [1]: soup.head.contents
Out[1]: [<title>The Dormouse's story</title>]
```

而通过`children`同样的是获取某个节点的所有子节点，但是返回的是一个迭代器，这种方式会比列表格式更加的节省内存。

```python
In [1]: tags = soup.head.children

In [2]: tags
Out[2]: <list_iterator at 0x110f76940>

In [3]: for tag in tags:
            print(tag)

<title>The Dormouse's story</title>
```

### descendants

上面的`contents`和`children`获取的是某个节点的直接子节点，而无法获得子孙节点。通过`descendants`可以获得所有子孙节点，返回的结果跟`children`一样，需要迭代或者转类型使用。

```python
In [1]: len(list(soup.body.descendants))
Out[1]: 19

In [2]: len(list(soup.body.children))
Out[2]: 6
```

### string和strings

我们常常会遇到需要获取某个节点中的文本值的情况，如果这个节点中只有一个字符串，那么使用`string`可以正常将其取出。

```python
In [1]: soup.title.string
Out[1]: "The Dormouse's story"
```

而如果这个节点中有多个字符串的时候，`BeautifulSoup`就无法确定要取出哪个字符串了，这时候需要使用`strings`。

```python
In [1]: list(soup.body.strings)
Out[1]:
["The Dormouse's story",
 '\n',
 'Once upon a time there were three little sisters; and their names were\n',
 'Elsie',
 ',\n',
 'Lacie',
 ' and\n',
 'Tillie',
 ';\nand they lived at the bottom of a well.',
 '\n',
 '...',
 '\n']
```

而使用`stripped_strings`可以将全是空白的行去掉。

```python
In [1]: list(soup.body.stripped_strings)
Out[1]:
["The Dormouse's story",
 'Once upon a time there were three little sisters; and their names were',
 'Elsie',
 ',',
 'Lacie',
 'and',
 'Tillie',
 ';\nand they lived at the bottom of a well.',
 '...']
```

### 父节点parent和parents

有时我们也需要去获取某个节点的父节点，也就是包裹着当前节点的节点。

```python
In [1]: soup.b.parent
Out[1]: <p class="title"><b>The Dormouse's story</b></p>
```

而使用`parents`则可以获得当前节点递归到顶层的所有父辈元素。

```python
In [1]: [i.name for i in soup.b.parents]
Out[1]: ['p', 'body', 'html', '[document]']
```

### 兄弟节点

兄弟节点指的就是父节点相同的节点。

- **next_sibling 和 previous_sibling**

  兄弟节点选取的方法与当前节点的位置有关，`next_sibling`获取的是当前节点的下一个兄弟节点，`previous_sibling`获取的是当前节点的上一个兄弟节点。

  所以，兄弟节点中排第一个的节点是没有`previous_sibling`的，最后一个节点是没有`next_sibling`的。

  ```python
  In [51]: soup.head.next_sibling
  Out[51]: '\n'
  
  In [52]: soup.head.previos_sibling
  
  In [59]: soup.body.previous_sibling
  Out[59]: '\n'
  ```

- **next_siblings 和 previous_siblings**

  相对应的，`next_siblings`获取的是下方所有的兄弟节点，`previous_siblings`获取的上方所有的兄弟节点。

  ```python
  In [47]: [i.name for i in soup.head.next_siblings]
  Out[47]: [None, 'body']
  
  In [48]: [i.name for i in soup.body.next_siblings]
  Out[48]: []
  
  In [49]: [i.name for i in soup.body.previous_siblings]
  Out[49]: [None, 'head']
  ```

------

### find_all()

上方这种直接通过属性来进行访问属性的方法，很多时候只能适用于比较简单的一些场景，所以`BeautifulSoup`还提供了搜索整个文档树的方法`find_all()`。

需要注意的是，`find_all()`方法基本所有节点对象都能调用。

### 通过name搜索

就像以下演示的，`find_all()`可以直接查找出整个文档树中所有的b标签，并返回列表。

```python
>>> soup.find_all('b')
[<b>The Dormouse's story</b>]
```

而如果传入的是一个列表，则会与列表中任意一个元素进行匹配。可以看到，搜索的结果包含了所有的a标签和b标签。

```python
>>> soup.find_all(["a", "b"])
[<b>The Dormouse's story</b>,
 <a class="sister" href="http://example.com/elsie" id="link1">Elsie</a>,
 <a class="sister" href="http://example.com/lacie" id="link2">Lacie</a>,
 <a class="sister" href="http://example.com/tillie" id="link3">Tillie</a>]
```

### 通过属性搜索

我们在搜索的时候一般只有标签名是不够的，因为可能同名的标签很多，那么这时候我们就要通过标签的属性来进行搜索。

这时候我们可以通过传递给attrs一个字典参数来搜索属性。

```python
In [1]: soup.find_all(attrs={'class': 'sister'})
Out[1]:
[<a class="sister" href="http://example.com/elsie" id="link1">Elsie</a>,
 <a class="sister" href="http://example.com/lacie" id="link2">Lacie</a>,
 <a class="sister" href="http://example.com/tillie" id="link3">Tillie</a>]
```

可以看到找出了所有class属性为sister的标签。

### 通过文本搜索

在`find_all()`方法中，还可以根据文本内容来进行搜索。

```python
>>> soup.find_all(text="Elsie")
[u'Elsie']

>>> soup.find_all(text=["Tillie", "Elsie", "Lacie"])
[u'Elsie', u'Lacie', u'Tillie']
```

可见找到的都是字符串对象，如果想要找到包含某个文本的`tag`，加上`tag`名即可。

```python
>>> soup.find_all("a", text="Elsie")
[<a href="http://example.com/elsie" class="sister" id="link1">Elsie</a>]
```

### 限制查找范围为子节点

`find_all()`方法会默认的去所有的子孙节点中搜索，而如果将`recursive`参数设置为False，则可以将搜索范围限制在直接子节点中。

```python
>>> soup.html.find_all("title")
[<title>The Dormouse's story</title>]

>>> soup.html.find_all("title", recursive=False)
[]
```

### 通过正则表达式来筛选查找结果

在`BeautifulSoup`中，也是可以与`re`模块进行相互配合的，将re.compile编译的对象传入`find_all()`方法，即可通过正则来进行搜索。

```python
In [1]: import re

In [2]: tags = soup.find_all(re.compile("^b"))

In [3]: [i.name for i in tags]
Out[3]: ['body', 'b']
```

可以看到，找到了标签名是以'b'开头的两个标签。

同样的，也能够以正则来筛选`tag`的属性。

```python
In [1]: soup.find_all(attrs={'class': re.compile("si")})
Out[1]:
[<a class="sister" href="http://example.com/elsie" id="link1">Elsie</a>,
 <a class="sister" href="http://example.com/lacie" id="link2">Lacie</a>,
 <a class="sister" href="http://example.com/tillie" id="link3">Tillie</a>]
```

------

### CSS选择器

在`BeautifulSoup`中，同样也支持使用CSS选择器来进行搜索。使用`select()`，在其中传入字符串参数，就可以使用CSS选择器的语法来找到tag。

```python
>>> soup.select("title")
[<title>The Dormouse's story</title>]

>>> soup.select("p > a")
[<a class="sister" href="http://example.com/elsie" id="link1">Elsie</a>,
 <a class="sister" href="http://example.com/lacie"  id="link2">Lacie</a>,
 <a class="sister" href="http://example.com/tillie" id="link3">Tillie</a>]
```

转载请注明：[Seven的博客](http://sevenold.github.io)