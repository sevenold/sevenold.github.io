---
layout: post
title: "Python网络请求urllib和urllib3详解"
date: 2018-11-21
description: "Python网络请求urllib和urllib3详解"
tag: 爬虫
---

转载自：简书-[王南北](https://www.jianshu.com/u/948da055a416)

### Python网络请求urllib和urllib3详解

`urllib`是Python中请求url连接的官方标准库，在Python2中主要为urllib和urllib2，在Python3中整合成了urllib。

而urllib3则是增加了连接池等功能，两者互相都有补充的部分。

### urllib

urllib作为Python的标准库，基本上涵盖了基础的网络请求功能。

------

### urllib.request

urllib中，`request`这个模块主要负责构造和发起网络请求，并在其中加入Headers、Proxy等。

### 发起GET请求

主要使用`urlopen()`方法来发起请求：

```python
from urllib import request

resp = request.urlopen('http://www.baidu.com')
print(resp.read().decode())
```

在`urlopen()`方法中传入字符串格式的url地址，则此方法会访问目标网址，然后返回访问的结果。

访问的结果会是一个`http.client.HTTPResponse`对象，使用此对象的`read()`方法，则可以获取访问网页获得的数据。但是要注意的是，获得的数据会是`bytes`的二进制格式，所以需要`decode()`一下，转换成字符串格式。

### 发起POST请求

`urlopen()`默认的访问方式是GET，当在`urlopen()`方法中传入data参数时，则会发起POST请求。**注意：传递的data数据需要为bytes格式。**

设置timeout参数还可以设置超时时间，如果请求时间超出，那么就会抛出异常。

```python
from urllib import request

resp = request.urlopen('http://httpbin.org', data=b'word=hello', timeout=10)
print(resp.read().decode())
```

### 添加Headers

通过`urllib`发起的请求会有默认的一个Headers："User-Agent":"Python-urllib/3.6"，指明请求是由`urllib`发送的。

所以遇到一些验证User-Agent的网站时，我们需要自定义Headers，而这需要借助于urllib.request中的`Request`对象。

```python
from urllib import request

url = 'http://httpbin.org/get'
headers = {'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_13_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/66.0.3359.181 Safari/537.36'}

# 需要使用url和headers生成一个Request对象，然后将其传入urlopen方法中
req = request.Request(url, headers=headers)
resp = request.urlopen(req)
print(resp.read().decode())
```

### Request对象

如上所示，`urlopen()`方法中不止可以传入字符串格式的url，也可以传入一个`Request`对象来扩展功能，`Request`对象如下所示。

```python
class urllib.request.Request(url, data=None, headers={},
                             origin_req_host=None,
                             unverifiable=False, method=None)
```

构造`Request`对象必须传入url参数，data数据和headers都是可选的。

最后，`Request`方法可以使用method参数来自由选择请求的方法，如PUT，DELETE等等，默认为GET。

### 添加Cookie

为了在请求时能带上Cookie信息，我们需要重新构造一个opener。

使用request.build_opener方法来进行构造opener，将我们想要传递的cookie配置到opener中，然后使用这个opener的open方法来发起请求。

```python
from http import cookiejar
from urllib import request

url = 'http://httpbin.org/cookies'
# 创建一个cookiejar对象
cookie = cookiejar.CookieJar()
# 使用HTTPCookieProcessor创建cookie处理器
cookies = request.HTTPCookieProcessor(cookie)
# 并以它为参数创建Opener对象
opener = request.build_opener(cookies)
# 使用这个opener来发起请求
resp = opener.open(url)
print(resp.read().decode())
```

或者也可以把这个生成的opener使用install_opener方法来设置为全局的。

则之后使用urlopen方法发起请求时，都会带上这个cookie。

```python
# 将这个opener设置为全局的opener
request.install_opener(opener)
resp = request.urlopen(url)
```

### 设置Proxy代理

使用爬虫来爬取数据的时候，常常需要使用代理来隐藏我们的真实IP。

```python
from urllib import request

url = 'http://httpbin.org/ip'
proxy = {'http':'50.233.137.33:80','https':'50.233.137.33:80'}
# 创建代理处理器
proxies = request.ProxyHandler(proxy)
# 创建opener对象
opener = request.build_opener(proxies)

resp = opener.open(url)
print(resp.read().decode())
```

### 下载数据到本地

在我们进行网络请求时常常需要保存图片或音频等数据到本地，一种方法是使用python的文件操作，将read()获取的数据保存到文件中。

而`urllib`提供了一个`urlretrieve()`方法，可以简单的直接将请求获取的数据保存成文件。

```python
from urllib import request

url = 'http://python.org/'
request.urlretrieve(url, 'python.html')
```

`urlretrieve()`方法传入的第二个参数为文件保存的位置，以及文件名。

注：`urlretrieve()`方法是python2直接移植过来的方法，以后有可能在某个版本中弃用。

------

### urllib.response

在使用`urlopen()`方法或者opener的`open()`方法发起请求后，获得的结果是一个`response`对象。

这个对象有一些方法和属性，可以让我们对请求返回的结果进行一些处理。

- **read()**

  获取响应返回的数据，只能使用一次。

- **getcode()**

  获取服务器返回的状态码。

- **getheaders()**

  获取返回响应的响应报头。

- **geturl()**

  获取访问的url。

------

### urllib.parse

`urllib.parse`是urllib中用来解析各种数据格式的模块。

### urllib.parse.quote

在url中，是只能使用ASCII中包含的字符的，也就是说，ASCII不包含的特殊字符，以及中文等字符都是不可以在url中使用的。而我们有时候又有将中文字符加入到url中的需求，例如百度的搜索地址：

`https://www.baidu.com/s?wd=南北`

？之后的wd参数，则是我们搜索的关键词。那么我们实现的方法就是将特殊字符进行url编码，转换成可以url可以传输的格式，urllib中可以使用`quote()`方法来实现这个功能。

```python
>>> from urllib import parse
>>> keyword = '南北'
>>> parse.quote(keyword)
'%E5%8D%97%E5%8C%97'
```

如果需要将编码后的数据转换回来，可以使用`unquote()`方法。

```python
>>> parse.unquote('%E5%8D%97%E5%8C%97')
'南北'
```

### urllib.parse.urlencode

在访问url时，我们常常需要传递很多的url参数，而如果用字符串的方法去拼接url的话，会比较麻烦，所以`urllib`中提供了`urlencode`这个方法来拼接url参数。

```python
>>> from urllib import parse
>>> params = {'wd': '南北', 'code': '1', 'height': '188'}
>>> parse.urlencode(params)
'wd=%E5%8D%97%E5%8C%97&code=1&height=188'
```

------

### urllib.error

在`urllib`中主要设置了两个异常，一个是`URLError`，一个是`HTTPError`，`HTTPError`是`URLError`的子类。

`HTTPError`还包含了三个属性：

- code：请求的状态码
- reason：错误的原因
- headers：响应的报头

例子：

```python
In [1]: from urllib.error import HTTPError

In [2]: try:
   ...:     request.urlopen('https://www.jianshu.com')
   ...: except HTTPError as e:
   ...:     print(e.code)

403
```

------

### urllib3

Urllib3是一个功能强大，条理清晰，用于HTTP客户端的Python库。许多Python的原生系统已经开始使用urllib3。Urllib3提供了很多python标准库urllib里所没有的重要特性：

1. 线程安全
2. 连接池
3. 客户端SSL/TLS验证
4. 文件分部编码上传
5. 协助处理重复请求和HTTP重定位
6. 支持压缩编码
7. 支持HTTP和SOCKS代理

------

### 安装

urllib3是一个第三方库，安装非常简单，pip安装即可：

```python
pip install urllib3
```

------

### 使用

`urllib3`主要使用连接池进行网络请求的访问，所以访问之前我们需要创建一个连接池对象，如下所示：

```python
>>> import urllib3
>>> http = urllib3.PoolManager()
>>> r = http.request('GET', 'http://httpbin.org/robots.txt')
>>> r.status
200
>>> r.data
'User-agent: *\nDisallow: /deny\n'
```

### 设置headers

```python
headers={'X-Something': 'value'}
resp = http.request('GET', 'http://httpbin.org/headers', headers=headers)
```

### 设置url参数

对于GET等没有请求正文的请求方法，可以简单的通过设置`fields`参数来设置url参数。

```python
fields = {'arg': 'value'}
resp = http.request('GET', 'http://httpbin.org/get', fields=fields)
```

如果使用的是POST等方法，则会将fields作为请求的请求正文发送。

所以，如果你的POST请求是需要url参数的话，那么需要自己对url进行拼接。

```python
fields = {'arg': 'value'}
resp = http.request('POST', 'http://httpbin.org/get', fields=fields)
```

### 设置代理

```python
>>> import urllib3
>>> proxy = urllib3.ProxyManager('http://50.233.137.33:80', headers={'connection': 'keep-alive'})
>>> resp = proxy.request('get', 'http://httpbin.org/ip')
>>> resp.status
200
>>> resp.data
b'{"origin":"50.233.136.254"}\n'
```

**注：`urllib3`中没有直接设置cookies的方法和参数，只能将cookies设置到headers中**

转载请注明：[Seven的博客](http://sevenold.github.io)