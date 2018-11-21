---
layout: post
title: "优雅到骨子里的Requests"
date: 2018-11-21
description: "优雅到骨子里的Requests"
tag: 爬虫
---

转载自：简书-[王南北](https://www.jianshu.com/u/948da055a416)

### 优雅到骨子里的Requests

![Requests-标志](/images/reptile/5.png)

------

### 简介

上一篇文章介绍了Python的网络请求库`urllib`和`urllib3`的使用方法，那么，作为同样是网络请求库的`Requests`，相对于`urllib`，有什么优点呢？

其实，只有两个词，简单优雅。

`Requests`的宣言就是：**HTTP for Humans**。可以说，`Requests`彻底贯彻了Python所代表的简单优雅的精神。

之前的`urllib`做为Python的标准库，因为历史原因，使用的方式可以说是非常的麻烦而复杂的，而且官方文档也十分的简陋，常常需要去查看源码。与之相反的是，`Requests`的使用方式非常的简单、直观、人性化，让程序员的精力完全从库的使用中解放出来。

甚至在官方的urllib.request的文档中，有这样一句话来推荐`Requests`：

> The **Requests packageis** recommended for a higher-level HTTP client interface.

`Requests`的官方文档同样也非常的完善详尽，而且少见的有中文官方文档：[http://cn.python-requests.org/zh_CN/latest/](http://cn.python-requests.org/zh_CN/latest/)。

当然，为了保证准确性，还是尽量查看英文文档为好。

------

### 作者

`Requests`的作者**Kenneth Reitz**同样是一个富有传奇色彩的人物。

**Kenneth Reitz**在有着“云服务鼻祖”之称的Heroku 公司，28岁时就担任了Python 语言的总架构师。他做了什么呢？随便列几个项目名称: requests、python-guide、pipenv、legit、autoenv，当然它也给Python界很多知名的开源项目贡献了代码，比如Flask。

可以说他是Python领域举足轻重的人物，他的代码追求一种强迫症般的美感。

大佬的传奇还不止于此，这是他当年在PyCON演讲时的照片：

![Requests-作者1](/images/reptile/6.png)

非常可爱的小胖子，同时也符合着大众对于程序员的一些刻板印象：胖、不太修边幅、腼腆。

但是几年后，他变成了这样：

![Requests-作者2](/images/reptile/7.png)

emmmmm，帅哥，你这是去哪整的容？

哈哈，开个玩笑。不过确实外貌方面的改变非常的巨大，由一个小肥宅的形象变得帅气潇洒。可见只要愿意去追求，我们都能变成我们想要的样子。

------

### 例子与特性

可以说`Requests`最大的特性就是其风格的简单直接优雅。无论是请求方法，还是响应结果的处理，还有cookies，url参数，post提交数据，都体现出了这种风格。

以下是一个简单例子：

```python
>>> import requests
>>> resp = requests.get('https://www.baidu.com')

>>> resp.status_code
200

>>> resp.headers['content-type']
'application/json; charset=utf8'

>>> resp.encoding
'utf-8'

>>> resp.text
u'{"type":"User"...'
```

可以看到，不论是请求的发起还是相应的处理，都是非常直观明了的。

`Requests`目前基本上完全满足web请求的所有需求，以下是`Requests`的特性：

- Keep-Alive & 连接池
- 国际化域名和 URL
- 带持久 Cookie 的会话
- 浏览器式的 SSL 认证
- 自动内容解码
- 基本/摘要式的身份认证
- 优雅的 key/value Cookie
- 自动解压
- Unicode 响应体
- HTTP(S) 代理支持
- 文件分块上传
- 流下载
- 连接超时
- 分块请求
- 支持 `.netrc`

而`Requests 3.0`目前也募集到了资金正在开发中，预计会支持async/await来实现并发请求，且可能会支持HTTP 2.0。

------

### 安装

`Requests`的安装非常的简单，直接PIP安装即可：

```python
pip install requests
```

------

### 使用

`Requests`的请求不再像`urllib`一样需要去构造各种Request、opener和handler，使用`Requests`构造的方法，并在其中传入需要的参数即可。

### 发起请求

### 请求方法

每一个请求方法都有一个对应的API，比如GET请求就可以使用`get()`方法：

```python
>>> import requests
>>> resp = requests.get('https://www.baidu.com')
```

而POST请求就可以使用`post()`方法，并且将需要提交的数据传递给data参数即可：

```python
>>> import requests
>>> resp = requests.post('http://httpbin.org/post', data = {'key':'value'})
```

而其他的请求类型，都有各自对应的方法：

```python
>>> resp = requests.put('http://httpbin.org/put', data = {'key':'value'})
>>> resp = requests.delete('http://httpbin.org/delete')
>>> resp = requests.head('http://httpbin.org/get')
>>> resp = requests.options('http://httpbin.org/get')
```

非常的简单直观明了。

### 传递URL参数

传递URL参数也不用再像`urllib`中那样需要去拼接URL，而是简单的，构造一个字典，并在请求时将其传递给params参数：

```python
>>> import requests
>>> params = {'key1': 'value1', 'key2': 'value2'}
>>> resp = requests.get("http://httpbin.org/get", params=params)
```

此时，查看请求的URL，则可以看到URL已经构造正确了：

```python
>>> print(resp.url)
http://httpbin.org/get?key2=value2&key1=value1
```

并且，有时候我们会遇到相同的url参数名，但有不同的值，而python的字典又不支持键的重名，那么我们可以把键的值用列表表示：

```python
>>> params = {'key1': 'value1', 'key2': ['value2', 'value3']}
>>> resp = requests.get('http://httpbin.org/get', params=params)
>>> print(resp.url)
http://httpbin.org/get?key1=value1&key2=value2&key2=value3
```

### 自定义Headers

如果想自定义请求的Headers，同样的将字典数据传递给headers参数。

```python
>>> url = 'https://api.github.com/some/endpoint'
>>> headers = {'user-agent': 'my-app/0.0.1'}
>>> resp = requests.get(url, headers=headers)
```

### 自定义Cookies

`Requests`中自定义Cookies也不用再去构造CookieJar对象，直接将字典递给cookies参数。

```python
>>> url = 'http://httpbin.org/cookies'
>>> cookies = {'cookies_are': 'working'}

>>> resp = requests.get(url, cookies=cookies)
>>> resp.text
'{"cookies": {"cookies_are": "working"}}'
```

### 设置代理

当我们需要使用代理时，同样构造代理字典，传递给`proxies`参数。

```python
import requests

proxies = {
  'http': 'http://10.10.1.10:3128',
  'https': 'http://10.10.1.10:1080',
}

requests.get('http://example.org', proxies=proxies)
```

### 重定向

在网络请求中，我们常常会遇到状态码是3开头的重定向问题，在`Requests`中是默认开启允许重定向的，即遇到重定向时，会自动继续访问。

```python
>>> resp = requests.get('http://github.com', allow_redirects=False)
>>> resp.status_code
301
```

### 禁止证书验证

有时候我们使用了抓包工具，这个时候由于抓包工具提供的证书并不是由受信任的数字证书颁发机构颁发的，所以证书的验证会失败，所以我们就需要关闭证书验证。

在请求的时候把`verify`参数设置为`False`就可以关闭证书验证了。

```python
>>> import requests
>>> resp = requests.get('http://httpbin.org/post', verify=False)
```

但是关闭验证后，会有一个比较烦人的`warning`

```python
py:858: InsecureRequestWarning: Unverified HTTPS request is being made. Adding certificate verification is strongly advised. See: https://urllib3.readthedocs.io/en/latest/advanced-usage.html#ssl-warnings
  InsecureRequestWarning)
```

可以使用以下方法关闭警告：

```python
from requests.packages.urllib3.exceptions import InsecureRequestWarning
# 禁用安全请求警告
requests.packages.urllib3.disable_warnings(InsecureRequestWarning)
```

### 设置超时

设置访问超时，设置`timeout`参数即可。

```python
>>> requests.get('http://github.com', timeout=0.001)
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
requests.exceptions.Timeout: HTTPConnectionPool(host='github.com', port=80): Request timed out. (timeout=0.001)
```

可见，通过`Requests`发起请求，只需要构造好几个需要的字典，并将其传入请求的方法中，即可完成基本的网络请求。

------

### 响应

通过`Requests`发起请求获取到的，是一个`requests.models.Response`对象。通过这个对象我们可以很方便的获取响应的内容。

### 响应内容

之前通过`urllib`获取的响应，读取的内容都是bytes的二进制格式，需要我们自己去将结果`decode()`一次转换成字符串数据。

而`Requests`通过`text`属性，就可以获得字符串格式的响应内容。

```python
>>> import requests

>>> resp = requests.get('https://api.github.com/events')
>>> resp.text
u'[{"repository":{"open_issues":0,"url":"https://github.com/...
```

`Requests`会自动的根据响应的报头来猜测网页的编码是什么，然后根据猜测的编码来解码网页内容，基本上大部分的网页都能够正确的被解码。而如果发现`text`解码不正确的时候，就需要我们自己手动的去指定解码的编码格式。

```python
>>> import requests

>>> resp = requests.get('https://api.github.com/events')
>>> resp.encoding = 'utf-8'
>>> resp.text
u'[{"repository":{"open_issues":0,"url":"https://github.com/...
```

而如果你需要获得原始的二进制数据，那么使用`content`属性即可。

```python
>>> resp.content
b'[{"repository":{"open_issues":0,"url":"https://github.com/...
```

如果我们访问之后获得的数据是JSON格式的，那么我们可以使用`json()`方法，直接获取转换成字典格式的数据。

```python
>>> import requests

>>> resp = requests.get('https://api.github.com/events')
>>> resp.json()
[{u'repository': {u'open_issues': 0, u'url': 'https://github.com/...
```

### 状态码

通过`status_code`属性获取响应的状态码

```python
>>> resp = requests.get('http://httpbin.org/get')
>>> resp.status_code
200
```

### 响应报头

通过`headers`属性获取响应的报头

```python
>>> r.headers
{
    'content-encoding': 'gzip',
    'transfer-encoding': 'chunked',
    'connection': 'close',
    'server': 'nginx/1.0.4',
    'x-runtime': '148ms',
    'etag': '"e1ca502697e5c9317743dc078f67693f"',
    'content-type': 'application/json'
}
```

### 服务器返回的cookies

通过`cookies`属性获取服务器返回的`cookies`

```python
>>> url = 'http://example.com/some/cookie/setting/url'
>>> resp = requests.get(url)
>>> resp.cookies['example_cookie_name']
'example_cookie_value'
```

### url

还可以使用`url`属性查看访问的url。

```python
>>> import requests
>>> params = {'key1': 'value1', 'key2': 'value2'}
>>> resp = requests.get("http://httpbin.org/get", params=params)
>>> print(resp.url)
http://httpbin.org/get?key2=value2&key1=value1
```

------

### Session

在`Requests`中，实现了`Session(会话)`功能，当我们使用`Session`时，能够像浏览器一样，在没有关闭关闭浏览器时，能够保持住访问的状态。

这个功能常常被我们用于登陆之后的数据获取，使我们不用再一次又一次的传递cookies。

```python
import requests

session = requests.Session()

session.get('http://httpbin.org/cookies/set/sessioncookie/123456789')
resp = session.get('http://httpbin.org/cookies')

print(resp.text)
# '{"cookies": {"sessioncookie": "123456789"}}'
```

首先我们需要去生成一个`Session`对象，然后用这个`Session`对象来发起访问，发起访问的方法与正常的请求是一摸一样的。

同时，需要注意的是，如果是我们在`get()`方法中传入`headers`和`cookies`等数据，那么这些数据只在当前这一次请求中有效。如果你想要让一个`headers`在`Session`的整个生命周期内都有效的话，需要用以下的方式来进行设置：

```python
# 设置整个headers
session.headers = {
    'user-agent': 'my-app/0.0.1'
}
# 增加一条headers
session.headers.update({'x-test': 'true'})
```

------

** 后记：或许有人不认可代码的美学，认为代码写的丑没事，能跑起来就好。但是我始终认为，世间万物都应该是美好的，追求美好的脚步也不应该停止。**

转载请注明：[Seven的博客](http://sevenold.github.io)