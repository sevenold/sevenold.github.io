---
layout: post
title: "Python与常见加密方式"
date: 2018-11-21
description: "Python与常见加密方式"
tag: python
---

转载自：简书-[王南北](https://www.jianshu.com/u/948da055a416)

### Python与常见加密方式

本文地址：https://www.jianshu.com/p/4ba20afacce2

------

### 前言

我们所说的加密方式，都是对二进制编码的格式进行加密的，对应到Python中，则是我们的`Bytes`。

所以当我们在Python中进行加密操作的时候，要确保我们操作的是`Bytes`，否则就会报错。

将字符串和`Bytes`互相转换可以使用`encode()`和`decode()`方法。如下所示：

```python
# 方法中不传参数则是以默认的utf-8编码进行转换
In [1]: '南北'.encode()
Out[1]: b'\xe5\x8d\x97\xe5\x8c\x97'

In [2]: b'\xe5\x8d\x97\xe5\x8c\x97'.decode()
Out[2]: '南北'
```

**注：两位十六进制常常用来显示一个二进制字节。**

利用`binascii`模块可以将十六进制显示的字节转换成我们在加解密中更常用的显示方式：

```python
In [1]: import binascii

In [2]: '南北'.encode()
Out[2]: b'\xe5\x8d\x97\xe5\x8c\x97'

In [3]: binascii.b2a_hex('南北'.encode())
Out[3]: b'e58d97e58c97'

In [4]: binascii.a2b_hex(b'e58d97e58c97')
Out[4]: b'\xe5\x8d\x97\xe5\x8c\x97'

In [5]: binascii.a2b_hex(b'e58d97e58c97').decode()
Out[5]: '南北'
```

------

## URL编码

### 简介

正常的URL中是只能包含ASCII字符的，也就是字符、数字和一些符号。而URL编码就是一种浏览器用来避免url中出现特殊字符（如汉字）的编码方式。

其实就是将超出ASCII范围的字符转换成带`%`的十六进制格式。

### Python实现

```python
In [1]: from urllib import parse

# quote()方法会自动将str转换成bytes，所以这里传入str和bytes都可以
In [2]: parse.quote('南北')
Out[2]: '%E5%8D%97%E5%8C%97'

In [3]: parse.unquote('%E5%8D%97%E5%8C%97')
Out[3]: '南北'
```

------

## Base64编码

### 简述

Base64是一种用64个字符来表示任意二进制数据的方法。

Base64编码可以成为密码学的基石。可以将任意的二进制数据进行Base64编码。所有的数据都能被编码为并只用65个字符就能表示的文本文件。（ 65字符：A~Z a~z 0~9 + / = ）编码后的数据~=编码前数据的4/3，会大1/3左右。

### Base64编码的原理 

![加密方式-base64原理](/images/py/69.png)

1. 将所有字符转化为ASCII码。
2. 将ASCII码转化为8位二进制 。
3. 将二进制3个归成一组(不足3个在后边补0)共24位，再拆分成4组，每组6位。
4. 统一在6位二进制前补两个0凑足8位。
5. 将补0后的二进制转为十进制。
6. 从Base64编码表获取十进制对应的Base64编码。

### Base64编码的说明 

1. 转换的时候，将三个byte的数据，先后放入一个24bit的缓冲区中，先来的byte占高位。 
2. 数据不足3byte的话，于缓冲区中剩下的bit用0补足。然后，每次取出6个bit，按照其值选择查表选择对应的字符作为编码后的输出。 
3. 不断进行，直到全部输入数据转换完成。 
4. 如果最后剩下两个输入数据，在编码结果后加1个“=”。
5. 如果最后剩下一个输入数据，编码结果后加2个“=”。
6. 如果没有剩下任何数据，就什么都不要加，这样才可以保证资料还原的正确性。

### Python的Base64使用

Python内置的`base64`模块可以直接进行base64的编解码

***注意：用于base64编码的，要么是ASCII包含的字符，要么是二进制数据***

```python
In [1]: import base64

In [2]: base64.b64encode(b'hello world')
Out[2]: b'aGVsbG8gd29ybGQ='

In [3]: base64.b64decode(b'aGVsbG8gd29ybGQ=')
Out[3]: b'hello world'
```

------

## MD5（信息-摘要算法）

### 简述

message-digest algorithm 5（信息-摘要算法）。经常说的“MD5加密”，就是它→信息-摘要算法。

md5，其实就是一种算法。可以将一个字符串，或文件，或压缩包，执行md5后，就可以生成一个固定长度为128bit的串。这个串，基本上是唯一的。

### 不可逆性

每个人都有不同的指纹，看到这个人，可以得出他的指纹等信息，并且唯一对应，但你只看一个指纹，是不可能看到或读到这个人的长相或身份等信息。

### 特点

1. 压缩性：任意长度的数据，算出的MD5值长度都是固定的。
2. 容易计算：从原数据计算出MD5值很容易。
3. 抗修改性：对原数据进行任何改动，哪怕只修改1个字节，所得到的MD5值都有很大区别。
4. 强抗碰撞：已知原数据和其MD5值，想找到一个具有相同MD5值的数据（即伪造数据）是非常困难的。

举个栗子：世界上只有一个我，但是但是妞却是非常非常多的，以一个有限的我对几乎是无限的妞，所以可能能搞定非常多（100+）的妞，这个理论上的确是通的，可是实际情况下....

### Python的MD5使用

由于MD5模块在python3中被移除，在python3中使用`hashlib`模块进行md5操作

```python
import hashlib

# 待加密信息
str = '这是一个测试'

# 创建md5对象
hl = hashlib.md5()

# 此处必须声明encode
# 若写法为hl.update(str)  报错为： Unicode-objects must be encoded before hashing
hl.update(str.encode(encoding='utf-8'))

print('MD5加密前为 ：' + str)
print('MD5加密后为 ：' + hl.hexdigest())
```

运行结果

```
MD5加密前为 ：这是一个测试
MD5加密后为 ：cfca700b9e09cf664f3ae80733274d9f
```

### MD5长度

md5的长度，默认为128bit，也就是128个0和1的二进制串。这样表达是很不友好的。所以将二进制转成了16进制，每4个bit表示一个16进制，所以128/4 = 32 换成16进制表示后，为32位了。

为什么网上还有md5是16位的呢？

其实16位的长度，是从32位md5值来的。是将32位md5去掉前八位，去掉后八位得到的。

------

## Python加密库PyCryptodome

PyCrypto是 Python 中密码学方面最有名的第三方软件包。可惜的是，它的开发工作于2012年就已停止。

幸运的是，有一个该项目的分支PyCrytodome 取代了 PyCrypto 。

### 安装与导入

安装之前需要先安装***Microsoft Visual c++ 2015***。

在Linux上安装，可以使用以下 pip 命令：

```shell
pip install pycryptodome
```

导入：

```python
import Crypto
```

在Windows 系统上安装则稍有不同：

```shell
pip install pycryptodomex
```

导入：

```python
import Cryptodome
```

------

## DES

### 简介

DES算法为密码体制中的对称密码体制，又被称为美国数据加密标准。

DES是一个分组加密算法，典型的DES以64位为分组对数据加密，加密和解密用的是同一个算法。

DES算法的入口参数有三个：Key、Data、Mode。其中Key为7个字节共56位，是DES算法的工作密钥；Data为8个字节64位，是要被加密或被解密的数据；Mode为DES的工作方式,有两种:加密或解密。

密钥长64位，密钥事实上是56位参与DES运算（第8、16、24、32、40、48、56、64位是校验位，使得每个密钥都有奇数个1），分组后的明文组和56位的密钥按位替代或交换的方法形成密文组。

### 例子

```python
# 导入DES模块
from Cryptodome.Cipher import DES
import binascii

# 这是密钥
key = b'abcdefgh'
# 需要去生成一个DES对象
des = DES.new(key, DES.MODE_ECB)
# 需要加密的数据
text = 'python spider!'
text = text + (8 - (len(text) % 8)) * '='

# 加密的过程
encrypto_text = des.encrypt(text.encode())
encrypto_text = binascii.b2a_hex(encrypto_text)
print(encrypto_text)
```

------

## 3DES

### 简介

3DES（或称为Triple DES）是三重数据加密算法（TDEA，Triple Data Encryption Algorithm）块密码的通称。它相当于是对每个数据块应用三次DES加密算法。

由于计算机运算能力的增强，原版DES密码的密钥长度变得容易被暴力破解。3DES即是设计用来提供一种相对简单的方法，即通过增加DES的密钥长度来避免类似的攻击，而不是设计一种全新的块密码算法。

3DES（即Triple DES）是DES向AES过渡的加密算法（1999年，NIST将3-DES指定为过渡的加密标准），加密算法，其具体实现如下：设Ek()和Dk()代表DES算法的加密和解密过程，K代表DES算法使用的密钥，M代表明文，C代表密文，这样：

3DES加密过程为：C=Ek3(Dk2(Ek1(M)))

3DES解密过程为：M=Dk1(EK2(Dk3(C)))

------

## AES

### 简介

**高级加密标准**（英语：**Advanced Encryption Standard**，缩写：**AES**），在密码学中又称**Rijndael加密法**，是美国联邦政府采用的一种区块加密标准。这个标准用来替代原先的DES，已经被多方分析且广为全世界所使用。经过五年的甄选流程，高级加密标准由美国国家标准与技术研究院（NIST）于2001年11月26日发布于FIPS PUB 197，并在2002年5月26日成为有效的标准。2006年，高级加密标准已然成为对称密钥加密中最流行的算法之一。

AES在软件及硬件上都能快速地加解密，相对来说较易于实作，且只需要很少的存储器。作为一个新的加密标准，目前正被部署应用到更广大的范围。

### 特点与思想

1. 抵抗所有已知的攻击。
2. 在多个平台上速度快，编码紧凑。
3. 设计简单。

### 详解

![加密方式-对称加密](/images/py/70.png)

AES为分组密码，分组密码也就是把明文分成一组一组的，每组长度相等，每次加密一组数据，直到加密完整个明文。在AES标准规范中，分组长度只能是128位，也就是说，每个分组为16个字节（每个字节8位）。密钥的长度可以使用128位、192位或256位。密钥的长度不同，推荐加密轮数也不同。

**一般常用的是128位**

###Python实现

```python
from Cryptodome.Cipher import AES
from Cryptodome import Random
from binascii import b2a_hex  

# 要加密的明文
data = '南来北往'
# 密钥key 长度必须为16（AES-128）、24（AES-192）、或32（AES-256）Bytes 长度.
# 目前AES-128足够用
key = b'this is a 16 key'
# 生成长度等于AES块大小的不可重复的密钥向量
iv = Random.new().read(AES.block_size)

# 使用key和iv初始化AES对象, 使用MODE_CFB模式
mycipher = AES.new(key, AES.MODE_CFB, iv)
# 加密的明文长度必须为16的倍数，如果长度不为16的倍数，则需要补足为16的倍数
# 将iv（密钥向量）加到加密的密文开头，一起传输
ciphertext = iv + mycipher.encrypt(data.encode())

# 解密的话要用key和iv生成新的AES对象
mydecrypt = AES.new(key, AES.MODE_CFB, ciphertext[:16])
# 使用新生成的AES对象，将加密的密文解密
decrypttext = mydecrypt.decrypt(ciphertext[16:])


print('密钥k为：', key)
print('iv为：', b2a_hex(ciphertext)[:16])
print('加密后数据为：', b2a_hex(ciphertext)[16:])
print('解密后数据为：', decrypttext.decode())

```

运行结果：

```python
密钥k为： b'this is a 16 key'
iv为： b'a78a177cffd50878'
加密后数据为： b'33f61e7678c25d795d565d40f2f68371da051202'
解密后数据为： 南来北往
```

------

### RSA

### 非对称加密

典型的如RSA等，常见方法，使用openssl ,keytools等工具生成一对公私钥对，使用被公钥加密的数据可以使用私钥来解密，反之亦然（被私钥加密的数据也可以被公钥解密) 。

在实际使用中私钥一般保存在发布者手中，是私有的不对外公开的，只将公钥对外公布，就能实现只有私钥的持有者才能将数据解密的方法。   这种加密方式安全系数很高，因为它不用将解密的密钥进行传递，从而没有密钥在传递过程中被截获的风险，而破解密文几乎又是不可能的。

但是算法的效率低，所以常用于很重要数据的加密，常和对称配合使用，使用非对称加密的密钥去加密对称加密的密钥。

### 简介

**RSA加密算法**是一种`非对称加密算法`。在公开密钥加密和电子商业中RSA被广泛使用。

该算法基于一个十分简单的数论事实：将两个大素数相乘十分容易，但那时想要对其乘积进行因式分解却极其困难，因此可以将乘积公开作为加密密钥，即公钥，而两个大素数组合成私钥。公钥是可发布的供任何人使用，私钥则为自己所有，供解密之用。

### Python实现

首先我们需要安装一个`rsa`模块：

```shell
pip install rsa
```

而且，因为RSA加密算法的特性，RSA的公钥私钥都是10进制的，但公钥的值常常保存为16进制的格式，所以需要将其用`int()`方法转换为10进制格式。

#### 用网页中的公钥把数据加密

```python
import rsa
import binascii

# 使用网页中获得的n和e值，将明文加密
def rsa_encrypt(rsa_n, rsa_e, message):
    # 用n值和e值生成公钥
    key = rsa.PublicKey(rsa_n, rsa_e)
    # 用公钥把明文加密
    message = rsa.encrypt(message.encode(), key)
    # 转化成常用的可读性高的十六进制
    message = binascii.b2a_hex(message)
    # 将加密结果转化回字符串并返回
    return message.decode()

# RSA的公钥有两个值n和e，我们在网站中获得的公钥一般就是这样的两个值。
# n常常为长度为256的十六进制字符串
# e常常为十六进制‘10001’
pubkey_n = '8d7e6949d411ce14d7d233d7160f5b2cc753930caba4d5ad24f923a505253b9c39b09a059732250e56c594d735077cfcb0c3508e9f544f101bdf7e97fe1b0d97f273468264b8b24caaa2a90cd9708a417c51cf8ba35444d37c514a0490441a773ccb121034f29748763c6c4f76eb0303559c57071fd89234d140c8bb965f9725'
pubkey_e = '10001'
# 需要将十六进制转换成十进制
rsa_n = int(pubkey_n, 16)
rsa_e = int(pubkey_e, 16)
# 要加密的明文
message = '南北今天很忙'

print("公钥n值长度：", len(pubkey_n))
print(rsa_encrypt(rsa_n, rsa_e, message))
```

运行结果：

```python 
公钥n值长度： 256
480f302eed822c8250256511ddeb017fcb28949cc05739ae66440eecc4ab76e7a7b2f1df398aefdfef2b9bfce6d6152bf6cc1552a0ed8bebee9e094a7ce9a52622487a6412632144787aa81f6ec9b96be95890c4c28a31b3e8d9ea430080d79297c5d75cd11df04df6e71b237511164399d72ccb2f4c34022b1ea7b76189a56e
```



转载请注明：[Seven的博客](http://sevenold.github.io)