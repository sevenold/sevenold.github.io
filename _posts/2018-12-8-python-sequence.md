---
layout: post
title: "python第二话之序列类型方法"
date: 2018-12-08
description: "python序列类型方法，列表字符串，元组，增删改查"
tag: python
---

[TOC]



### 序列类型方法

前面我们简单的介绍了什么是序列类型，并且简单的给大家提了下`字符串`、`元组`、`列表`的一些基本操作。接下来我们详细的来看下关于`字符串`、`元组`、`列表`的常用操作。

#### 列表常用方法

我们先查看下`list`可以使用的一些方法：

>   append(...)
> |      L.append(object) -> None -- append object to end
> |  
> |  clear(...)
> |      L.clear() -> None -- remove all items from L
> |  
> |  copy(...)
> |      L.copy() -> list -- a shallow copy of L
> |  
> |  count(...)
> |      L.count(value) -> integer -- return number of occurrences of value
> |  
> |  extend(...)
> |      L.extend(iterable) -> None -- extend list by appending elements from the iterable
> |  
> |  index(...)
> |      L.index(value, [start, [stop]]) -> integer -- return first index of value.
> |      Raises ValueError if the value is not present.
> |  
> |  insert(...)
> |      L.insert(index, object) -- insert object before index
> |  
> |  pop(...)
> |      L.pop([index]) -> item -- remove and return item at index (default last).
> |      Raises IndexError if list is empty or index is out of range.
> |  
> |  remove(...)
> |      L.remove(value) -> None -- remove first occurrence of value.
> |      Raises ValueError if the value is not present.
> |  
> |  reverse(...)
> |      L.reverse() -- reverse *IN PLACE*
> |  
> |  sort(...)
> |      L.sort(key=None, reverse=False) -> None -- stable sort *IN PLAC

接下来我们给大家分类讲解一下这样操作方法。

##### `增加元素`

**append**

查看下`append`这个方法的用法：

```python
In [1]: s = [1,2,3,4,5,'a','b','c']                                    

In [2]: help(s.append)  
```

> append(...) method of builtins.list instance
> ​    L.append(object) -> None -- append object to end
>
> 使用方式：L.append('s')
>
> 注意：append方法是在列表的最后添加元素

**演示**：

```python
In [3]: s.append('w')                                                  

In [4]: s                                                              
Out[4]: [1, 2, 3, 4, 5, 'a', 'b', 'c', 'w']
```

**insert**

查看`insert`的使用方法：

```python
In [5]: help(s.insert)  
```

>insert(...) method of builtins.list instance
>​    L.insert(index, object) -- insert object before index
>
>使用方法： L.insert(index,'s')
>
>注意：index: 索引值，s为元素
>
>insert方法是在指定的索引值处添加元素

**演示：**

```python
In [6]: s.insert(3,'s')                                                

In [7]: s                                                              
Out[7]: [1, 2, 3, 's', 4, 5, 'a', 'b', 'c', 'w']
```

**extend**

查看`extend`的使用方法

```python
In [8]: help(s.extend) 
```

> extend(...) method of builtins.list instance
> ​    L.extend(iterable) -> None -- extend list by appending elements from the iterable
>
> 使用方法：L.extend([1,2,'a']) 
>
> 注意：extend方法是指在列表中追加可迭代器

**演示：**

```python
In [9]: a = ['w','o',0]                                                

In [10]: s.extend(a)                                                   

In [11]: s                                                             
Out[11]: [1, 2, 3, 's', 4, 5, 'a', 'b', 'c', 'w', 'w', 'o', 0]
```

##### `删除元素`

**clear**

查看下`clear`的使用方法

```python
In [14]: help(s.clear)  
```



> clear(...) method of builtins.list instance
> ​    L.clear() -> None -- remove all items from L
>
> 使用方法：L.clear()
>
> 注意：clear是删除所有的元素

**演示**：

```python
In [20]: s = [1,2,3,4,5,'a','b','c']                                   

In [21]: s.clear()                                                     

In [22]: s                                                             
Out[22]: []
```

**pop**

查看`pop`的使用方法：

```python
In [23]: help(s.pop)   
```

> pop(...) method of builtins.list instance
> ​    L.pop([index]) -> item -- remove and return item at index (default last).
> ​    Raises IndexError if list is empty or index is out of range.
>
> 使用方法：L.pop()、L.pop(index)
>
> 注意：L.pop()指的是删除最后一个元素
>
> ​		L.pop(index)指的是指定索引值删除元素

**演示**：

```python
In [24]: s = [1,2,3,4,5,'a','b','c']                                   

In [25]: s.pop()                                                       
Out[25]: 'c'

In [26]: s.pop()                                                       
Out[26]: 'b'

In [27]: s.pop()                                                       
Out[27]: 'a'

In [28]: s.pop()                                                       
Out[28]: 5

In [29]: s.pop(1)                                                      
Out[29]: 2

In [30]: s                                                             
Out[30]: [1, 3, 4]
```

**remove**

查看`remove`的使用方法：

```python
In [35]: help(s.remove)  
```

> remove(...) method of builtins.list instance
> ​    L.remove(value) -> None -- remove first occurrence of value.
> ​    Raises ValueError if the value is not present.
>
> 使用方法：L.remove(obj) 
>
> 注意：移除指定元素从左边开始的第一个。

**演示：**

```python
In [56]: s = [1,2,3,4,5,'a','b','c',1,2,3]                             

In [57]: s.remove(2)                                                   

In [58]: s                                                             
Out[58]: [1, 3, 4, 5, 'a', 'b', 'c', 1, 2, 3]

In [59]: s.remove(2)                                                   

In [60]: s                                                             
Out[60]: [1, 3, 4, 5, 'a', 'b', 'c', 1, 3]
```

##### `修改元素`

修改元素在列表中就非常简单了。

> 使用方法：L[index] = obj
>
> 注意：修改元素是根据元素的索引来修改。

**演示：**

```python
In [61]: s = [1,2,3,4,5]                                               

In [62]: s[3]='s'                                                      

In [63]: s                                                             
Out[63]: [1, 2, 3, 's', 5]
```

##### `查找元素`

**index**

查看index的使用方法

```python
In [64]: help(s.index)  
```

> index(...) method of builtins.list instance
> ​    L.index(value, [start, [stop]]) -> integer -- return first index of value.
> ​    Raises ValueError if the value is not present.
>
> 使用方法：L.index(obj) , L.index(value, [start, [stop]])
>
> 注意：L.index(obj) 从列表中找某个值第一个匹配项的索引位置。
>
> ​		L.index(value, [start, [stop]])指定索引范围查找元素

**演示**：

```python
In [71]: s = [1,2,3,4,5]                                               

In [72]: s.index(1)                                                    
Out[72]: 0

In [73]: s.index(3)                                                    
Out[73]: 2

In [98]: s = [1,2,3,4,5,'a','b','c',1,2,3]                             

In [99]: s.index(2,2)                                                  
Out[99]: 9

In [100]: s.index(2)                                                   
Out[100]: 1

```

**count**

查看`count`的使用方法

```python
In [74]: help(s.count)                                                 
```

> count(...) method of builtins.list instance
> ​    L.count(value) -> integer -- return number of occurrences of value
>
> 使用方法：L.count(obj)
>
> 注意：统计某个元素在列表中出现的次数。

**演示：**

```python
In [75]: s = [1,2,3,4,5,3,4,2]                                         

In [76]: s.count(2)                                                    
Out[76]: 2

In [77]: s.count(3)                                                    
Out[77]: 2

In [78]: s.count(1)                                                    
Out[78]: 1
```

##### `扩展`

**copy**

查看`copy`的使用方法

```python
In [79]: help(s.copy) 
```

> copy(...) method of builtins.list instance
> ​    L.copy() -> list -- a shallow copy of L
>
> 使用方法： L.copy()
>
> 注意： 复制列表，和L[:]的复制方式一样属于浅复制。

**演示：**

```python
In [101]: s = [1,2,3,4,5]                                              

In [102]: a = s.copy()                                                 

In [103]: a                                                            
Out[103]: [1, 2, 3, 4, 5]

In [104]: id(s)                                                        
Out[104]: 140462445434696

In [105]: id(a)                                                        
Out[105]: 140462410623560
```

**reverse**

查看`reverse`的使用方法

```python
In [83]: help(s.reverse)   
```

>reverse(...) method of builtins.list instance
>​    L.reverse() -- reverse *IN PLACE*
>
>使用方法：L.reverse()
>
>注意： 反向列表中元素。

**演示：**

```python
In [84]: s = [1,2,3,4,5]                                               

In [85]: s.reverse()                                                   

In [86]: s                                                             
Out[86]: [5, 4, 3, 2, 1]
```

**sort**

查看`sort`的使用方法

```python
In [87]: help(s.sort)  
```

> sort(...) method of builtins.list instance
> ​    L.sort(key=None, reverse=False) -> None -- stable sort *IN PLACE*
>
> 使用方法：L.sort() 
>
> 注意： 对原列表进行排序。列表中的元素要类型相同  (key = len int lambda)，不同元素需要改变元素类型，然后根据ASCII码进行排序。

**演示**：

```python
In [92]: s                                                             
Out[92]: [5, 4, 3, 2, 1]

In [93]: s.sort()                                                      

In [94]: s                                                             
Out[94]: [1, 2, 3, 4, 5]
    
In [110]: s = [1,2,3,4,5,'b','a']                                      

In [111]: s.sort()                                                     
-----------------------------------------------------------------------
TypeError                             Traceback (most recent call last)
<ipython-input-111-474c8408a842> in <module>
----> 1 s.sort()

TypeError: unorderable types: str() < int()

In [112]: s.sort(key=str)                                              

In [113]: s                                                            
Out[113]: [1, 2, 3, 4, 5, 'a', 'b']

```



#### 字符串常用方法

##### `增加元素`

**使用 +**

**演示：**

```python
In [215]: a = 'hello'                                                  

In [216]: b = 'python'                                                 

In [217]: c = '!'                                                      

In [218]: a+b+c                                                        
Out[218]: 'hellopython!'

In [219]: a+' '+b+' '+c                                                
Out[219]: 'hello python !'
```



**格式化字符串**

**演示：**

```python
In [220]: a = 'hello'                                                  

In [221]: b = 'python'                                                 

In [222]: c = '!'                                                      

In [223]: '%s %s %s'%(a,b,c)                                           
Out[223]: 'hello python !'
```



**使用join**

**演示：**

```python
In [229]: a = 'hello'                                                  

In [230]: b = 'python'                                                 

In [231]: c = '!'                                                      

In [232]: ' '.join([a,b,c])                                            
Out[232]: 'hello python !'

In [233]: '****'.join('abc')                                           
Out[233]: 'a****b****c'
```



**使用format**

**演示：**

```python
In [234]: a = 'hello'                                                  

In [235]: b = 'python'                                                 

In [236]: c = '!'                                                      

In [237]: '{} {} {}'.format(a,b,c)                                     
Out[237]: 'hello python !'

In [238]: '{0} {1} {2}'.format(a,b,c)                                  
Out[238]: 'hello python !'

In [239]: '{2} {1} {0}'.format(a,b,c)                                  
Out[239]: '! python hello'

In [240]: '{1} {1} {1}'.format(a,b,c)                                  
Out[240]: 'python python python'

In [241]: '{n1} {n2} {n3}'.format(n1=a, n2=b, n3=c)                    
Out[241]: 'hello python !'
```



##### `删除元素`

**replace**

查看`replace` 的使用方法

```python
In [115]: help(s.replace)     
```

> replace(...) method of builtins.str instance
> ​    S.replace(old, new[, count]) -> str
> ​    Return a copy of S with all occurrences of substring old replaced by new.  If the optional argument count is given, only the first count occurrences are replaced.
>
> 使用方法：s.replace (x,y) ：
>
> 注意： 子串替换,在字符串s中出现字符串x的任意位置都用y进行替换

**演示：**

```python
In [205]: s = 'abc cnn dnn'                                            

In [206]: s.replace('a','2')                                           
Out[206]: '2bc cnn dnn'

In [207]: s.replace('n','w',1)                                         
Out[207]: 'abc cwn dnn'

In [208]: s.replace('n','w',2)                                         
Out[208]: 'abc cww dnn'

In [209]: s.replace('n','w')                                           
Out[209]: 'abc cww dww'
```



##### `修改元素`

**upper**

查看`upper` 的使用方法

```python
In [116]: help(s.upper) 
```

> upper(...) method of builtins.str instance
> ​    S.upper() -> str
>
> Return a copy of S converted to uppercase.
>
> 使用方法： s.upper () 
>
> 注意： 将字符串转为大写 

**演示**：

```python
In [189]: s = '123456abc'                                              

In [190]: s.upper()                                                    
Out[190]: '123456ABC'
```



**lower**

查看`lower` 的使用方法

```python
In [116]: help(s.lower) 
```

> lower(...) method of builtins.str instance
> ​    S.lower() -> str
> ​    Return a copy of the string S converted to lowercase.
>
> 使用方法：s.lower () 
>
> 注意：将字符串转为小写

**演示：**

```python
In [193]: s = '123456abcDF'                                            

In [194]: s.lower()                                                    
Out[194]: '123456abcdf'
```



**strip(lstrip、rstrip)**

查看`strip(lstrip、rstrip)`的使用方法

```python
In [116]: help(s.strip) 
In [117]: help(s.lstrip) 
In [118]: help(s.rstrip) 
```

> strip(...) method of builtins.str instance
> ​    S.strip([chars]) -> str
>    Return a copy of the string S with leading and trailing whitespace removed. If chars is given and not None, remove characters in chars instead
>
> 使用方式：s.trip()
>
> 注意：去除两边的空格

> lstrip(...) method of builtins.str instance
> ​    S.lstrip([chars]) -> str    
>
> Return a copy of the string S with leading whitespace removed.  If chars is given and not None, remove characters in chars instead
>
> 使用方式：s.lstrip()
>
> 注意:去除左边的空格

> rstrip(...) method of builtins.str instance
> ​    S.rstrip([chars]) -> str
>    Return a copy of the string S with trailing whitespace removed.  If chars is given and not None, remove characters in chars instead
>
> 使用方式：s.rstrip()
>
> 注意：去除右边空格

**演示：**

```python
In [195]: s = '    abc    '                                            

In [196]: s.strip()                                                    
Out[196]: 'abc'

In [197]: s.lstrip()                                                   
Out[197]: 'abc    '

In [198]: s.rstrip()                                                   
Out[198]: '    abc'
```



**capitalize**

查看`capitalize`的使用方法

```python
In [115]: help(s.capitalize)     
```

> capitalize(...) method of builtins.str instance
> ​    S.capitalize() -> str
> ​    Return a capitalized version of S, i.e. make the first character  have upper case and the rest lower case.
>
> 使用方式： s.capitalize()
>
> 注意：首字母大写

**演示：**

```
In [201]: s = 'abc cnn dnn'                                            

In [202]: s.capitalize()                                               
Out[202]: 'Abc cnn dnn'

```



**title**

查看`title`的使用方法

```python
In [115]: help(s.title)     
```

> title(...) method of builtins.str instance
> ​    S.title() -> str
> ​    Return a titlecased version of S, i.e. words start with title case characters, all remaining cased characters have lower case.
>
> 使用方式： S.title() 
>
> 注意：每个单词的首字母大写

**演示：**

```python
In [203]: s = 'abc cnn dnn'                                            

In [204]: s.title()                                                    
Out[204]: 'Abc Cnn Dnn'
```



**split**

查看`split`的使用方法

```python
In [115]: help(s.split)     
```

> split(...) method of builtins.str instance
> ​    S.split(sep=None, maxsplit=-1) -> list of strings
> ​    Return a list of the words in S, using sep as the delimiter string.  If maxsplit is given, at most maxsplit splits are done. If sep is not specified or is None, any whitespace string is a separator and empty strings are removed from the result.
>
> 使用方法：s.split()，s.split(a,b)
>
> 注意：s.split()指的是返回一系列用空格分割的字符串列表
>
> ​		s.split(a,b)指的是a,b为可选参数，a是将要分割的字符串，b是说明最多要分割几个

**演示：**

```python
In [210]: s = 'abc123cba'                                              

In [211]: s.split('b')                                                 
Out[211]: ['a', 'c123c', 'a']

In [213]: s = 'abc cnn dnn'                                            

In [214]: s.split(' ')                                                 
Out[214]: ['abc', 'cnn', 'dnn']
```



##### `查找元素`

**count**

查看`count`的使用方法

```python
In [115]: help(s.count)     
```

> count(...) method of builtins.str instance
> ​    S.count(sub[, start[, end]]) -> int
> ​    Return the number of non-overlapping occurrences of substring sub in string S[start:end].  Optional arguments start and end are interpreted as in slice notation.
>
> 使用方法：s.count(x)
>
> 注意： 返回字符串x在s中出现的次数，带可选参数

**演示：**

```python
In [138]: s = 'abc123abc321678'                                        

In [139]: s.count('a')                                                 
Out[139]: 2

In [140]: s.count('8')                                                 
Out[140]: 1
```



**index**

查看`index`的使用方法

```python
In [115]: help(s.index)     
```

> index(...) method of builtins.str instance
> ​    S.index(sub[, start[, end]]) -> int
> ​    Like S.find() but raise ValueError when the substring is not found
>
> 使用方法：s.index(x)
>
> 注意：返回字符串中出现x的最左端的索引值，如果不在则抛出valueError异常

**演示**：

```python
In [141]: s = 'abc123abc321678'                                        

In [142]: s.index('a')                                                 
Out[142]: 0

In [143]: s.index('a',2)                                               
Out[143]: 6
```



**find**

查看`find`的使用方法

```python
In [115]: help(s.find)     
```

> find(...) method of builtins.str instance
> ​    S.find(sub[, start[, end]]) -> int
> ​    Return the lowest index in S where substring sub is found, such that sub is contained within S[start:end].Optional
> arguments start and end are interpreted as in slice notation.
>
> Return -1 on failure.
>
> 使用方法：s.find(x) 
>
> 注意：返回字符串中出现x的最左端字符的索引值，如果不在则返回-1

**演示**：

```python
In [146]: s = 'abc123abc321678'                                        

In [147]: s.index('a',1,4)  # index没有找到就报错                                            
-----------------------------------------------------------------------
ValueError                            Traceback (most recent call last)
<ipython-input-147-fbbfe36ae6a6> in <module>
----> 1 s.index('a',1,4)

ValueError: substring not found

In [148]: s.find('a',1,4)  # find 没有找到就提示-1                                             
Out[148]: -1

In [149]: s.find('a',1)                                                
Out[149]: 6

In [150]: s.find('a')                                                  
Out[150]: 0
```



**isdigit**

查看`isdigit`的使用方法

```python
In [115]: help(s.isdigit)     
```

> isdigit(...) method of builtins.str instance
> ​    S.isdigit() -> bool
> ​    Return True if all characters in S are digits and there is at least one character in S, False otherwise.
>
> 使用方法：s.isdigit ()
>
> 注意 ：测试是否全是数字，都是数字则返回 True 否则返回 False.

**演示**：

```python
In [151]: s = '123456'                                                 

In [152]: s1 = '123abc'                                                

In [153]: s2 = 'abcdef'                                                

In [154]: s.isdigit()                                                  
Out[154]: True

In [155]: s1.isdigit()                                                 
Out[155]: False

In [156]: s2.isdigit()                                                 
Out[156]: False
```



**isalpha **

查看`isalpha `的使用方法

```python
In [115]: help(s.isalpha )     
```

> isalpha(...) method of builtins.str instance
> ​    S.isalpha() -> bool
> ​    Return True if all characters in S are alphabetic and there is at least one character in S, False otherwise.
>
> 使用方法：s.isalpha () 
>
> 注意 ：测试是否全是字母，都是字母则返回 True,否则返回 False.

**演示：**

```python
In [157]: s = '123456'                                                 

In [158]: s1 = '123abc'                                                

In [159]: s2 = 'abcdef'                                                

In [160]: s.isalpha()                                                  
Out[160]: False

In [161]: s1.isalpha()                                                 
Out[161]: False

In [162]: s2.isalpha()                                                 
Out[162]: True
```



**endswith**

查看`endswith `的使用方法

```python
In [115]: help(s.endswith )     
```

> endswith(...) method of builtins.str instance
> ​    S.endswith(suffix[, start[, end]]) -> bool
> ​    Return True if S ends with the specified suffix, False otherwise. With optional start, test S beginning at that position. With optional end, stop comparing S at that position. suffix can also be a tuple of strings to try.
>
> 使用方法：s.endswith(x)
>
> 注意：如果字符串s以x结尾，返回True

**演示：**

```python
In [163]: s = '123456abc'                                              

In [164]: s1 = '123abced'                                              

In [165]: s2 = 'abcdef'                                                

In [166]: s.endswith('c')                                              
Out[166]: True

In [167]: s.endswith('2')                                              
Out[167]: False

In [168]: s1.endswith('d')                                             
Out[168]: True

In [169]: s2.endswith('f')                                             
Out[169]: True
```

> 备注：方法s.startwith与s.endwith相反，前者是以什么开始，后者是以什么结尾。



**islower**

查看`islower `的使用方法

```python
In [115]: help(s.islower )     
```

> islower(...) method of builtins.str instance
> ​    S.islower() -> bool
> ​    Return True if all cased characters in S are lowercase and there is at least one cased character in S, False otherwise.
>
> 使用方法：s.islower () 
>
> 注意：测试是否全是小写

**演示：**

```python
In [173]: s = '123456abc'                                              

In [174]: s1 = 'abcDE'                                                 

In [175]: s2 = 'DE'                                                    

In [176]: s.islower()                                                  
Out[176]: True

In [177]: s1.islower()                                                 
Out[177]: False

In [178]: s2.islower()                                                 
Out[178]: False
```



**isupper**

查看`isupper `的使用方法

```python
In [115]: help(s.isupper )     
```

> isupper(...) method of builtins.str instance
> ​    S.isupper() -> bool
> ​    Return True if all cased characters in S are uppercase and there is at least one cased character in S, False otherwise.
>
> 使用方式：s.isupper () 
>
> 注意：测试是否全是大写

**演示：**

```python
In [181]: s = '123456abc'                                              

In [182]: s1 = 'abcDE'                                                 

In [183]: s2 = 'DE'                                                    

In [184]: s3 = '123ADFAFA'                                             

In [185]: s.isupper()                                                  
Out[185]: False

In [186]: s1.isupper()                                                 
Out[186]: False

In [187]: s2.isupper()                                                 
Out[187]: True

In [188]: s3.isupper()                                                 
Out[188]: True
```



#### 元组常用方法

> 元组为不可变序列，所以只有两种方法。

##### `查找元素`

**count**

查看`count `的使用方法

```python
In [115]: help(s.count )     
```

> count(...) method of builtins.tuple instance
> ​    T.count(value) -> integer -- return number of occurrences of value
>
> 使用方法：s.count(value)
>
> 注意：统计元素的个数

**演示：**

```python
In [243]: s = (1,2,3,4,3,2,1)                                          

In [244]: s.count(1)                                                   
Out[244]: 2

In [245]: s.count(4)                                                   
Out[245]: 1
```



**index**

查看`index `的使用方法

```python
In [115]: help(s.index )     
```

> index(...) method of builtins.tuple instance
> ​    T.index(value, [start, [stop]]) -> integer -- return first index of value.
> ​    Raises ValueError if the value is not present.
>
> 使用方式：  T.index(value, [start, [stop]])
>
> 注意：查看元素的索引

```python
In [246]: s = (1,2,3,4,3,2,1)                                          

In [247]: s.index(1)                                                   
Out[247]: 0

In [248]: s.index(1,1)                                                 
Out[248]: 6

In [249]: s.index(4)                                                   
Out[249]: 3
```


转载请注明：[Seven的博客](http://sevenold.github.io)
