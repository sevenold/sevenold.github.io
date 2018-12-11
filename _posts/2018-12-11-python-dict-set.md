---
layout: post
title: "python第四话之散列类型、运算优先级和逻辑运算"
date: 2018-12-11
description: "python散列类型、运算优先级和逻辑运算，字典，集合的增删改查"
tag: python
---

[TOC]

### 散列类型、运算优先级和逻辑运算

`散列类型`也就是我们所熟知的`字典`和`集合`，我们今天来看看散列类型的相关逻辑运算。

### 集合（set）

#### 集合的特点：

无序、元素是唯一的。

#### 集合的创建：

用大括号“{}”，各元素之间用逗号隔开；也可以通过类型转换的方式使用set()内置函数将列表或元祖转换为集合类型。在创建的过程中会自动过滤掉重复的元素，保证元素的唯一性。

#### 演示

```python
In [1]: s = [1,2,3,4,1,2]                                              

In [2]: s                                                              
Out[2]: [1, 2, 3, 4, 1, 2]

In [3]: se = set(s)                                                    

In [4]: se                                                             
Out[4]: {1, 2, 3, 4}

In [5]: type(se)                                                       
Out[5]: set
    
In [6]: {1,2,3,1,2,3}                                                  
Out[6]: {1, 2, 3}

```

> 注意：列表是允许元素重复的，但是当我们把列表转成集合后，里面重复的元素就去掉了。

#### 集合的运算

交集：&

并集：\| 

差集：-

![](http://eveseven.oss-cn-shanghai.aliyuncs.com/18-12-11/79094336.jpg)

##### `交集`

两个集合(s 和t)的差补或相对补集是指一个集合C，该集合中的元素，只属于集合s，而不属于集合t。

```python
In [7]: s1 = {1, 2, 3, 4, 'a', 'b'}                                    

In [8]: s2 = {0,2,3,'b','c'}                                           

In [9]: s1 & s2                                                        
Out[9]: {2, 3, 'b'}
```

> 两个集合取交集，最后输出的元素是属于两个集合所共有的元素。

##### `并集`

联合(union)操作和集合的OR(又称可兼析取(inclusive disjunction))其实是等价的，两个集合的联合是一个新集合，该集合中的每个元素都至少是其中一个集合的成员，即：属于两个集合其中之一的成员。

```python
In [10]: s1 = {1, 2, 3, 4, 'a', 'b'}                                   

In [11]: s2 = {0,2,3,'b','c'}                                          

In [12]: s1 | s2                                                       
Out[12]: {0, 1, 2, 3, 4, 'a', 'b', 'c'}
```

> 并集指的是两个集合的元素进行一个整合，最后生成的元素都是属于原来两个集合之中的某一个。

##### `差集`

和其他的布尔集合操作相似，对称差分是集合的XOR(又称"异 或" (exclusive disjunction)).两个集合(s 和t)的对称差分是指另外一个集合C,该集合中的元素，只能是属于集合s 或者集合t的成员，不能同时属于两个集合。

```python
In [13]: s1 = {1, 2, 3, 4, 'a', 'b'}                                   

In [14]: s2 = {0,2,3,'b','c'}                                          

In [15]: s1 - s2                                                       
Out[15]: {1, 4, 'a'}
```

> 差集也叫被减集合的补集。

##### `扩展`

**add**

```python
In [16]: s1 = {1, 2, 3, 4, 'a', 'b'}                                   

In [17]: s1.add(8)                                                     

In [18]: s1                                                            
Out[18]: {1, 2, 3, 4, 8, 'a', 'b'}

In [19]: s1.add('w')                                                   

In [20]: s1                                                            
Out[20]: {1, 2, 3, 4, 8, 'a', 'b', 'w'}
```

> 往集合里添加元素。

**pop**

```python
In [23]: s1                                                            
Out[23]: {0, 1, 2, 3, 4, 8, 'a', 'b', 'w'}

In [24]: s1.pop()                                                      
Out[24]: 0

In [25]: s1.pop()                                                      
Out[25]: 1

In [26]: s1.pop()                                                      
Out[26]: 2

In [27]: s1.pop()                                                      
Out[27]: 3

In [28]: s1.pop()                                                      
Out[28]: 4
```

> pop方法是没有参数的，因为集合是无序的，所以在移除的时候是随机移除的。

**remove**

```python
In [30]: s1 = {1, 2, 3, 4, 'a', 'b'}                                   

In [31]: s1.remove(1)                                                  

In [32]: s1                                                            
Out[32]: {2, 3, 4, 'a', 'b'}

In [33]: s1.remove('a')                                                

In [34]: s1                                                            
Out[34]: {2, 3, 4, 'b'}
```

> remove方法是指定元素进行删除。

**update**

```python
In [35]: s1                                                            
Out[35]: {2, 3, 4, 'b'}

In [36]: s1.update({'w', 'c'})                                         

In [37]: s1                                                            
Out[37]: {2, 3, 4, 'b', 'c', 'w'}
```

> update方法是往集合里面添加集合。

**isdisjoint**

```python
In [38]: s1 = {1, 2, 3, 4, 'a', 'b'}                                   

In [39]: s2 = {0,2,3,'b','c'}                                          

In [40]: s1.isdisjoint(s2)                                             
Out[40]: False

In [41]: s1.isdisjoint({6, 8, 7})                                      
Out[41]: True
```

> isdisjoint方法是判断两个集合有没有交集，有返回False，没有则返回True

**issubset**

```python
In [42]: s1 = {1, 2, 3, 4, 'a', 'b'}                                   

In [43]: s2 = {0,2,3,'b','c'}                                          

In [44]: s1.issubset(s2)                                               
Out[44]: False

In [45]: s2.issubset(s1)                                               
Out[45]: False

In [46]: s1.issubset(s1)                                               
Out[46]: True

In [47]: {1, 2}.issubset(s1)                                           
Out[47]: True

In [48]: s1                                                            
Out[48]: {1, 2, 3, 4, 'a', 'b'}
```

> 判断前面的集合是不是后面的集合的子集。

**issuperset**

```python
In [51]: s1 = {1, 2, 3, 4, 'a', 'b'}                                   

In [52]: s2 = {2,3,'b'}                                                

In [53]: s2.issubset(s1)                                               
Out[53]: True

In [54]: s1.issuperset(s2)                                             
Out[54]: True
```

> 判断后面的集合是前面集合的子集。

#### 总结

> - 集合唯一性：集合中的元素具有唯一性，不存在两个相同的元素。
> - 集合可变性：集合中的元素是可变的，集合是可变对象。
> - 集合无序性：集合中的元素是无序的，所以没有存在索引。

### 字典（dict）

字典是除了列表外的另一种`可变类型`，字典的元素是以键值对的形式存在，字典的键必须是唯一，可以是数字、字符串或者是元组，键可以为任何不可变类型，列表和集合不能作为字典的键。

#### 字典的创建

第一种 { key :value } ，字典里的键和值用“：”隔开，一对键和值组成一个项，项和项之间用“，”隔开。

第二种使用内置函数dict(key=value)，要注意的是这里使用的是“=”赋值的方式，键是以名字的形式所以这种方法的键就必须符合名字的要求，且不能使用关键字作为键。

如果你要使用关键字作为键名那么就只能用第一种方法，关键字以字符串的形式来创建。

通过字典的键可以访问这个键所对应的值，字典是可变类型，所以可以直接对字典的项进行修改，使用dictname[key] = value，如果这个键存在于字典中，则是修改这个键所对应的值，如果这个键不存在则是往字典中添加这个项。

#### **演示**

```python
In [56]: {'a':1, 'b':2}                                                
Out[56]: {'a': 1, 'b': 2}

In [57]: s = {'a':1, 'b':2}                                            

In [58]: s,type(s)                                                     
Out[58]: ({'a': 1, 'b': 2}, dict)

In [56]: {'a':1, 'b':2}                                                
Out[56]: {'a': 1, 'b': 2}

In [57]: s = {'a':1, 'b':2}                                            

In [58]: s,type(s)                                                     
Out[58]: ({'a': 1, 'b': 2}, dict)
```

> 字典形式：{key:value}

#### 字典的运用

##### `查看`

```python
In [67]: a = dict(a=1, b=2)                                            

In [68]: a                                                             
Out[68]: {'a': 1, 'b': 2}

In [69]: a['a']                                                        
Out[69]: 1
```

> 由于字典也是无序的，所以我们在取值的时候，是根据key来取出对应的value的。

##### `增加`

```python
In [70]: a                                                             
Out[70]: {'a': 1, 'b': 2}

In [71]: a['c'] = 3                                                    

In [72]: a                                                             
Out[72]: {'a': 1, 'b': 2, 'c': 3}
```

> 往字典里添加元素时，是key和value对应增加的。

##### `修改`

```python
In [73]: a                                                             
Out[73]: {'a': 1, 'b': 2, 'c': 3}

In [74]: a['a'] = 'w'                                                  

In [75]: a                                                             
Out[75]: {'a': 'w', 'b': 2, 'c': 3}
```

> 修改字典是通过key取出value，然后对应的去重新赋值。

#### 字典的增删改查

##### 增加

`copy`

```python
In [76]: a                                                             
Out[76]: {'a': 'w', 'b': 2, 'c': 3}

In [77]: b = a.copy()                                                  

In [78]: b                                                             
Out[78]: {'a': 'w', 'b': 2, 'c': 3}

```

> 复制成一个新字典。

`fromkeys`

查看fromkeys的使用方法

```python
In [79]: help(a.fromkeys)                                              
```

> fromkeys(iterable, value=None, /) method of builtins.type instance  Returns a new dict with keys from iterable and values equal to value.
>
> 注意： 返回一个新的dict，其中包含来自iterable的键，值等于value。

```python
In [93]: a                                                             
Out[93]: {'a': 1, 'b': 2}

In [94]: s = a.fromkeys(['c', 'd'])                                    

In [95]: s                                                             
Out[95]: {'c': None, 'd': None}

In [96]: s = a.fromkeys(['c', 'd'], 7)                                 

In [97]: s                                                             
Out[97]: {'c': 7, 'd': 7}

```

> 使用fromkey方法的时候，原字典是不变的，会返回一个新的字典。

`setfefault`

```python
In [103]: a                                                            
Out[103]: {'a': 'w', 'b': 2}

In [104]: a.setdefault('a')                                            
Out[104]: 'w'

In [105]: a.setdefault('b')                                            
Out[105]: 2

In [106]: a.setdefault('c')                                            

In [107]: a                                                            
Out[107]: {'a': 'w', 'b': 2, 'c': None}

In [108]: a.setdefault('d', 4)                                         
Out[108]: 4

In [109]: a                                                            
Out[109]: {'a': 'w', 'b': 2, 'c': None, 'd': 4}
```

> 查询并返回key所对应的值，如果没有这个key,则会新建。有则查，无则增。

##### 删除

`clear`

```python
In [110]: a                                                            
Out[110]: {'a': 'w', 'b': 2, 'c': None, 'd': 4}

In [111]: a.clear()                                                    

In [112]: a                                                            
Out[112]: {}
```

> 删除所有键值对

`pop`

查看pop 的方法

```python
In [121]: help(a.pop) 
```

> pop(...) method of builtins.dict instance
> ​    D.pop(k[,d]) -> v, remove specified key and return the corresponding value.
> ​    If key is not found, d is returned if given, otherwise KeyError is raised

```python
In [116]: a                                                            
Out[116]: {'a': 1, 'b': 2, 'd': 4}

In [117]: a.pop('a')                                                   
Out[117]: 1

In [118]: a                                                            
Out[118]: {'b': 2, 'd': 4}

In [119]: a.pop('d')                                                   
Out[119]: 4

In [120]: a.pop()                                                      
-----------------------------------------------------------------------
TypeError                             Traceback (most recent call last)
<ipython-input-120-9c070c907602> in <module>
----> 1 a.pop()

TypeError: pop expected at least 1 arguments, got 0
```

> pop方法是删除指定的键并返回相应的值。

```python
In [134]: a                                                            
Out[134]: {'a': 1, 'b': 2, 'd': 4}

In [135]: a.pop('c', 'b')                                              
Out[135]: 'b'

In [136]: a.pop('a', 'b')                                              
Out[136]: 1

In [137]: a                                                            
Out[137]: {'b': 2, 'd': 4}
```

> 如果传入两个值，第一个是key，第二个是一个值，如果找到key, 就删除对应键值对，并返回该值，如果没有找到key,就返回你所传入的第二个值。

`popitem`

```python
In [140]: a                                                            
Out[140]: {'b': 2, 'd': 4, 's': 6}

In [141]: a.popitem()                                                  
Out[141]: ('s', 6)

In [142]: a.popitem()                                                  
Out[142]: ('d', 4)

In [143]: a                                                            
Out[143]: {'b': 2}
```

> 由于字典也是无序的，多以popitem是随机删除一个键值对。

##### 修改

`update`

```python
In [145]: a                                                            
Out[145]: {'b': 2}

In [146]: a.update({'a':1, 'c':4, 's': 9})                             

In [147]: a                                                            
Out[147]: {'a': 1, 'b': 2, 'c': 4, 's': 9}

In [148]: a.update({'a':0})                                            

In [149]: a                                                            
Out[149]: {'a': 0, 'b': 2, 'c': 4, 's': 9}
```

> update方法，对于键值对的处理是，有则改，无则增。

##### 查询

`get`

查看get的使用方法

```python
In [155]: help(a.get)  
```

> get(...) method of builtins.dict instance
> ​    D.get(k[,d]) -> D[k] if k in D, else d.  d defaults to None.
>
> 注意：默认返回None

```python
In [151]: a                                                            
Out[151]: {'a': 0, 'b': 2, 'c': 4, 's': 9}

In [152]: a.get('a')                                                   
Out[152]: 0

In [153]: a.get('f')                                                   

In [154]: a.get('f', "没有")                                           
Out[154]: '没有'
```

> get方法是如果查询到key就返回对应的value，如果没有，就返回你给定的提示值。

```python
In [156]: c,d = a.get('f', (2,3))                                      

In [157]: c                                                            
Out[157]: 2

In [158]: d                                                            
Out[158]: 3

```

> 也可以通过这个功能，做些操作。

`keys`

```python
In [159]: a                                                            
Out[159]: {'a': 0, 'b': 2, 'c': 4, 's': 9}

In [160]: a.keys()                                                     
Out[160]: dict_keys(['s', 'a', 'c', 'b'])

In [161]: list(a.keys())                                               
Out[161]: ['s', 'a', 'c', 'b']

```

> 获取字典里所有的key。



`value`

```python
In [162]: a                                                            
Out[162]: {'a': 0, 'b': 2, 'c': 4, 's': 9}

In [163]: a.values()                                                   
Out[163]: dict_values([9, 0, 4, 2])

In [164]: list(a.values())                                             
Out[164]: [9, 0, 4, 2]

```

> 获取所有的value。

`items`

```python
In [165]: a                                                            
Out[165]: {'a': 0, 'b': 2, 'c': 4, 's': 9}

In [166]: a.items()                                                    
Out[166]: dict_items([('s', 9), ('a', 0), ('c', 4), ('b', 2)])

In [167]: list(a.items())                                              
Out[167]: [('s', 9), ('a', 0), ('c', 4), ('b', 2)]
```

> 获取所有的键值对。

#### 总结

> - 键（key）唯一性： 字典中的键（key）具有唯一性，不存在两个相同的键（key）
> - 可变性： 字典是可变对象，但是自动减的键（key）必须是不可变对象
> - 无序性：字典中的键也是无序的，所以不能通过索引取值。

### 运算符及优先级

#### Python中的运算符

![1](http://eveseven.oss-cn-shanghai.aliyuncs.com/18-12-11/41589190.jpg)



#### 演示：

```python
In [168]: 2 **3                                                        
Out[168]: 8

In [169]: 2+2                                                          
Out[169]: 4

In [170]: 2-1                                                          
Out[170]: 1

In [171]: 2<2                                                          
Out[171]: False

In [172]: 2>20                                                         
Out[172]: False

In [173]: 3<=(1+2)                                                     
Out[173]: True

In [174]: 5>=1                                                         
Out[174]: True

In [175]: 2==2                                                         
Out[175]: True

In [176]: 2!=2                                                         
Out[176]: False

In [177]: a =1 
In [181]: 8 %2                                                         
Out[181]: 0

In [182]: a                                                            
Out[182]: 1

In [183]: a += 1                                                       

In [184]: a                                                            
Out[184]: 2

In [185]: a /= 1.2                                                     

In [186]: a                                                            
Out[186]: 1.6666666666666667

In [187]: a %= 1                                                       

In [188]: a                                                            
Out[188]: 0.6666666666666667
    
In [189]: a = 1                                                        

In [190]: b = a                                                        

In [191]: a is b  # 判断是否是id一致                                                      
Out[191]: True
    
In [192]: 1 in [1, 2]                                                  
Out[192]: True

In [193]: 1 not in [1, 2]                                              
Out[193]: False

```

#### 逻辑运算符

##### 查看对象类型

`type`

```python
In [204]: a = 1                                                        

In [205]: b = 's'                                                      

In [206]: c = [1, 2]                                                   

In [207]: type(a), type(b), type(c)                                    
Out[207]: (int, str, list)

```

> 直接返回对象的类型

`isinstance`

```python
In [208]: a                                                            
Out[208]: 1

In [209]: b                                                            
Out[209]: 's'

In [210]: c                                                            
Out[210]: [1, 2]

In [211]: isinstance(a, int)                                           
Out[211]: True

In [212]: isinstance(a, str)                                           
Out[212]: False

In [213]: isinstance(b, str)                                           
Out[213]: True
```

> 判断对象的类型

##### 比较运算符

```python
In [216]: a = 1                                                        

In [217]: b = 2                                                        

In [218]: c = 1                                                        

In [219]: a == b                                                       
Out[219]: False

In [220]: a == c                                                       
Out[220]: True

In [221]: b != c                                                       
Out[221]: True
```

##### 如果有多个条件

> - 判断语句1    and    判断语句2
> - 判断语句1    or       判断语句2
> - not    判断语句1

```python
In [227]: a==b and b!=c                                                
Out[227]: False

In [228]: a==b or b!=c                                                 
Out[228]: True

In [229]: not a==b                                                     
Out[229]: True
```

转载请注明：[Seven的博客](http://sevenold.github.io)