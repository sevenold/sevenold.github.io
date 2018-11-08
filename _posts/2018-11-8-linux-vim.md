---
layout: post
title: "Linux-vim及python虚拟环境"
date: 2018-11-8
description: "linux简介、基本操作命令、vim、python虚拟环境"
tag: LINUX
---

## linux常用命令

### **添加或删除用户**：

#### `adduser`: 

```shell
adduser username # 添加一个叫username的用户
```

#### `userdel`:  

```shell
userdel  username  # 删除用户  不删错home目录
userdel -r  username # 删除用户 同时删除用户home目录
```

### 文件压缩解压

#### `.tar格式`

在`Linux`中打包就是使用`tar`命令    命令有点特殊 前面的参数  可加可不加 ` - `

```
# 打包
 tar -cvf  文件名.tar   # 要打包的文件

# 解包
 tar  -xvf  文件名.tar  

#查看包里的内容
tar -tvf 包的文件名.tar
```

#### `.gz格式`

或者说配合 `tar` 则是

```shell
tar -zcvf  xxx.tar.gz  文件 # 压缩
tar -zxvf  xxx.tar.gz 文件  # 解压
# 解压到指定目录 
tar -zxvf xxx.tar.gz -C dirname
```

#### `.bz2格式 `

```shell
tar -jcvf  xxx.tar.bz2  文件  # 压缩
tar -jxvf  xxx.tar.bz2       # 解压
```

#### `.zip格式`

通过`zip` 压缩的话 不需要指定拓展名  默认拓展名为 `zip`

```shell
  sudo apt-get install zip
```

```
压缩文件
	zip  压缩文件  源文件

解压 
	unzip  压缩文件
-d 解压到指定目录   如果目录不存在 会自动创建新目录 并压缩进去
unzip test.zip -d filename
```



### 进程

### `ps`   aux 查看进程

查看所有进程 ： `ps aux` 

### `top` 动态查看进程

查看所有进程 ： 

```shell
top
```

top命令用来动态显示运行中的进程     top命令能够在运行后     在指定的时间间隔更新显示信息     可以在使用top命令时加上-d 来指定显示信息更新的时间间隔

```shell
jobs # 查看所有在后台的进程
```

`ctrl＋Z` 　 暂停当前进程　比如你正运行一个命令　突然觉得有点问题想暂停一下　就可以使用这个快捷键　暂停后　可以使用`fg` 恢复它

###`kill`      杀死一个进程

```shell
kill -9  id号  强制杀死
```

```shell
ps -aux 
	查看正在内存中的程序 
会配合 管道符 
ps aux | grep ssh/python 

#### top动态查看
默认3秒  
-d 时间 

#### 结束进程

kill -9  id号  强制杀死

ctrl+z 可以让你正在运行的东西 暂停 

jobs 查看所有在后台的进程
```

#### ctrl  z   暂停 。   fg   继续



### vim **编辑器**

工作模式：命令模式、输入模式、末行模式

#### 模式之间切换

```shell
当打开一个文件时处于命令模式
在命令模式下，按 i 进入输入模式
在输入模式，按ESC回到命令模式。
在命令模式下，按shift+; ，末行出现:冒号，则进入末行模式
```

#### 进入与退出

```shell
进入
    vim   filename
退出
    :wq    末行模式，wq 保存退出
    :q       末行模式，q 直接退出
    :q!      末行模式，q! 强制退出，不保存
```

#### **输入模式**

```shell
输入模式

    i    从光标所在位置前面开始插入
    I    在当前行首插入
    a   从光标所在位置后面开始输入
    A   在当前行尾插入
    o   在光标所在行下方新增一行并进入输入模式
    O  在当前上面一行插入

进入输入模式后，在最后一行会出现--INSERT—的字样
```

#### **移动光标**

```shell
移动光标
    gg    到文件第一行
    G      到文件最后一行   (Shift + g)
    ^      非空格行首
    0       行首(数字0)
    $       行尾
```

#### **复制和粘贴**

```shell
复制和粘贴
    yy    复制整行内容
    3yy  复制3行内容
    yw   复制当前光标到单词尾内容

    p      粘贴

```

#### **删除**

```shell
删除
    dd  删除光标所在行
    dw  删除一个单词
    x     删除光标所在字符
    u    撤销上一次操作
    s     替换
    ctrl + r    撤销   u

```

#### **块操作**

```shell
块操作
    v  + 方向键进行块选择
    ctrl + v +方向键进行列块选择
```

#### **查找**

```shell
查找
    /    命令模式下输入：/   向前搜索     不能空格
    ?    命令模式下输入：?   向后搜索
# / 方式
    n    向下查找
    N   向上查找
# ? 方式
    n    向上查找
    N   向下查找
```

#### 注意

```shell
在vim下千万不要按Ctrl + s 。 不然就会卡死
如果按了Ctrl + s，按Ctrl + q 可以退出
```

#### **实例**

```shell
vim hello.py # 创建一个hello.py的python文件
```

filename：hello.py

```python
print('hello word') # 打印hello word
```

保存，退出

```shell
seven@seven:~$ python hello.py 
hello word
```

### python**虚拟环境的使用**

#### **安装虚拟环境**

```shell
$ sudo pip install virtualenv
```

#### **创建虚拟环境**

```shell
$ virtualenv envname # envname 自定义的虚拟环境名字
```

#### **激活虚拟环境**

```shell
$ source envname/bin/activate 
```

#### **退出虚拟环境**

```shell
(envname)$ deactivate
```

#### **删除虚拟环境**

```shell
$ sudo rm -rf envname # 只需删除建立的文件夹就行
```

