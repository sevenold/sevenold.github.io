---
layout: post
title: "python第一步helloWord"
date: 2015-08-25 
description: "python第一步"
tag: python
---   

## Python--HelloWord程序

在开始第一个程序之前,我们先来讲一下什么是命令行模式和Python交互模式.

\--------------------------------------------------------------------------------------------------------------------------

### 命令行模式

在linux开始菜单选择“终端”，就进入到命令行模式，它的提示符类似name@name~>：

[![img](https://3.bp.blogspot.com/-oI_puv53EU0/WidN9_NMnpI/AAAAAAAAADc/1eVqJbiPwncE5HBpQLO_MJ6UFZQ5LM9xACLcBGAs/s640/1.png)](https://3.bp.blogspot.com/-oI_puv53EU0/WidN9_NMnpI/AAAAAAAAADc/1eVqJbiPwncE5HBpQLO_MJ6UFZQ5LM9xACLcBGAs/s1600/1.png)

```

```

###  Python交互模式

在命令行模式下敲命令python，就看到类似如下的一堆文本输出，然后就进入到Python交互模式，它的提示符是>>>。

[![img](https://3.bp.blogspot.com/-Tc9I0YT7VY0/WidOWO-fP7I/AAAAAAAAADg/7xBep9PdfcYFAvzkIlwKfVm2_gjaDC_6gCLcBGAs/s640/2.png)](https://3.bp.blogspot.com/-Tc9I0YT7VY0/WidOWO-fP7I/AAAAAAAAADg/7xBep9PdfcYFAvzkIlwKfVm2_gjaDC_6gCLcBGAs/s1600/2.png)

```

```

在Python交互模式下输入exit()并回车，就退出了Python交互模式，并回到命令行模式：

[![img](https://3.bp.blogspot.com/-OLVJg41UYQo/WidOWA3rm0I/AAAAAAAAADo/xM1diPKEeOQCZ5LNZZhm4Zfl0lu7oVA-wCEwYBhgL/s640/3.png)](https://3.bp.blogspot.com/-OLVJg41UYQo/WidOWA3rm0I/AAAAAAAAADo/xM1diPKEeOQCZ5LNZZhm4Zfl0lu7oVA-wCEwYBhgL/s1600/3.png)

```

```

在windows也可以直接通过开始菜单选择Python (command line)菜单项，直接进入Python交互模式，但是输入exit()后窗口会直接关闭，不会回到命令行模式。

\--------------------------------------------------------------------------------------------------------------------------

在写代码之前，请千万不要用“复制”-“粘贴”把代码从页面粘贴到你自己的电脑上。写程序也讲究一个感觉，你需要一个字母一个字母地把代码自己敲进去，在敲代码的过程中，初学者经常会敲错代码：拼写不对，大小写不对，混用中英文标点，混用空格和Tab键，所以，你需要仔细地检查、对照，才能以最快的速度掌握如何写程序。

\--------------------------------------------------------------------------------------------------------------------------

在Python的交互模式>>>下,直接输入代码,按回车,就可以立刻得到代码执行结果.现在,试试输入10+20,看看计算结果是不是30.

\--------------------------------------------------------------------------------------------------------------------------

\>>> 10+20

30

\--------------------------------------------------------------------------------------------------------------------------

其实,任何有效的数学计算都可以计算出来.

如果要让Python打印出指定的文字，可以用print()函数，然后把希望打印的文字用单引号或者双引号括起来，但不能混用单引号和双引号：

\--------------------------------------------------------------------------------------------------------------------------

\>>> print('hello, world') 

hello, world

\--------------------------------------------------------------------------------------------------------------------------

这种用单引号或者双引号括起来的文本在程序中叫字符串，今后我们还会经常遇到。

最后，用exit()退出Python，我们的第一个Python程序完成！

\--------------------------------------------------------------------------------------------------------------------------

### 命令行模式和Python交互模式

**注意:区别命令行模式和Python交互模式**

在命令行模式下，可以执行python进入Python交互式环境，也可以执行python hello.py运行一个.py文件。

执行一个.py文件只能在命令行模式执行。如果敲一个命令python hello.py，看到如下错误：

[![img](https://4.bp.blogspot.com/-H8epy96icZU/WidfCc8nCdI/AAAAAAAAAEA/rKJgt3gHPzI05_YkjWiMrBrwI5ut73EIQCLcBGAs/s640/4.png)](https://4.bp.blogspot.com/-H8epy96icZU/WidfCc8nCdI/AAAAAAAAAEA/rKJgt3gHPzI05_YkjWiMrBrwI5ut73EIQCLcBGAs/s1600/4.png)

错误提示No such file or directory说明这个hello.py在当前目录找不到，必须先把当前目录切换到hello.py所在的目录下，才能正常执行：

[![img](https://4.bp.blogspot.com/-5sua1HXQrok/Widf7ZlK34I/AAAAAAAAAEI/1bBS8JSjdh8vQNORPhCWBOdHeaTDxUEaQCLcBGAs/s640/5.png)](https://4.bp.blogspot.com/-5sua1HXQrok/Widf7ZlK34I/AAAAAAAAAEI/1bBS8JSjdh8vQNORPhCWBOdHeaTDxUEaQCLcBGAs/s1600/5.png)

此外，在命令行模式运行.py文件和在Python交互式环境下直接运行Python代码有所不同。Python交互式环境会把每一行Python代码的结果自动打印出来，但是，直接运行Python代码却不会。

例如，在Python交互式环境下，输入：

\--------------------------------------------------------------------------------------------------------------------------

\>>> 10+20

30

\--------------------------------------------------------------------------------------------------------------------------

直接可以看到结果30.

但是,写一个calc.py的文件,内容如下:

\--------------------------------------------------------------------------------------------------------------------------

print (10+20)

\--------------------------------------------------------------------------------------------------------------------------

[![img](https://3.bp.blogspot.com/-EAyJOUDjreY/WidhKdnAUsI/AAAAAAAAAEU/tz0gLzfOmFUnB9I38PU3NjRpBgEzqq1QACLcBGAs/s640/6.png)](https://3.bp.blogspot.com/-EAyJOUDjreY/WidhKdnAUsI/AAAAAAAAAEU/tz0gLzfOmFUnB9I38PU3NjRpBgEzqq1QACLcBGAs/s1600/6.png)

最后，Python交互模式的代码是输入一行，执行一行，而命令行模式下直接运行.py文件是一次性执行该文件内的所有代码。可见，Python交互模式主要是为了调试Python代码用的，也便于初学者学习，它不是正式运行Python代码的环境！

### 小结

在今后的学习中,Python交互模式是用来验证一些简单的代码,可以直接输入代码,然后执行,并立刻得到结果.

在命令行模式,通常是直接运行.py程序,来验证我们的代码.

转载请注明：[Seven的博客](http://seven.github.io) » [点击阅读原文](https://sevenold.github.io/2016/06/Develop_Tool/)
