---
layout: post
title: "TensorFlow从入门到实践"
date: 2018-08-13
description: "TensorFlow使用，TensorFlow简单教程"
tag: TensorFlow
---



### TensorFlow 从入门到实践

导入TensorFlow模块，查看下当前模块的版本

```python
import tensorflow as tf

print(tf.__version__)
```

```
1.4.0
```

tensorflow是一种图计算框架，所有的计算操作被声明为图（graph）中的节点（Node）

即使只是声明一个变量或者常量，也并不执行实际的操作，而是向图中增加节点

我们来看一上前面安装TensorFlow的时候的测试代码

```python
# 创建名为hello_constant的TensorFlow对象
hello_constant = tf.constant('Hello World!')

with tf.Session() as sess:
    
    # 在session中运行tf.constant 操作
    output = sess.run(hello_constant)
    
    print(output)
```

```
b'Hello World!'
```

### Tensor

#### 在 TensorFlow 中，数据不是以整数，浮点数或者字符串形式存在的。

#### 这些值被封装在一个叫做 tensor 的对象中。

在 hello_constant = tf.constant('Hello World!') 代码中，
hello_constant是一个 0 维度的字符串 tensor。

#### tensors 还有很多不同大小：

```python
# A 是一个0维的 int32 tensor
A = tf.constant(1234) 
# B 是一个1维的  int32 tensor
B = tf.constant([123,456,789]) 
 # C 是一个2维的  int32 tensor
C = tf.constant([ [123,456,789], [222,333,444] ])

with tf.Session() as sess:
    out_A = sess.run(A)
    out_B = sess.run(B)
    out_C = sess.run(C)
    
    print('\n', out_A, '\n', '----------', '\n', out_B, '\n', '----------', '\n', out_C)
    
```

```
 1234 
 ---------- 
 [123 456 789] 
 ---------- 
 [[123 456 789]
 [222 333 444]]
```

### Session

TensorFlow 的 api 构建在 computational graph（计算图) 的概念上，它是一种对数学运算过程进行可视化的一种方法。让我们把你刚才运行的 TensorFlow 的代码变成一个图：

![44.png](/images/dl/44.png)

如上图所示，一个 "TensorFlow Session" 是用来运行图的环境。这个 session 负责分配 GPU(s) 和／或 CPU(s) 包括远程计算机的运算。让我们看看怎么使用它：



```python
with tf.Session() as sess:
    output = sess.run(hello_constant)
```

这段代码已经从上一行创建了一个 tensor hello_constant。下一行是在session里对 tensor 求值。

这段代码用 tf.Session 创建了一个sess的 session 实例。 sess.run() 函数对 tensor 求值，并返回结果。

### constant运算

我们先计算一个线性函数，y=wx+b

```python
w = tf.Variable([[1., 2.]])
x = tf.Variable([[3.], [4.]])
b = tf.constant(.9)

y = tf.add(tf.matmul(w, x), b)

# 变量必须要初始化后才能使用
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    print(y.eval())
```

```
[[11.9]]
```



计算图中所有的数据均以tensor来存储和表达。
tensor是一个高阶张量，二阶张量为矩阵，一阶张量为向量，0阶张量为一个数（标量）。



```python
a = tf.constant(0, name='B')
b = tf.constant(1)
```

```python
print(a)
print(b)
```

```
Tensor("B:0", shape=(), dtype=int32)
Tensor("Const_5:0", shape=(), dtype=int32)
```

其他计算操作也同样如此。 tensorflow中的大部分操作都需要通过tf.xxxxx的方式进行调用。

```python
c = tf.add(a,b)
print(c)
```

```
Tensor("Add_1:0", shape=(), dtype=int32)
```

从加法开始， tf.add() 完成的工作与你期望的一样。它把两个数字，两个 tensor，返回他们的和。减法和乘法同样也是直接调用对应的接口函数

```python
x = tf.subtract(10, 4) # 6
y = tf.multiply(2, 5)  # 10
```

### 还有很多关于数学的api,你可以自己去查阅[文档](https://www.tensorflow.org/api_guides/python/math_ops)

```python
mat_a = tf.constant([[1, 1, 1], [3, 3, 3]])
mat_b = tf.constant([[2, 2, 2],[5, 5, 5]], name='mat_b')
```

```python
mul_a_b = mat_a * mat_b
tf_mul_a_b = tf.multiply(mat_a, mat_b)
tf_matmul_a_b = tf.matmul(mat_a, tf.transpose(mat_b), name='matmul_with_name')
```

```python
print(mul_a_b)
print(tf_mul_a_b)
print(tf_matmul_a_b)
```

```
Tensor("mul:0", shape=(2, 3), dtype=int32)
Tensor("Mul_1:0", shape=(2, 3), dtype=int32)
Tensor("matmul_with_name:0", shape=(2, 2), dtype=int32)
```



```python
with tf.Session() as sess:
    mul_value, tf_mul_value, tf_matmul_value = sess.run([mul_a_b, tf_mul_a_b, tf_matmul_a_b])
```

```python
print(mul_value)
print(tf_mul_value)
print(tf_matmul_value)
```

```
[[ 2  2  2]
 [15 15 15]]
[[ 2  2  2]
 [15 15 15]]
[[ 6 15]
 [18 45]]
```

我们再举几个具体的实例来看看

```python
a = tf.constant(2)
b = tf.constant(3)
x = tf.add(a, b)
with tf.Session() as sess:
	writer = tf.summary.FileWriter('./graphs/const_add', sess.graph)
	print(sess.run(x))
writer.close()
```

```
5
```



```python
a = tf.constant([2, 2], name='a')
b = tf.constant([[0, 1], [2, 3]], name='b')
x = tf.multiply(a, b, name='dot_product')
with tf.Session() as sess:
	writer = tf.summary.FileWriter('./graphs/const_mul', sess.graph) 
	print(sess.run(x))
writer.close()
```

```
[[0 2]
 [4 6]]
```



```python
a = tf.constant([[2, 2],[1, 4]], name='a')
b = tf.constant([[0, 1], [2, 3]], name='b')
x = tf.multiply(a, b, name='dot_product')
y = tf.matmul(a, b, name='mat_mul')
with tf.Session() as sess:
	writer = tf.summary.FileWriter('./graphs/const_mul_2', sess.graph) 
	print(sess.run(x))
	print(sess.run(y))
writer.close()
```

```
[[ 0  2]
 [ 2 12]]
[[ 4  8]
 [ 8 13]]
```

### 各式各样的常量

```python
x = tf.zeros([2, 3], tf.int32)
y = tf.zeros_like(x, optimize=True)
```

```python
print(y)
```

```
Tensor("zeros_like:0", shape=(2, 3), dtype=int32)

```



```python
with tf.Session() as sess:
    print(sess.run(y))
```

```
[[0 0 0]
 [0 0 0]]

```



```python
t_0 = 19 
x = tf.zeros_like(t_0) # ==> 0
y = tf.ones_like(t_0) # ==> 1

with tf.Session() as sess:
    print(sess.run([x]))
    print(sess.run([y]))
```

```
[0]
[1]

```

### 变量

```python
with tf.variable_scope('meh') as scope:
    a = tf.get_variable('a', [10])
    b = tf.get_variable('b', [100])

writer = tf.summary.FileWriter('./graphs/test', tf.get_default_graph())
writer.close()
```

例如：求导也是一个运算

```python
x = tf.Variable(2.0)
y = 2.0 * (x ** 3)
z = 3.0 + y ** 2
grad_z = tf.gradients(z, y)
with tf.Session() as sess:
    sess.run(x.initializer)
    print(sess.run(grad_z))
```

```
[32.0]

```

#### 一次性初始化所有变量

```python
my_var = tf.Variable(2, name="my_var") 

my_var_times_two = my_var.assign(2 * my_var)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(my_var_times_two))
    print(sess.run(my_var_times_two))
    print(sess.run(my_var_times_two))
```

```
4
8
16

```

------

在前面我们都是采用，先赋值再计算的方法，但是在我们的工作过程中，我会遇到先定义变量，后赋值的情况。
如果你想用一个非常量 non-constant 该怎么办？这就是 tf.placeholder() 和 feed_dict 派上用场的时候了。我们来看看如何向 TensorFlow 传输数据的基本知识。

### tf.placeholder()

当你你不能把数据赋值到 x 在把它传给 TensorFlow。因为后面你需要你的 TensorFlow 模型对不同的数据集采取不同的参数。这时你需要 tf.placeholder()！

数据经过 tf.session.run() 函数得到的值，由 tf.placeholder() 返回成一个 tensor，这样你可以在 session 开始跑之前，设置输入。

#### Session’s feed_dict 功能

```python
x = tf.placeholder(tf.string)

with tf.Session() as sess:
    out = sess.run(x, feed_dict={x: 'Hello World'})
    print(out)
```

```
Hello World

```

用 tf.session.run() 里 feed_dict 参数设置占位 tensor。上面的例子显示 tensor x 被设置成字符串 "Hello, world"。当然你也可以用 feed_dict 设置多个 tensor。

placeholder是一个占位符，它通常代表着从外界输入的值。 其中None代表着尚不确定的维度。

```python
input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)
output = tf.multiply(input1, input2)
with tf.Session() as sess:
    print(sess.run([output], feed_dict={input1:[7.], input2:[2.]}))
```

```
[array([14.], dtype=float32)]

```

### 把numpy转换成Tensor

```python
import numpy as np
a = np.zeros((3,3))
print(a)
print('----------------')
ta = tf.convert_to_tensor(a)
with tf.Session() as sess:
     print(sess.run(ta))
```

```
[[0. 0. 0.]
 [0. 0. 0.]
 [0. 0. 0.]]
----------------
[[0. 0. 0.]
 [0. 0. 0.]
 [0. 0. 0.]]

```

### 类型转换

为了让特定运算能运行，有时会对类型进行转换。例如，你尝试下列代码，会报错：

```python
tf.subtract(tf.constant(2.0),tf.constant(1))
```

```
TypeError: Input 'y' of 'Sub' Op has type int32 that does not match type float32 of argument 'x'.

```

只是因为常量 1 是整数，但是常量 2.0 是浮点数 subtract 需要他们能相符。

在这种情况下，你可以让数据都是同一类型，或者强制转换一个值到另一个类型。这里，我们可以把 2.0 转换成整数再相减，这样就能得出正确的结果：

```python
number = tf.subtract(tf.cast(tf.constant(2.0), tf.int32), tf.constant(1))   
with tf.Session() as sess:
    print(sess.run(number))
```

```
1

```

### tensorboard 了解图结构/可视化利器

```python
import tensorflow as tf
a = tf.constant(2, name='A')
b = tf.constant(3, name='B')
x = tf.add(a,b, name='addAB')
with tf.Session() as sess:
    print(sess.run(x))
    #写到日志文件里
    writer = tf.summary.FileWriter('./graphs/add_test', sess.graph)
    #关闭writer
    writer.close()
```

```
5

```

### 启动 TensorBoard

在命令端运行：tensorboard --logdir="./graphs" --port 7007

然后打开Google浏览器访问：http://localhost:7007/

### 实践

#### TensorFlow实现线性回归

```python
import numpy as np
import os
import tensorflow as tf
import matplotlib.pyplot as plt
```

设置生成的图像尺寸和去除警告

```python
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
plt.rcParams["figure.figsize"] = (14, 8)  # 生成的图像尺寸
```

随机生成一个线性的数据，当然你可以换成读取对应的数据集

```python
n_observations = 100
xs = np.linspace(-3, 3, n_observations)
ys = 0.8*xs + 0.1 + np.random.uniform(-0.5, 0.5, n_observations)
plt.scatter(xs, ys)
plt.show()
```

![images](/images/dl/45.png)

### 准备好placeholder

```python
X = tf.placeholder(tf.float32, name='X')
Y = tf.placeholder(tf.float32, name='Y')
```

### 初始化参数/权重

```python
W = tf.Variable(tf.random_normal([1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')
```

### 计算预测结果

```python
Y_pred = tf.add(tf.multiply(X, W), b)
```

### 计算损失值函数

```python
loss = tf.square(Y - Y_pred, name='loss')
```

### 初始化optimizer

```python
learning_rate = 0.01
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
```

### 指定迭代次数，并在session里执行graph

```python
n_samples = xs.shape[0]
init = tf.global_variables_initializer()
with tf.Session() as sess:
    # 记得初始化所有变量
    sess.run(init)

    writer = tf.summary.FileWriter('./graphs/linear_regression', sess.graph)

    # 训练模型
    for i in range(50):
        total_loss = 0
        for x, y in zip(xs, ys):
            # 通过feed_dic把数据灌进去
            _, loss_value = sess.run([optimizer, loss], feed_dict={X: x, Y: y})
            total_loss += loss_value
        if i % 5 == 0:
            print('Epoch {0}: {1}'.format(i, total_loss / n_samples))

    # 关闭writer
    writer.close()

    # 取出w和b的值
    W, b = sess.run([W, b])
```

```
Epoch 0: [0.7908481]
Epoch 5: [0.07929672]
Epoch 10: [0.07929661]
Epoch 15: [0.07929661]
Epoch 20: [0.07929661]
Epoch 25: [0.07929661]
Epoch 30: [0.07929661]
Epoch 35: [0.07929661]
Epoch 40: [0.07929661]
Epoch 45: [0.07929661]

```

### 打印最后更新的w、b的值

```python
print(W, b)
print("W:"+str(W[0]))
print("b:"+str(b[0]))
```

```
[0.82705235] [0.16835527]
W:0.82705235
b:0.16835527

```

### 画出线性回归线

```python
plt.plot(xs, ys, 'bo', label='Real data')
plt.plot(xs, xs * W + b, 'r', label='Predicted data')
plt.legend()
plt.show()
```

![44.png](/images/dl/46.png)

