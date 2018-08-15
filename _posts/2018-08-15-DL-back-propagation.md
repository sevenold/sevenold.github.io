---
layout: post
title: "神经网络入门-反向传播"
date: 2018-08-15
description: "神经网络基础知识点，反向传播、正则项、softmax、交叉熵"
tag: 深度学习
---



### 反向传播（back propagation）算法推导

![images](/images/dl/60.png)

#### 定义损失函数：

#### $E_{total}=\frac12 (y-outo)^2$ 

#### 定义激活函数：

#### $\sigma(x)=sigmod(x) $

### 前向传播

#### 第一层(输入层)：

- #### $x_1\ \ \ x_2\ \ \ b_1$

#### 加权和：

- #### $net h_1=x_1w_1+x_2w_2+b_1$

#### 第二层(隐层)：

- #### $outh_1=sigmod(neth_1)$

#### 加权和：

- #### $neto_1=outh_1w_3+outo_2w_4+b_2$

#### 第三层(输出层)：

- #### $outo_1=sigmod(neto_1)$

#### 计算误差值：

- #### $Eo_1 = \frac12 (y_1-outo_1)^2$

- #### $Eo_2 = \frac12 (y_2-outo_2)^2$

- #### $E_{total}=Eo_1+Eo_2$

`总结`：要是使误差值最小，就需要`误差反向传播算法`，更新得到最小误差的权重参数`w和b`。

### 反向传播

`须知`：我们需要反向传递回去更新每一层对应的权重参数`w和b`。我们使用`链式法则`来`反向模式求导`。

#### **更新第三层（输出层）的权重参数：**

#### 更新参数`w`：

#### $\frac{\partial E_{total}}{\partial w_3}=\frac{\partial E_{total}}{\partial outo_1} \cdot \frac{\partial outo_1}{\partial neto_1} \cdot \frac{\partial neto_1}{\partial w_3}$

####           $= \frac{\partial \frac12(y_1-outo_1)^2}{\partial outo_1} \cdot \frac{\partial sigmod(neto_1)}{\partial neto_1} \cdot \frac{\partial neto_1}{\partial w_3}$

####           $=(outo_1-y_1)\cdot outo_1(1-outo_1)\cdot outh_1$

$w_{3new}=w_{3old}-\eta \frac{\partial E_{total}}{\partial w_3}$， $\eta$是学习率

#### 更新参数`b`：

#### $\frac{\partial E_{total}}{\partial b_2}=\frac{\partial E_{total}}{\partial outo_1} \cdot \frac{\partial outo_1}{\partial neto_1} \cdot \frac{\partial neto_1}{\partial b_2}$

####            $= \frac{\partial \frac12(y_1-outo_1)^2}{\partial outo_1} \cdot \frac{\partial sigmod(neto_1)}{\partial neto_1} \cdot \frac{\partial neto_1}{\partial b_2}$

####              $=(outo_1-y_1)\cdot outo_1(1-outo_1)$

#### $b_{2new}=b_{2old}-\eta \frac{\partial E_{total}}{\partial b_2}$， $\eta$是学习率

#### 同理可得：`w4`：也就是同一层的`w`都可以用这种方式更新。

#### **更新上一层(隐层)的权重参数**：

#### 更新权重参数`w和b`：

#### $\frac{\partial E_{total}}{\partial w_1}=\frac{\partial E_{total}}{\partial outh_1} \cdot \frac{\partial outh_1}{\partial neth_1} \cdot \frac{\partial neth_1}{\partial w_1}$

#### $\frac{\partial E_{total}}{\partial b_1}=\frac{\partial E_{total}}{\partial outh_1} \cdot \frac{\partial outh_1}{\partial neth_1} \cdot \frac{\partial neth_1}{\partial b_1}$

#### 其中：

- #### $\frac{\partial E_{total}}{\partial outh_1} = \frac{\partial Eo_1}{\partial outh_1}+ \frac{\partial Eo_2}{\partial outh_1}$

- #### $\frac{\partial Eo_1}{\partial outh_1} = \frac{\partial Eo_1}{\partial neto_1} \cdot \frac{\partial neto_1}{\partial outh_1}$

- #### $ \frac{\partial Eo_1}{\partial neto_1} = \frac{\partial E_{o_1}}{\partial outo_1} \cdot \frac{\partial outo_1}{\partial neto_1} = (outo_1-y_1)\cdot outo_1(1-outo_1)$

- #### $ \frac{\partial neto_1}{\partial outh_1} = \frac{\partial (outh_1w_3+outo_2w_4+b_2)}{\partial outh_1} = w_3$

- #### 同理可得：

  - #### $\frac{\partial Eo_2}{\partial outh_1} = \frac{\partial Eo_2}{\partial neto_2} \cdot \frac{\partial neto_2}{\partial outh_1}$

  - #### $ \frac{\partial Eo_2}{\partial neto_2} = \frac{\partial E_{o_2}}{\partial outo_2} \cdot \frac{\partial outo_2}{\partial neto_2} = (outo_2-y_2)\cdot outo_2(1-outo_2)$

  - #### $ \frac{\partial neto_2}{\partial outh_1} = w_4$

#### 综合上式：

#### $\frac{\partial E_{total}}{\partial w_1}= [w_3 (outo_1-y_1)\cdot outo_1(1-outo_1) + w_4(outo_2-y_2)\cdot outo_2(1-outo_2)] \cdot outh_1(1-outh_1) \cdot x_1$

#### $\frac{\partial E_{total}}{\partial b_1}= [w_3 (outo_1-y_1)\cdot outo_1(1-outo_1) +w_4(outo_2-y_2)\cdot outo_2(1-outo_2)] \cdot outh_1(1-outh_1) $

#### 更新：

#### $w_{1new}=w_{1old}-\eta \frac{\partial E_{total}}{\partial w_1}$

#### $b_{1new}=b_{1old}-\eta \frac{\partial E_{total}}{\partial b_1}$

#### 同理可得：`w2`：也就是同一层的`w`都可以用这种方式更新。



#### 至此，我们所有的参数都更新完毕了，然后我们可以再次用新的参数进行前向传播，得到的误差就会是最小的



### 推广:

![images](/images/dl/62.png)

#### 我们定义第`L`层的第`i`个神经元更新权重参数时(上标表示层数，下标表示神经元)：

- #### $\frac{\partial E_{total}}{\partial net_i^{(L)}} = \delta_i^{(L)}$

- #### $\frac{\partial E_{total}}{\partial w_{ij}^{(l)}}=outh_j^{(l-1)}\delta_i^{(l)}$ ，其中$w_{ij}^{(l)}$表示第$l$层的第$j$个神经元连接第$l+1$层的第$i$的神经元的相连的权重参数`w`。如下图所示：

  ![images](/images/dl/63.png)

  

### 推广总结：

根据前面我们所定义的：

$E_{total}=\frac12 (y-outo)^2$ 

#### $\sigma(x)=sigmod(x) $

#### $\frac{\partial E_{total}}{\partial w_{ij}^{(l)}}=outh_j^{(l-1)}\delta_i^{(l)}$ 

#### $\delta_i^{(L)}=\frac{\partial E_{total}}{\partial net_i^{(L)}} = \frac{\partial E_{total}}{\partial outh_i} \cdot \frac{\partial outh_i}{\partial net_i^{(L)}}  =\bigtriangledown _{out}E_{total}\times)$

$  \sigma^{\prime}(net_i^{(L)}$

### 举个栗子

![images](/images/dl/61.png)