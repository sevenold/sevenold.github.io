---
layout: post
title: "Keras-cifar10-图像分类"
date: 2018-09-7
description: "TensorFlow, cifar10"
tag: Keras
---

### **预处理数据**：

```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/9/6 15:31
# @Author  : Seven
# @Site    : 
# @File    : Read_data.py
# @Software: PyCharm


# TODO: 加载数据
import pickle
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelBinarizer


def load_cifar10_batch(path, batch_id):
    """
    加载batch的数据
    :param path: 数据存储的目录
    :param batch_id:batch的编号
    :return:features and labels
    """

    with open(path + '/data_batch_' + str(batch_id), mode='rb') as file:
        batch = pickle.load(file, encoding='latin1')

    # features and labels
    features = batch['data'].reshape((len(batch['data']), 3, 32, 32)).transpose(0, 2, 3, 1)
    labels = batch['labels']

    return features, labels


# 数据预处理
def pre_processing_data(x_train, y_train, x_test, y_test):
    # features
    minmax = MinMaxScaler()
    # 重塑数据
    # (50000, 32, 32, 3) --> (50000, 32*32*3)
    x_train_rows = x_train.reshape(x_train.shape[0], 32*32*3)
    # (10000, 32, 32, 3) --> (10000, 32*32*3)
    x_test_rows = x_test.reshape(x_test.shape[0], 32*32*3)
    # 归一化
    x_train_norm = minmax.fit_transform(x_train_rows)
    x_test_norm = minmax.fit_transform(x_test_rows)
    # 重塑数据
    x_train = x_train_norm.reshape(x_train_norm.shape[0], 32, 32, 3)
    x_test = x_test_norm.reshape(x_test_norm.shape[0], 32, 32, 3)

    # labels
    # 对标签进行one-hot
    n_class = 10
    label_binarizer = LabelBinarizer().fit(np.array(range(n_class)))
    y_train = label_binarizer.transform(y_train)
    y_test = label_binarizer.transform(y_test)

    return x_train, y_train, x_test, y_test


def cifar10_data():
    # 加载训练数据
    cifar10_path = 'data'
    # 一共是有5个batch的训练数据
    x_train, y_train = load_cifar10_batch(cifar10_path, 1)
    for n in range(2, 6):
        features, labels = load_cifar10_batch(cifar10_path, n)
        x_train = np.concatenate([x_train, features])
        y_train = np.concatenate([y_train, labels])

    # 加载测试数据
    with open(cifar10_path + '/test_batch', mode='rb') as file:
        batch = pickle.load(file, encoding='latin1')
        x_test = batch['data'].reshape((len(batch['data']), 3, 32, 32)).transpose(0, 2, 3, 1)
        y_test = batch['labels']

    x_train, y_train, x_test, y_test = pre_processing_data(x_train, y_train, x_test, y_test)

    return x_train, y_train, x_test, y_test
```

### **配置文件**：

```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/9/7 14:10
# @Author  : Seven
# @Site    : 
# @File    : config.py
# @Software: PyCharm
# TODO: 基本配置

nb_classes = 10
class_name = {
    0: 'airplane',
    1: 'automobile',
    2: 'bird',
    3: 'cat',
    4: 'deer',
    5: 'dog',
    6: 'frog',
    7: 'horse',
    8: 'ship',
    9: 'truck',
}

```

### **类VGG16Net网络**：

```python
def KerasVGG():
    """
    模型采用类似于 VGG16 的结构：
        使用固定尺寸的小卷积核 (3x3)
        以2的幂次递增的卷积核数量 (64, 128, 256)
        两层卷积搭配一层池化
        全连接层没有采用 VGG16 庞大的三层结构，避免运算量过大，仅使用 128 个节点的单个FC
        权重初始化采用He Normal
    :return:
    """
    name = 'VGG'
    inputs = Input(shape=(32, 32, 3))
    net = inputs
    # (32, 32, 3)-->(32, 32, 64)
    net = Convolution2D(filters=64, kernel_size=3, strides=1,
                        padding='same', activation='relu',
                        kernel_initializer='he_normal')(net)
    # (32, 32, 64)-->(32, 32, 64)
    net = Convolution2D(filters=64, kernel_size=3, strides=1,
                        padding='same', activation='relu',
                        kernel_initializer='he_normal')(net)
    # (32, 32, 64)-->(16, 16, 64)
    net = MaxPooling2D(pool_size=2, strides=2, padding='valid')(net)

    # (16, 16, 64)-->(16, 16, 128)
    net = Convolution2D(filters=128, kernel_size=3, strides=1,
                        padding='same', activation='relu',
                        kernel_initializer='he_normal')(net)
    # (16, 16, 64)-->(16, 16, 128)
    net = Convolution2D(filters=128, kernel_size=3, strides=1,
                        padding='same', activation='relu',
                        kernel_initializer='he_normal')(net)
    # (16, 16, 128)-->(8, 8, 128)
    net = MaxPooling2D(pool_size=2, strides=2, padding='valid')(net)

    # (8, 8, 128)-->(8, 8, 256)
    net = Convolution2D(filters=256, kernel_size=3, strides=1,
                        padding='same', activation='relu',
                        kernel_initializer='he_normal')(net)
    # (8, 8, 256)-->(8, 8, 256)
    net = Convolution2D(filters=256, kernel_size=3, strides=1,
                        padding='same', activation='relu',
                        kernel_initializer='he_normal')(net)
    # (8, 8, 256)-->(4, 4, 256)
    net = MaxPooling2D(pool_size=2, strides=2, padding='valid')(net)

    # (4, 4, 256) --> 4*4*256=4096
    net = Flatten()(net)
    # 4096 --> 128
    net = Dense(units=128, activation='relu',
                kernel_initializer='he_normal')(net)
    # Dropout
    net = Dropout(0.5)(net)
    # 128 --> 10
    net = Dense(units=config.nb_classes, activation='softmax',
                kernel_initializer='he_normal')(net)
    return inputs, net, name
```

### **添加BN层**:

```python
def KerasBN():
    """
    添加batch norm 层
    :return:
    """
    name = 'BN'
    inputs = Input(shape=(32, 32, 3))
    net = inputs

    # (32, 32, 3)-->(32, 32, 64)
    net = Convolution2D(filters=64, kernel_size=3, strides=1,
                        padding='same', activation='relu',
                        kernel_initializer='he_normal')(net)
    net = BatchNormalization()(net)
    net = Activation('relu')(net)
    # (32, 32, 64)-->(32, 32, 64)
    net = Convolution2D(filters=64, kernel_size=3, strides=1,
                        padding='same', activation='relu',
                        kernel_initializer='he_normal')(net)
    net = BatchNormalization()(net)
    net = Activation('relu')(net)
    # (32, 32, 64)-->(16, 16, 64)
    net = MaxPooling2D(pool_size=2, strides=2, padding='valid')(net)

    # (16, 16, 64)-->(16, 16, 128)
    net = Convolution2D(filters=128, kernel_size=3, strides=1,
                        padding='same', activation='relu',
                        kernel_initializer='he_normal')(net)
    net = BatchNormalization()(net)
    net = Activation('relu')(net)
    # (16, 16, 64)-->(16, 16, 128)
    net = Convolution2D(filters=128, kernel_size=3, strides=1,
                        padding='same', activation='relu',
                        kernel_initializer='he_normal')(net)
    net = BatchNormalization()(net)
    net = Activation('relu')(net)
    # (16, 16, 128)-->(8, 8, 128)
    net = MaxPooling2D(pool_size=2, strides=2, padding='valid')(net)

    # (8, 8, 128)-->(8, 8, 256)
    net = Convolution2D(filters=256, kernel_size=3, strides=1,
                        padding='same', activation='relu',
                        kernel_initializer='he_normal')(net)
    net = BatchNormalization()(net)
    net = Activation('relu')(net)
    # (8, 8, 128)-->(8, 8, 256)
    net = Convolution2D(filters=256, kernel_size=3, strides=1,
                        padding='same', activation='relu',
                        kernel_initializer='he_normal')(net)
    net = BatchNormalization()(net)
    net = Activation('relu')(net)
    # (8, 8, 256)-->(4, 4, 256)
    net = MaxPooling2D(pool_size=2, strides=2, padding='valid')(net)

    # (4, 4, 256) --> 4*4*256=4096
    net = Flatten()(net)
    # 4096 --> 128
    net = Dense(units=128, activation='relu',
                kernel_initializer='he_normal')(net)
    # Dropout
    net = Dropout(0.5)(net)
    # 128 --> 10
    net = Dense(units=config.nb_classes, activation='softmax',
                kernel_initializer='he_normal')(net)

    return inputs, net, name


```

### **训练文件**:

```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/9/7 14:07
# @Author  : Seven
# @Site    : 
# @File    : TrainModel.py
# @Software: PyCharm
# TODO: 模型训练

# TODO：导入环境
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
import time


def run(network, X_train, y_train, X_test, y_test, augmentation=False):

    print('X_train shape:', X_train.shape)
    print('Y_train shape:', y_train.shape)
    print(X_train.shape[0], 'x_training samples')
    print(X_test.shape[0], 'validation samples')

    # TODO: 规一化处理
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255

    # TODO： 初始化模型
    inputs, logits, name = network
    model = Model(inputs=inputs, outputs=logits, name='model')

    # TODO: 计算损失值并初始化optimizer
    model.compile(loss='categorical_crossentropy',
                  optimizer='adadelta',
                  metrics=['accuracy'])

    model.summary()
    print('FUNCTION READY!!')
    # TODO: 开始训练
    print('TRAINING....')
    epoch = 100
    batch_size = 256
    start = time.time()
    # 数据增强
    if augmentation:
        aug_gen = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
            width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
            horizontal_flip=True,  # randomly flip images
            vertical_flip=False,  # randomly flip images
        )
        aug_gen.fit(X_train)
        generator = aug_gen.flow(X_train, y_train, batch_size=batch_size)
        out = model.fit_generator(generator=generator,
                                  steps_per_epoch=50000 // batch_size,
                                  epochs=epoch,
                                  validation_data=(X_test, y_test))
        # TODO: 保存模型
        model.save('CIFAR10_model_with_data_augmentation_%s.h5' % name)
    # 不使用数据增强
    else:
        out = model.fit(x=X_train, y=y_train,
                        batch_size=batch_size,
                        epochs=epoch,
                        validation_data=(X_test, y_test),
                        shuffle=True)
        # TODO: 保存模型
        model.save('CIFAR10_model_no_data_augmentation_%s.h5' % name)

    loss, accuracy = model.evaluate(X_train, y_train, verbose=0)
    print("Training Accuracy = %.2f %%     loss = %f" % (accuracy * 100, loss))
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print("Testing Accuracy = %.2f %%    loss = %f" % (accuracy * 100, loss))
    print('@ Total Time Spent: %.2f seconds' % (time.time() - start))

    return out, epoch, model

```

### **可视化数据**：

```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/9/7 15:00
# @Author  : Seven
# @Site    : 
# @File    : visualization.py
# @Software: PyCharm
# TODO: 可视化数据
import matplotlib.pyplot as plt
import numpy as np


def plot_acc_loss(data, epoch):
    """
    可视化数据
    :param data: 数据
    :param epoch: 迭代次数
    :return:
    """
    acc, loss, val_acc, val_loss = data.history['acc'], data.history['loss'], \
                                   data.history['val_acc'], data.history['val_loss']
    plt.figure(figsize=(15, 5))
    plt.subplot(121)
    plt.plot(range(epoch), acc, label='Train')
    plt.plot(range(epoch), val_acc, label='Test')
    plt.title('Accuracy over ' + str(epoch) + ' Epochs', size=15)
    plt.legend()
    plt.grid(True)

    plt.subplot(122)
    plt.plot(range(epoch), loss, label='Train')
    plt.plot(range(epoch), val_loss, label='Test')
    plt.title('Loss over ' + str(epoch) + ' Epochs', size=15)
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_image(x_test, y_test, class_name, model):
    rand_id = np.random.choice(range(10000), size=10)
    X_pred = np.array([x_test[i] for i in rand_id])
    y_true = [y_test[i] for i in rand_id]
    y_true = np.argmax(y_true, axis=1)
    y_true = [class_name[name] for name in y_true]
    y_pred = model.predict(X_pred)
    y_pred = np.argmax(y_pred, axis=1)
    y_pred = [class_name[name] for name in y_pred]
    plt.figure(figsize=(15, 7))
    for i in range(10):
        plt.subplot(2, 5, i + 1)
        plt.imshow(X_pred[i].reshape(32, 32, 3), cmap='gray')
        plt.title('True: %s \n Pred: %s' % (y_true[i], y_pred[i]), size=15)
    plt.show()

```

### **验证本地图片**：

```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/9/7 19:18
# @Author  : Seven
# @Site    : 
# @File    : TestModel.py
# @Software: PyCharm
import numpy as np
from PIL import Image
from keras.models import load_model
import Read_data
import config
import visualization


def run():
    # 加载模型
    model = load_model('CIFAR10_model_with_data_augmentation_VGG.h5')
    # 加载数据
    X_train, y_train, X_test, y_test = Read_data.load_data()
    visualization.plot_image(X_test, y_test, config.class_name, model)


def local_photos():
    # input
    im = Image.open('image/dog-1.jpg')
    # im.show()
    im = im.resize((32, 32))

    # print(im.size, im.mode)

    im = np.array(im).astype(np.float32)
    im = np.reshape(im, [-1, 32*32*3])
    im = (im - (255 / 2.0)) / 255
    batch_xs = np.reshape(im, [-1, 32, 32, 3])

    model = load_model('CIFAR10_model_with_data_augmentation_VGG.h5')
    output = model.predict(batch_xs)

    print(output)
    print('the out put is :', config.class_name[np.argmax(output)])


if __name__ == '__main__':
    local_photos()

```

### **输出：**

```python
[[0. 0. 0. 0. 0. 1. 0. 0. 0. 0.]]
the out put is : dog
```

