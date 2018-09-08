---
layout: post
title: "TensorFlow-cifar10-图像分类之训练模型及可视化数据"
date: 2018-09-7
description: "TensorFlow, cifar10"
tag: TensorFlow
---

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

