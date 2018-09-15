---
layout: post
title: "Pytorch实现CIFAR10之训练模型"
date: 2018-09-15
description: "pytorch, cifar10"
tag: Pytorch
---

### 示例代码：

```python

lr = 0.001
best_acc = 0  # best test accuracy
epochs = 20
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=5e-4)
# Training
def train():
    for epoch in range(epochs):
        print('\nEpoch: {}'.format(epoch + 1))
        train_loss = 0
        correct = 0
        total = 0
        for step, (inputs, targets) in enumerate(trainloader):

            outputs = net(inputs)[0]
            loss = criterion(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            if step % 50 == 0:
                print("step:{} ".format(step))
                print("Loss:%.4f " % (train_loss / (step + 1)))
                print("train Accuracy: %4f" % (100.*correct/total))

    print('Finished Training')


def test():
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for step, (inputs, targets) in enumerate(testloader):
            outputs, _ = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            if step % 50 == 0:
                print("step:{} ".format(step))
                print("Loss:%.4f " % (test_loss / (step + 1)))
                print("Test Accuracy: %4f" % (100.*correct/total))
    print("TEST Finished")

```





