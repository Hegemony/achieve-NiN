# achieve-NiN


## Dependencies
* [Python 3.5+](https://www.continuum.io/downloads)
* [PyTorch 0.4.0+](http://pytorch.org/)

### NiN重复使用由卷积层和代替全连接层的1×11×1卷积层构成的NiN块来构建深层网络。
### NiN去除了容易造成过拟合的全连接输出层，而是将其替换成输出通道数等于标签类别数的NiN块和全局平均池化层。
### NiN的以上设计思想影响了后面一系列卷积神经网络的设计。
