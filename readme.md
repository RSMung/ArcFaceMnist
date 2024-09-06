# 基于Resnet18模型在MNIST数据集上复现ArcFace
近期我了解到ArcFace Loss在人脸识别算法中有着重大影响，截止2024.9.4引用量达到了7080，因此我想复现这篇经典文章。MNIST数据集是比较简单，容易处理的数据集，因此基于它来完成复现实验。

文章完整的标题为：ArcFace: Additive Angular Margin Loss for Deep Face Recognition
发表于CVPR2019

官方的代码库为：https://github.com/deepinsight/insightface/tree/master/recognition/arcface_torch

我的运行环境如下：

> python==3.8.0  
> pytorch==1.11.0  
> torchvision==0.12.0  
> ubuntu==22.04  
> GPU==RTX 3090  


## 1 使用ArcFace Loss训练Resnet18
修改main.py中的flag为flag1后，运行:  
> python main.py

运行过程中的参数可以在 `trainArcFace.py`的`TrainArcFaceParams`中修改

## 2 使用常规方法训练Resnet18
修改main.py中的flag为flag2后，运行:  
> python main.py

运行过程中的参数可以在 `trainResnet.py`的`TrainResnetParams`中修改


## 3 测试
将`trainArcFace.py`或者`trainResnet.py`中的`trainArcFaceMain`或`trainResnetSoftmaxMain` 注释掉, 然后运行`python main.py`即可