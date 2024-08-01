# Introduction

Simplified PyTorch implementation of image classification, support multi-gpu training and validating, automatic mixed precision training, knowledge distillation and different datasets, like CIFAR10, MNIST etc.  



# Requirements

torch == 1.8.1  
torchvision  
torchmetrics == 1.2.0  
albumentations  
loguru  
tqdm  
timm == 0.6.12 (optional)  



# Supported models

- [ResNets](models/resnet.py) [^resnet]  
- [MobileNetV2](models/mobilenetv2.py) [^mobilenetv2]  
- timm [^timm]  

This repo provides modified ResNets and MobileNetV2 if you want to train datasets of small-resolution images, e.g. CIFAR10 (32x32) or MNIST (28x28). You can also train datasets of normal-size images like ImageNet using this repo. Besides ResNets and MobileNetV2, you may also refer to timm[^timm] which provides hundereds of pretrained models. For example, if you want to train `mobilenetv3_small` from timm, you may change the [config file](configs/my_config.py) to  

```
config.model = 'timm'
config.timm_model = 'mobilenetv3_small_100'
```

or use [command-line arguments](configs/parser.py)  

```
python main.py --model timm --timm_model mobilenetv3_small_100
```

Details of the configurations can also be found in this [file](configs/parser.py).  

Since most timm models are downsampled 32 times, to retain more details and gain better performances, you may need to modify the downsampling rates of timm model if you want to train datasets of small-resolution images.  

[^resnet]: [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)  
[^mobilenetv2]: [MobileNetV2: Inverted Residuals and Linear Bottlenecks](https://arxiv.org/abs/1801.04381)  
[^timm]: [PyTorch Image Models](https://github.com/huggingface/pytorch-image-models)  



# Supported datasets

 - [CIFAR10](datasets/torchvision_dataset.py) [^cifar]  
 - [CIFAR100](datasets/torchvision_dataset.py) [^cifar]  
 - [MNIST](datasets/torchvision_dataset.py) [^mnist]  
 - [Fashion-MNIST](datasets/torchvision_dataset.py) [^fashion-mnist]  
 - [Custom](datasets/custom_dataset.py)  

 If you want to test other datasets from torchvision, you may refer to this [site](https://pytorch.org/vision/0.9/). Noted that this site is outdated since the version of torchvision(0.9.1) is bounded to torch(1.8.1). If you want to test datasets from newer version of torchvision, you need to update this codebase to be compatible with newer torch. You can also download the image files and build your own dataset following the style of `Custom` dataset if you don't want to update the codebase.  

[^cifar]: [The CIFAR-10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html)  
[^mnist]: [The MNIST database of handwritten digits](https://yann.lecun.com/exdb/mnist/)  
[^fashion-mnist]: [Fashion MNIST](https://github.com/zalandoresearch/fashion-mnist)  



# Knowledge Distillation

Currently only support the original knowledge distillation method proposed by Geoffrey Hinton.[^kd]  

[^kd]: [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531)  



# MixUp

This repo provides batch-wise mixup augmentation.[^mixup] You may control the probability of mixup through this parameter `config.mixup`. If you want to perform mixup for individual images, you may need to implement yourself.  

[^mixup]: [mixup: Beyond Empirical Risk Minimization](https://arxiv.org/abs/1710.09412)



# How to use

## DDP training (recommend)

```
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 main.py
```

## DP training

```
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py
```



# Performances

Coming



# References
