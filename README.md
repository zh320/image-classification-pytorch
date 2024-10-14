# Introduction

Simplified PyTorch implementation of image classification, support multi-gpu training and validating, automatic mixed precision training, knowledge distillation, hyperparameter optimization using Optuna, and different datasets, like CIFAR10, MNIST etc.  

# Requirements

torch == 1.8.1  
torchvision  
torchmetrics == 1.2.0  
albumentations  
loguru  
tqdm  
timm == 0.6.12 (optional)  
optuna == 4.0.0 (optional)  
optuna-integration == 4.0.0 (optional)  

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

# Hyperparameter Optimization

This repo also support hyperparameter optimization using Optuna.[^optuna] For example, if you want to search hyperparameters for CIFAR10 dataset using MobileNetv2, you may simply run

```
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 optuna_search.py
```

[^optuna]: [Optuna: A hyperparameter optimization framework](https://github.com/optuna/optuna)

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

## CIFAR10

| Model       | pretrained         | kd                 | optuna                                            | mixup | Epoch | Accuracy(%)        |
| ----------- |:------------------:|:------------------:|:-------------------------------------------------:|:-----:|:-----:| ------------------ |
| ResNet50    |                    |                    |                                                   | 1.0   | 200   | 95.99              |
| ResNet50    |                    |                    |                                                   | 1.0   | 400   | 96.62 (teacher)    |
| ResNet18    |                    |                    |                                                   | 1.0   | 200   | 95.34 (base)       |
| ResNet18    |                    |                    |                                                   | 0.0   | 200   | 94.25 :arrow_down: |
| ResNet18    |                    |                    |                                                   | 0.5   | 200   | 95.08 :arrow_down: |
| ResNet18    | :white_check_mark: |                    |                                                   | 1.0   | 200   | 95.91 :arrow_up:   |
| ResNet18    | :white_check_mark: |                    |                                                   | 1.0   | 400   | 95.95 :arrow_up:   |
| ResNet18    |                    | :white_check_mark: |                                                   | 1.0   | 200   | 95.69 :arrow_up:   |
| ResNet18    |                    |                    |                                                   | 1.0   | 400   | 96.03 :arrow_up:   |
| ResNet18    |                    | :white_check_mark: |                                                   | 1.0   | 400   | 96.12 :arrow_up:   |
| MobileNetv2 |                    |                    |                                                   | 1.0   | 200   | 94.88 (base)       |
| MobileNetv2 | :white_check_mark: |                    |                                                   | 1.0   | 200   | 95.21 :arrow_up:   |
| MobileNetv2 | :white_check_mark: |                    |                                                   | 1.0   | 400   | 95.37 :arrow_up:   |
| MobileNetv2 |                    | :white_check_mark: |                                                   | 1.0   | 200   | 94.83 :arrow_down: |
| MobileNetv2 |                    |                    |                                                   | 1.0   | 400   | 95.29 :arrow_up:   |
| MobileNetv2 |                    | :white_check_mark: |                                                   | 1.0   | 400   | 95.12 :arrow_up:   |
| MobileNetv2 | -                  | -                  | [config](optuna_results/cifar10_mobilenetv2.json) | -     | 100   | 96.39 :arrow_up:   |

## CIFAR100

| Model       | pretrained         | kd                 | optuna                                             | mixup | Epoch | Accuracy(%)        |
| ----------- |:------------------:|:------------------:|:--------------------------------------------------:|:-----:|:-----:| ------------------ |
| ResNet50    |                    |                    |                                                    | 1.0   | 400   | 79.52 (teacher)    |
| ResNet18    |                    |                    |                                                    | 1.0   | 200   | 75.68 (base)       |
| ResNet18    | :white_check_mark: |                    |                                                    | 1.0   | 200   | 78.89 :arrow_up:   |
| ResNet18    | :white_check_mark: |                    |                                                    | 1.0   | 400   | 78.56 :arrow_down: |
| ResNet18    |                    | :white_check_mark: |                                                    | 1.0   | 200   | 75.82 :arrow_up:   |
| ResNet18    |                    |                    |                                                    | 1.0   | 400   | 76.53 :arrow_up:   |
| ResNet18    |                    | :white_check_mark: |                                                    | 1.0   | 400   | 76.85 :arrow_up:   |
| MobileNetv2 |                    |                    |                                                    | 1.0   | 200   | 76.90 (base)       |
| MobileNetv2 | :white_check_mark: |                    |                                                    | 1.0   | 200   | 78.41 :arrow_up:   |
| MobileNetv2 | :white_check_mark: |                    |                                                    | 1.0   | 400   | 78.37 :arrow_down: |
| MobileNetv2 |                    | :white_check_mark: |                                                    | 1.0   | 200   | 76.81 :arrow_down: |
| MobileNetv2 |                    |                    |                                                    | 1.0   | 400   | 77.30 :arrow_up:   |
| MobileNetv2 |                    | :white_check_mark: |                                                    | 1.0   | 400   | 77.85 :arrow_up:   |
| MobileNetv2 | -                  | -                  | [config](optuna_results/cifar100_mobilenetv2.json) | -     | 100   | 82.01 :arrow_up:   |

## MNIST

| Model       | pretrained         | optuna                                          | h_flip | mixup | Epoch | Accuracy(%)        |
| ----------- |:------------------:|:-----------------------------------------------:|:------:|:-----:|:-----:| ------------------ |
| ResNet18    |                    |                                                 | 0.5    | 1.0   | 200   | 99.65 (base)       |
| ResNet18    |                    |                                                 | 0.0    | 1.0   | 200   | 99.65              |
| ResNet18    | :white_check_mark: |                                                 | 0.0    | 1.0   | 200   | 99.65              |
| ResNet18    | :white_check_mark: |                                                 | 0.5    | 1.0   | 200   | 99.68 :arrow_up:   |
| ResNet18    |                    |                                                 | 0.0    | 1.0   | 400   | 99.67 :arrow_up:   |
| ResNet18    |                    |                                                 | 0.5    | 1.0   | 400   | 99.69 :arrow_up:   |
| MobileNetv2 |                    |                                                 | 0.5    | 1.0   | 200   | 99.67 (base)       |
| MobileNetv2 |                    |                                                 | 0.0    | 1.0   | 200   | 99.64 :arrow_down: |
| MobileNetv2 | :white_check_mark: |                                                 | 0.0    | 1.0   | 200   | 99.68 :arrow_up:   |
| MobileNetv2 | :white_check_mark: |                                                 | 0.5    | 1.0   | 200   | 99.62 :arrow_down: |
| MobileNetv2 |                    |                                                 | 0.0    | 1.0   | 400   | 99.64 :arrow_down: |
| MobileNetv2 |                    |                                                 | 0.5    | 1.0   | 400   | 99.65 :arrow_down: |
| MobileNetv2 | -                  | [config](optuna_results/mnist_mobilenetv2.json) | -      | -     | 100   | 99.73 :arrow_up:   |

## Fashion-MNIST

| Model       | pretrained         | optuna                                                  | h_flip | mixup | Epoch | Accuracy(%)        |
| ----------- |:------------------:|:-------------------------------------------------------:|:------:|:-----:|:-----:| ------------------ |
| ResNet18    |                    |                                                         | 0.5    | 1.0   | 200   | 94.33 (base)       |
| ResNet18    |                    |                                                         | 0.0    | 1.0   | 200   | 94.30 :arrow_down: |
| ResNet18    | :white_check_mark: |                                                         | 0.0    | 1.0   | 200   | 94.59 :arrow_up:   |
| ResNet18    | :white_check_mark: |                                                         | 0.5    | 1.0   | 200   | 94.55 :arrow_up:   |
| ResNet18    |                    |                                                         | 0.0    | 1.0   | 400   | 94.20 :arrow_down: |
| ResNet18    |                    |                                                         | 0.5    | 1.0   | 400   | 94.41 :arrow_up:   |
| MobileNetv2 |                    |                                                         | 0.5    | 1.0   | 200   | 94.81 (base)       |
| MobileNetv2 |                    |                                                         | 0.0    | 1.0   | 200   | 94.96 :arrow_up:   |
| MobileNetv2 | :white_check_mark: |                                                         | 0.0    | 1.0   | 200   | 95.28 :arrow_up:   |
| MobileNetv2 | :white_check_mark: |                                                         | 0.5    | 1.0   | 200   | 95.20 :arrow_up:   |
| MobileNetv2 |                    |                                                         | 0.0    | 1.0   | 400   | 95.05 :arrow_up:   |
| MobileNetv2 |                    |                                                         | 0.5    | 1.0   | 400   | 95.21 :arrow_up:   |
| MobileNetv2 | -                  | [config](optuna_results/fashion-mnist_mobilenetv2.json) | -      | -     | 100   | 95.53 :arrow_up:   |

# References
