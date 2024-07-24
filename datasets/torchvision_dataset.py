"""
Create by:  zh320
Date:       2024/07/13
"""

import torch
import numpy as np
from PIL import Image
from torchvision.datasets import CIFAR10, CIFAR100, FashionMNIST, MNIST # ImageNet

from .transforms import get_transform


class TorchvisionDataset:
    dataset_hub = {'cifar10':CIFAR10, 'cifar100':CIFAR100, 'fashion_mnist':FashionMNIST, 'mnist':MNIST,}

    def __init__(self, config, mode='train', transform=None):
        if config.dataset == 'imagenet':
            raise RuntimeError('Do you really want to train ImageNet from scratch? If YES, make sure you have enough storage/compute power \
                                and change the codes accordingly.\n')

        if config.dataset not in TorchvisionDataset.dataset_hub.keys():
            raise ValueError(f"Invalid dataset name: {config.dataset}")

        assert mode in ['train', 'val']
        is_train = mode == 'train'

        if transform is None:
            transform = get_transform(config, mode=mode)
        self.transform = transform

        self.dataset = TorchvisionDataset.dataset_hub[config.dataset](root=config.data_root, train=is_train, transform=None, download=config.download_dataset)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, label = self.dataset[index]

        if self.transform is not None:
            if isinstance(image, torch.Tensor):
                if len(image.shape) == 2:   # Change HW to HWC
                    image = torch.stack([image for _ in range(3)], dim=-1)
                image = image.numpy()
            elif isinstance(image, Image.Image):
                image = np.array(image.convert('RGB'))

            transformed = self.transform(image=image)
            image = transformed["image"]

        return image, label
