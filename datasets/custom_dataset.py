import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from .transforms import get_transform


class CustomDataset(Dataset):
    '''This is an example of how you can play with your own dataset. This toy dataset can be downloaded via the following link
        https://www.kaggle.com/datasets/karakaggle/kaggle-cat-vs-dog-dataset
        Make sure you have splitted your dataset into train-val subsets properly.
    '''
    labels = {'Cat':0, 'Dog':1}
    def __init__(self, config, mode='train'):
        assert mode in ['train', 'val']
        data_root = os.path.expanduser(config.data_root)
        data_folder = os.path.join(data_root, mode)

        if not os.path.isdir(data_folder):
            raise RuntimeError(f'Image directory: {data_folder} does not exist.')

        transform_list = [6,7,4,0,1] if mode == 'train' else [6,7,0,1]
        self.transform = get_transform(config, transform_list=transform_list)

        self.images, self.labels = [], []
        for pet_cls in os.listdir(data_folder):
            pet_folder = os.path.join(data_folder, pet_cls)
            label = CustomDataset.labels[pet_cls]
            for file_name in os.listdir(pet_folder):
                self.images.append(os.path.join(pet_folder, file_name))
                self.labels.append(label)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = np.asarray(Image.open(self.images[index]).convert('RGB'))
        label = self.labels[index]

        # Perform augmentation and normalization
        augmented = self.transform(image=image)
        image = augmented['image']

        return image, label
