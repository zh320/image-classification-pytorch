import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from .transforms import get_transform


class TestDataset(Dataset):
    def __init__(self, config):
        data_folder = os.path.expanduser(config.test_data_folder)

        if not os.path.isdir(data_folder):
            raise RuntimeError(f'Test image directory: {data_folder} does not exist.')

        self.transform = get_transform(config, transform_list=config.test_transform)

        self.images = []
        self.img_names = []

        for file_name in os.listdir(data_folder):
            self.images.append(os.path.join(data_folder, file_name))
            self.img_names.append(file_name)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = np.asarray(Image.open(self.images[index]).convert('RGB'))
        img_name = self.img_names[index]

        # Perform augmentation and normalization
        augmented = self.transform(image=image)
        image = augmented['image']

        return image, img_name
