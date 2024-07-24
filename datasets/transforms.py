import albumentations as AT
from albumentations.pytorch import ToTensorV2


def get_transform(config, mode='train', transform_list=None):
    assert mode in ['train', 'val']
    if transform_list is not None:
        assert isinstance(transform_list, list) and len(transform_list) > 0

    AT_TRANSFORMS = {
                    0: AT.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                    1: ToTensorV2(),
                    2: AT.RandomCrop(height=config.img_size-config.pad_size, width=config.img_size-config.pad_size),
                    3: AT.PadIfNeeded(min_height=config.img_size, min_width=config.img_size, border_mode=0, value=0),
                    4: AT.HorizontalFlip(p=config.h_flip),
                    5: AT.VerticalFlip(p=config.v_flip),
                    6: AT.Resize(height=config.img_size+config.pad_size, width=config.img_size+config.pad_size),
                    7: AT.CenterCrop(height=config.img_size, width=config.img_size),
                    8: AT.ColorJitter(brightness=config.brightness, contrast=config.contrast, saturation=config.saturation, hue=config.hue, p=0.5),
                    }

    transform_hub = {'train': {'cifar10': [2,3,4,0,1], 'cifar100':[2,3,4,0,1], 'fashion_mnist':[0,1], 'mnist':[0,1]},
                     'val': {'cifar10': [0,1], 'cifar100': [0,1], 'fashion_mnist':[0,1], 'mnist':[0,1]}}

    # Transforms are applied sequentially. E.g. for cifar10 training set, it's applied in the order: 2->3->4->0->1.

    if transform_list is not None:
        used_list = transform_list
    else:
        assert config.dataset in transform_hub[mode].keys(), f'Unsupported dataset: {config.dataset}\n'

        used_list = transform_hub[mode][config.dataset]

    transform = AT.Compose([AT_TRANSFORMS[i] for i in used_list])

    return transform
