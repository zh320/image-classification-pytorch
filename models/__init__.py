import os, torch, math
import torch.nn as nn

from .resnet import ResNet
from .mobilenetv2 import Mobilenetv2


def get_model(model_name, num_class, img_size, timm_model=None, pretrained=True):
    downsample_rate = get_downsample_rate(img_size)

    if model_name == 'timm':
        assert timm_model is not None, 'You need to choose a timm model.'
        assert img_size >= 64, 'Training small size(<64) images is not supported by timm. You may need to change the model architecture manually.'

        from timm import create_model
        model = create_model(timm_model, pretrained=pretrained, in_chans=3, num_classes=num_class)

    elif 'resnet' in model_name:
        model = ResNet(num_class, model_name, pretrained, downsample_rate)

    elif model_name == 'mobilenet_v2':
        model = Mobilenetv2(num_class, pretrained, downsample_rate)

    else:
        raise NotImplementedError

    return model


def get_teacher_model(config, device):
    if config.kd_training:
        if not os.path.isfile(config.teacher_ckpt):
            raise ValueError(f'Could not find teacher checkpoint at path {config.teacher_ckpt}.')

        downsample_rate = get_downsample_rate(config.img_size)

        if 'resnet' in config.teacher_model:
            model = ResNet(config.num_class, config.teacher_model, pretrained=False, downsample_rate=downsample_rate)

        elif config.teacher_model == 'mobilenet_v2':
            model = Mobilenetv2(config.num_class, pretrained=False, downsample_rate=downsample_rate)

        else:
            raise NotImplementedError

        teacher_ckpt = torch.load(config.teacher_ckpt, map_location=torch.device('cpu'))
        model.load_state_dict(teacher_ckpt['state_dict'])
        del teacher_ckpt

        model = model.to(device)    
        model.eval()
    else:
        model = None

    return model


def get_downsample_rate(img_size):
    assert img_size > 16, 'Minimum size for training is 16'
    downsample_rate = 2**min(int(max(math.log2(img_size/32), 0)), 2) * 8

    return downsample_rate
