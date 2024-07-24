from .base_config import BaseConfig


class MyConfig(BaseConfig):
    def __init__(self,):
        super(MyConfig, self).__init__()
        # Dataset
        self.dataset = 'cifar10'
        self.data_root = '/path/to/your/dataset'
        self.download_dataset = True

        # Model
        self.model = 'resnet18'
        self.pretrained = False

        # Training
        self.total_epoch = 200
        self.base_lr = 0.1
        self.train_bs = 128

        # Validating
        self.val_bs = 32
        self.top_k = 1

        # Loss
        self.loss_type = 'ce'

        # Scheduler
        self.lr_policy = 'cos_warmup'

        # Optimizer
        self.optimizer_type = 'sgd'

        # Training setting
        self.amp_training = False
        self.use_ema = True

        # Augmentation
        self.mixup = 1.0
        self.brightness = 0.
        self.contrast = 0.
        self.saturation = 0.
        self.hue = 0.
        self.h_flip = 0.5
