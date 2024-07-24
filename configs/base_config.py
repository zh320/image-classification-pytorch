class BaseConfig:
    def __init__(self,):
        # Dataset
        self.dataset = None
        self.data_root = None
        self.num_class = None
        self.ignore_index = 255
        self.download_dataset = False

        # Model
        self.model = None
        self.timm_model = None
        self.pretrained = True

        # Training
        self.total_epoch = 200
        self.base_lr = 0.01
        self.train_bs = 16      # For each GPU

        # Validating
        self.val_bs = 16        # For each GPU
        self.begin_val_epoch = 0    # Epoch to start validation
        self.val_interval = 1   # Epoch interval between validation
        self.top_k = 1

        # Testing
        self.is_testing = False
        self.test_bs = 16
        self.test_data_folder = None
        self.class_map = None
        self.test_transform = None

        # Loss
        self.loss_type = 'ce'
        self.class_weights = None

        # Scheduler
        self.lr_policy = 'cos_warmup'
        self.warmup_epochs = 3
        self.step_size = 5000        # For step lr

        # Optimizer
        self.optimizer_type = 'sgd'
        self.momentum = 0.9         # For SGD
        self.weight_decay = 1e-4    # For SGD

        # Monitoring
        self.save_ckpt = True
        self.save_dir = 'save'
        self.use_tb = True          # tensorboard
        self.tb_log_dir = None
        self.ckpt_name = None
        self.logger_name = 'cls_trainer'

        # Training setting
        self.amp_training = False
        self.resume_training = True
        self.load_ckpt = True
        self.load_ckpt_path = None
        self.base_workers = 8
        self.random_seed = 1
        self.use_ema = False

        # Augmentation
        self.img_size = None
        self.pad_size = None
        self.mixup = 0.
        self.mixup_alpha = 1.
        self.brightness = 0.
        self.contrast = 0.
        self.saturation = 0.
        self.hue = 0.
        self.h_flip = 0.
        self.v_flip = 0.

        # DDP
        self.synBN = False

        # Knowledge Distillation
        self.kd_training = False
        self.teacher_ckpt = ''
        self.teacher_model = None
        self.kd_loss_type = 'kl_div'
        self.kd_loss_coefficient = 1.
        self.kd_temperature = 4.

    def init_dependent_config(self):
        if self.load_ckpt_path is None and not self.is_testing:
            self.load_ckpt_path = f'{self.save_dir}/last.pth'

        if self.tb_log_dir is None:
            self.tb_log_dir = f'{self.save_dir}/tb_logs/'

        num_class_hub = {'cifar10':10, 'cifar100':100, 'fashion_mnist':10, 'mnist':10}
        if self.dataset in num_class_hub.keys():
            print(f'Override num_class from {self.num_class} to {num_class_hub[self.dataset]}.\n')
            self.num_class = num_class_hub[self.dataset]

        if self.num_class is None:
            raise ValueError(f'Please give the value of `num_class` for dataset: {config.dataset}\n')

        if self.img_size is None:
            img_size_hub = {'cifar10':32, 'cifar100':32, 'fashion_mnist':28, 'mnist':28, 'custom':224}
            if self.dataset in img_size_hub:
                self.img_size = img_size_hub[self.dataset]
            else:
                raise ValueError(f'Dataset: {self.dataset} does not have default `img_size`. Please give one.\n')

        if self.pad_size is None:
            pad_size_hub = {'cifar10':4, 'cifar100':4, 'custom':32}
            if self.dataset in pad_size_hub:
                self.pad_size = pad_size_hub[self.dataset]
            else:
                print(f'Dataset: {self.dataset} does not have default `pad_size`. It will be assigned `0`.\n')
                self.pad_size = 0
        assert self.pad_size < self.img_size

        if self.is_testing:
            print(f'Override dataset from `{self.dataset}` to `test` in test mode.\n')
            self.dataset = 'test'

            if self.class_map is None:
                raise ValueError('In test mode, you need to provide the class map for given dataset.')

            if self.test_transform is None:
                raise ValueError('In test mode, you need to provide the test transform for given dataset.')

            assert isinstance(self.class_map, dict)
            assert len(self.class_map) == self.num_class, 'Class map does not match the number of class.'
            assert isinstance(self.test_transform, list)
