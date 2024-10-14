try:
    import optuna
except:
    raise RuntimeError('Unable to import Optuna. Please check whether you have installed it correctly.\n')
from .base_config import BaseConfig


class OptunaConfig(BaseConfig):
    def __init__(self,):
        super(OptunaConfig, self).__init__()
        # Dataset
        self.dataset = 'cifar10'
        self.data_root = '/path/to/your/dataset'
        self.download_dataset = True

        # Model
        self.model = 'mobilenet_v2'

        # Training
        self.total_epoch = 100
        self.train_bs = 128

        # Validating
        self.val_bs = 32

        # Training setting
        self.load_ckpt = False

        # DDP
        self.synBN = True
        self.destroy_ddp_process = False

        # Optuna
        self.study_name = 'optuna-study'
        self.study_direction = 'maximize'
        self.num_trial = 100     # 100, 60
        self.save_every_trial = True

    def get_trial_params(self, trial):
        self.optimizer_type = trial.suggest_categorical('optimizer', ['sgd', 'adam', 'adamw'])
        self.base_lr = trial.suggest_loguniform('base_lr', 1e-2, 1e-1)
        self.pretrained = trial.suggest_categorical('pretrained', [True, False])
        self.use_ema = trial.suggest_categorical('use_ema', [True, False])
        self.mixup = trial.suggest_float('mixup', 0.0, 1.0)
        self.brightness = trial.suggest_float('brightness', 0.0, 0.5)
        self.contrast = trial.suggest_float('contrast', 0.0, 0.5)
        self.saturation = trial.suggest_float('saturation', 0.0, 0.5)
        self.h_flip = trial.suggest_float('h_flip', 0.0, 0.5)