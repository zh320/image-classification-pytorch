import argparse


def load_parser(config):
    args = get_parser()

    for k,v in vars(args).items():
        if v is not None:
            try:
                exec(f"config.{k} = v")
            except:
                raise RuntimeError(f'Unable to assign value to config.{k}')
    return config


def get_parser():
    parser = argparse.ArgumentParser()
    # Dataset
    parser.add_argument('--dataset', type=str, default=None,
        help='choose which dataset you want to use, choices are `cifar10`, `cifar100`, `fashion_mnist`, `mnist` or custom dataset')
    parser.add_argument('--dataroot', type=str, default=None, 
        help='path to your dataset')
    parser.add_argument('--num_class', type=int, default=None, 
        help='number of classes')
    parser.add_argument('--ignore_index', type=int, default=None, 
        help='ignore index used for cross_entropy/ohem loss')
    parser.add_argument('--download_dataset', action='store_true', default=None,
        help='whether to download dataset from torchvision or not (default: False)')

    # Model
    parser.add_argument('--model', type=str, default=None, 
        choices=['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'mobilenet_v2', 'timm'],
        help='choose which model you want to use')
    parser.add_argument('--timm_model', type=str, default=None, 
        help='choose which model from timm you want to use, please use `timm.list_models()` to check which models are available')
    parser.add_argument('--pretrained', action='store_false', default=None,
        help='whether to load pretrained ImageNet weights or not (default: True)')

    # Training
    parser.add_argument('--total_epoch', type=int, default=None, 
        help='number of total training epochs')
    parser.add_argument('--base_lr', type=float, default=None, 
        help='base learning rate for single GPU, total learning rate *= gpu number')
    parser.add_argument('--train_bs', type=int, default=None, 
        help='training batch size for single GPU, total batch size *= gpu number')

    # Validating
    parser.add_argument('--val_bs', type=int, default=None, 
        help='validating batch size for single GPU, total batch size *= gpu number')    
    parser.add_argument('--begin_val_epoch', type=int, default=None, 
        help='which epoch to start validating')    
    parser.add_argument('--val_interval', type=int, default=None, 
        help='epoch interval between two validations')
    parser.add_argument('--top_k', type=int, default=None, 
        help='how many highest logits to be considered for `accuracy` metric')

    # Testing
    parser.add_argument('--is_testing', action='store_true', default=None,
        help='whether to perform testing/predicting or not (default: False)')
    parser.add_argument('--test_bs', type=int, default=None, 
        help='testing batch size (currently only support single GPU)')
    parser.add_argument('--test_data_folder', type=str, default=None, 
        help='path to your testing image folder')
    parser.add_argument('--class_map', type=dict, default=None, 
        help='input dict to convert the results from number to meaningful strings')
    parser.add_argument('--test_transform', type=list, default=None, 
        help='transforms for given testing dataset, should be similar to your val_transform of given ckpt')

    # Loss
    parser.add_argument('--loss_type', type=str, default=None, choices = ['ce'],
        help='choose which loss you want to use')
    parser.add_argument('--class_weights', type=tuple, default=None, 
        help='class weights for cross entropy loss')

    # Scheduler
    parser.add_argument('--lr_policy', type=str, default=None, 
        choices = ['cos_warmup', 'linear', 'step'],
        help='choose which learning rate policy you want to use')
    parser.add_argument('--warmup_epochs', type=int, default=None, 
        help='warmup epoch number for `cos_warmup` learning rate policy')
    parser.add_argument('--step_size', type=int, default=None, 
        help='number of step to reduce lr for `step` learning rate policy')

    # Optimizer
    parser.add_argument('--optimizer_type', type=str, default=None, 
        choices = ['sgd', 'adam', 'adamw'],
        help='choose which optimizer you want to use')
    parser.add_argument('--momentum', type=float, default=None, 
        help='momentum of SGD optimizer')
    parser.add_argument('--weight_decay', type=float, default=None, 
        help='weight decay rate of SGD optimizer')

    # Monitoring
    parser.add_argument('--save_ckpt', action='store_false', default=None,
        help='whether to save checkpoint or not (default: True)')
    parser.add_argument('--save_dir', type=str, default=None, 
        help='path to save checkpoints and training configurations etc.')
    parser.add_argument('--use_tb', action='store_false', default=None,
        help='whether to use tensorboard or not (default: True)')
    parser.add_argument('--tb_log_dir', type=str, default=None, 
        help='path to save tensorboard logs')
    parser.add_argument('--ckpt_name', type=str, default=None, 
        help='given name of the saved checkpoint, otherwise use `last` and `best`')

    # Training setting
    parser.add_argument('--amp_training', action='store_true', default=None,
        help='whether to use automatic mixed precision training or not (default: False)')
    parser.add_argument('--resume_training', action='store_false', default=None,
        help='whether to load training state from specific checkpoint or not if present (default: True)')
    parser.add_argument('--load_ckpt', action='store_false', default=None,
        help='whether to load given checkpoint or not if exist (default: True)')
    parser.add_argument('--load_ckpt_path', type=str, default=None, 
        help='path to load specific checkpoint, otherwise try to load `last.pth`')
    parser.add_argument('--base_workers', type=int, default=None, 
        help='number of workers for single GPU, total workers *= number of GPU')
    parser.add_argument('--random_seed', type=int, default=None, 
        help='random seed')
    parser.add_argument('--use_ema', action='store_true', default=None,
        help='whether to use exponetial moving average to update weights or not (default: False)')

    # Augmentation
    parser.add_argument('--img_size', type=int, default=None, 
        help='training/validating image size')
    parser.add_argument('--pad_size', type=int, default=None, 
        help='pad size for cropping, crop_size = img_size - pad_size')
    parser.add_argument('--mixup', type=float, default=None, 
        help='probability to perform Mixup')
    parser.add_argument('--mixup_alpha', type=float, default=None, 
        help='parameter for Mixup augmentation')
    parser.add_argument('--brightness', type=float, default=None, 
        help='brightness limit for ColorJitter augmentation')
    parser.add_argument('--contrast', type=float, default=None, 
        help='contrast limit for ColorJitter augmentation')
    parser.add_argument('--saturation', type=float, default=None, 
        help='saturation limit for ColorJitter augmentation')
    parser.add_argument('--hue', type=float, default=None, 
        help='hue limit for ColorJitter augmentation')
    parser.add_argument('--h_flip', type=float, default=None, 
        help='probability to perform HorizontalFlip')
    parser.add_argument('--v_flip', type=float, default=None, 
        help='probability to perform VerticalFlip')

    # DDP
    parser.add_argument('--synBN', action='store_true', default=None, 
        help='whether to use SyncBatchNorm or not if trained with DDP (default: False)')
    parser.add_argument('--local_rank', type=int, default=None, 
        help='used for DDP, DO NOT CHANGE')

    # Knowledge Distillation
    parser.add_argument('--kd_training', action='store_true', default=None,
        help='whether to use knowledge distillation or not (default: False)')
    parser.add_argument('--teacher_ckpt', type=str, default=None, 
        help='path to your teacher checkpoint')
    parser.add_argument('--teacher_model', type=str, default=None, 
        help='name of your teacher model')
    parser.add_argument('--kd_loss_type', type=str, default=None, choices = ['kl_div', 'mse'],
        help='choose which loss you want to perform knowledge distillation')
    parser.add_argument('--kd_loss_coefficient', type=float, default=None, 
        help='coefficient of knowledge distillation loss')
    parser.add_argument('--kd_temperature', type=float, default=None, 
        help='temperature used for KL divergence loss')

    args = parser.parse_args()
    return args
