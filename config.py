import os
import yaml
from yacs.config import CfgNode as CN

_C = CN()
# -----------------------------------------------------------------------------
# Base settings
# -----------------------------------------------------------------------------
_C.BASE = ['']

_C.DIS = False
_C.WORLD_SIZE = 1
_C.SEED = 1234
_C.AMP = True
_C.EXPERIMENT_ID = ""
_C.SAVE_DIR = "save_pth"
_C.VAL_OUTPUT_PATH = "/home/lwt/code/flare/FLARE22-TwoStagePHTrans/save_results"
_C.COARSE_MODEL_PATH = "/home/lwt/code/flare/FLARE22-TwoStagePHTrans/save_pth/phtrans_c_220813_001000"
_C.FINE_MODEL_PATH = "/home/lwt/code/flare/FLARE22-TwoStagePHTrans/save_pth/phtrans_f_220813_001056"

# -----------------------------------------------------------------------------
# Wandb settings
# -----------------------------------------------------------------------------
_C.WANDB = CN()
_C.WANDB.COARSE_PROJECT = "FLARE2022_COARSE"
_C.WANDB.FINE_PROJECT = "FLARE2022_FINE"
_C.WANDB.TAG = "PHTrans"
_C.WANDB.MODE = "offline"

# -----------------------------------------------------------------------------
# Data settings
# -----------------------------------------------------------------------------
_C.DATASET = CN()
_C.DATASET.WITH_VAL = False
_C.DATASET.TRAIN_UNLABELED_IMAGE_PATH = "/home/lwt/data/flare22/UnlabeledCase"
_C.DATASET.TRAIN_UNLABELED_MASK_PATH = "/home/lwt/data/flare22/Unlabel2000_phtranPre"
_C.DATASET.TRAIN_IMAGE_PATH = "/home/lwt/data/flare22/Training/FLARE22_LabeledCase50/images"
_C.DATASET.TRAIN_MASK_PATH = "/home/lwt/data/flare22/Training/FLARE22_LabeledCase50/labels"
_C.DATASET.VAL_IMAGE_PATH = "/home/lwt/data/flare22/Validation"
_C.DATASET.EXTEND_SIZE = 20
_C.DATASET.IS_NORMALIZATION_DIRECTION = True

_C.DATASET.COARSE = CN()
_C.DATASET.COARSE.PROPRECESS_PATH = "/home/lwt/data_pro/flare22/Training/coarse_646464"
_C.DATASET.COARSE.PROPRECESS_UL_PATH = "/home/lwt/data_pro/flare22/Unlabel2000_coarse_646464"
_C.DATASET.COARSE.NUM_EACH_EPOCH = 512
_C.DATASET.COARSE.SIZE = [64, 64, 64]
_C.DATASET.COARSE.LABEL_CLASSES = 2

_C.DATASET.FINE = CN()
_C.DATASET.FINE.PROPRECESS_PATH = "/home/lwt/data_pro/flare22/Training/fine_96192192"
_C.DATASET.FINE.PROPRECESS_UL_PATH = "/home/lwt/data_pro/flare22/Unlabel2000_fine_96192192"
_C.DATASET.FINE.NUM_EACH_EPOCH = 512
_C.DATASET.FINE.SIZE = [96, 192, 192]
_C.DATASET.FINE.LABEL_CLASSES = 14

_C.DATASET.DA = CN()
_C.DATASET.DA.DO_2D_AUG = True
_C.DATASET.DA.DO_ELASTIC = True
_C.DATASET.DA.DO_SCALING = True
_C.DATASET.DA.DO_ROTATION = True
_C.DATASET.DA.RANDOM_CROP = False
_C.DATASET.DA.DO_GAMMA = True
_C.DATASET.DA.DO_MIRROR = False
_C.DATASET.DA.DO_ADDITIVE_BRIGHTNESS = True

# -----------------------------------------------------------------------------
# Dataloader settings
# -----------------------------------------------------------------------------
_C.DATALOADER = CN()
_C.DATALOADER.BATCH_SIZE = 1
_C.DATALOADER.PIN_MEMORY = True
_C.DATALOADER.NUM_WORKERS = 8

# -----------------------------------------------------------------------------
# Model settings
# -----------------------------------------------------------------------------
_C.MODEL = CN()
_C.MODEL.DEEP_SUPERVISION = True

_C.MODEL.COARSE = CN()
_C.MODEL.COARSE.TYPE = "phtrans"
_C.MODEL.COARSE.BASE_NUM_FEATURES = 16
_C.MODEL.COARSE.NUM_ONLY_CONV_STAGE = 2
_C.MODEL.COARSE.NUM_CONV_PER_STAGE = 2
_C.MODEL.COARSE.FEAT_MAP_MUL_ON_DOWNSCALE = 2
_C.MODEL.COARSE.POOL_OP_KERNEL_SIZES = [
    [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]]
_C.MODEL.COARSE.CONV_KERNEL_SIZES = [
    [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]]
_C.MODEL.COARSE.DROPOUT_P = 0.1

_C.MODEL.COARSE.MAX_NUM_FEATURES = 200
_C.MODEL.COARSE.DEPTHS = [2, 2, 2, 2]
_C.MODEL.COARSE.NUM_HEADS = [4, 4, 4, 4]
_C.MODEL.COARSE.WINDOW_SIZE = [4, 4, 4]
_C.MODEL.COARSE.MLP_RATIO = 1.
_C.MODEL.COARSE.QKV_BIAS = True
_C.MODEL.COARSE.QK_SCALE = None
_C.MODEL.COARSE.DROP_RATE = 0.
_C.MODEL.COARSE.DROP_PATH_RATE = 0.1

_C.MODEL.FINE = CN()
_C.MODEL.FINE.TYPE = "phtrans"
_C.MODEL.FINE.BASE_NUM_FEATURES = 16
_C.MODEL.FINE.NUM_ONLY_CONV_STAGE = 2
_C.MODEL.FINE.NUM_CONV_PER_STAGE = 2
_C.MODEL.FINE.FEAT_MAP_MUL_ON_DOWNSCALE = 2
_C.MODEL.FINE.POOL_OP_KERNEL_SIZES = [
    [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]]
_C.MODEL.FINE.CONV_KERNEL_SIZES = [[3, 3, 3], [
    3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]]
_C.MODEL.FINE.DROPOUT_P = 0.1

_C.MODEL.FINE.MAX_NUM_FEATURES = 200
_C.MODEL.FINE.DEPTHS = [2, 2, 2, 2]
_C.MODEL.FINE.NUM_HEADS = [4, 4, 4, 4]
_C.MODEL.FINE.WINDOW_SIZE = [3, 4, 4]
_C.MODEL.FINE.MLP_RATIO = 1.
_C.MODEL.FINE.QKV_BIAS = True
_C.MODEL.FINE.QK_SCALE = None
_C.MODEL.FINE.DROP_RATE = 0.
_C.MODEL.FINE.DROP_PATH_RATE = 0.1

# -----------------------------------------------------------------------------
# Training settings
# -----------------------------------------------------------------------------
_C.TRAIN = CN()
_C.TRAIN.DO_BACKPROP = True
_C.TRAIN.VAL_NUM_EPOCHS = 1
_C.TRAIN.SAVE_PERIOD = 1

_C.TRAIN.EPOCHS = 300
_C.TRAIN.WEIGHT_DECAY = 0.01
_C.TRAIN.WARMUP_EPOCHS = 20
_C.TRAIN.BASE_LR = 5e-4
_C.TRAIN.WARMUP_LR = 5e-7
_C.TRAIN.MIN_LR = 5e-6
# LR scheduler
_C.TRAIN.LR_SCHEDULER = CN()
_C.TRAIN.LR_SCHEDULER.NAME = 'cosine'

# Epoch interval to decay LR, used in StepLRScheduler
_C.TRAIN.LR_SCHEDULER.DECAY_EPOCHS = 30
# LR decay rate, used in StepLRScheduler
_C.TRAIN.LR_SCHEDULER.DECAY_RATE = 0.1

# Optimizer
_C.TRAIN.OPTIMIZER = CN()
_C.TRAIN.OPTIMIZER.NAME = 'adamw'
# Optimizer Epsilon
_C.TRAIN.OPTIMIZER.EPS = 1e-8
# Optimizer Betas
_C.TRAIN.OPTIMIZER.BETAS = (0.9, 0.999)
# SGD momentum
_C.TRAIN.OPTIMIZER.MOMENTUM = 0.9

# -----------------------------------------------------------------------------
# Test settings
# -----------------------------------------------------------------------------
_C.VAL = CN()
_C.VAL.IS_POST_PROCESS = True
_C.VAL.IS_WITH_DATALOADER = True


def _update_config_from_file(config, cfg_file):
    config.defrost()
    with open(cfg_file, 'r') as f:
        yaml_cfg = yaml.load(f, Loader=yaml.FullLoader)

    for cfg in yaml_cfg.setdefault('BASE', ['']):
        if cfg:
            _update_config_from_file(
                config, os.path.join(os.path.dirname(cfg_file), cfg)
            )
    print('=> merge config from {}'.format(cfg_file))
    config.merge_from_file(cfg_file)
    config.freeze()


def update_config(config, args):
    if args.cfg is not None:
        _update_config_from_file(config, args.cfg)

    config.defrost()
    if args.opts:
        config.merge_from_list(args.opts)
    if args.batch_size:
        config.DATALOADER.BATCH_SIZE = args.batch_size
    if args.tag:
        config.WANDB.TAG = args.tag
    if args.wandb_mode == "online":
        config.WANDB.MODE = args.wandb_mode
    if args.world_size:
        config.WORLD_SIZE = args.world_size
    if args.with_distributed:
        config.DIS = True
    config.freeze()


def update_val_config(config, args):
    if args.cfg is not None:
        _update_config_from_file(config, args.cfg)

    config.defrost()
    if args.opts:
        config.merge_from_list(args.opts)

    # merge from specific arguments
    if args.save_model_path:
        config.SAVE_MODEL_PATH = args.save_model_path
    if args.data_path:
        config.DATASET.VAL_IMAGE_PATH = args.data_path
    if args.output_path:
        config.VAL_OUTPUT_PATH = args.output_path

    config.freeze()


def get_config(args=None):
    config = _C.clone()
    update_config(config, args)

    return config


def get_config_no_args():
    config = _C.clone()

    return config


def get_val_config(args=None):
    config = _C.clone()
    update_val_config(config, args)

    return config
