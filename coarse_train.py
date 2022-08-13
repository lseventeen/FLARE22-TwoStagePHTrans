import argparse
from loguru import logger
from data import build_loader
from trainer import Trainer
from utils import seed_torch
from losses import build_loss
from datetime import datetime
import wandb
from config import get_config
from models import build_coarse_model
from lr_scheduler import build_scheduler
from optimizer import build_optimizer
import os
import torch.backends.cudnn as cudnn
import numpy as np
import torch
import torch.multiprocessing as mp
import torch.distributed as dist


def parse_option():
    parser = argparse.ArgumentParser("FLARE2022_corase_training")
    parser.add_argument('--cfg', type=str, metavar="FILE",
                        help='path to config file')
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )
    parser.add_argument("--tag", help='tag of experiment')
    parser.add_argument("-wm", "--wandb_mode", default="offline")
    parser.add_argument('-bs', '--batch-size', type=int,
                        help="batch size for single GPU")
    parser.add_argument('-wd', '--with_distributed', help="training without DDP",
                        required=False, default=False, action="store_true")
    parser.add_argument('-ws', '--world_size', type=int,
                        help="process number for DDP")
    args = parser.parse_args()
    config = get_config(args)

    return args, config


def main(config):
    if config.DIS:
        mp.spawn(main_worker,
                 args=(config,),
                 nprocs=config.WORLD_SIZE,)
    else:
        main_worker(0, config)


def main_worker(local_rank, config):
    if local_rank == 0:
        config.defrost()
        config.EXPERIMENT_ID = f"{config.WANDB.TAG}_{datetime.now().strftime('%y%m%d_%H%M%S')}"
        config.freeze()
        wandb.init(project=config.WANDB.COARSE_PROJECT,
                   name=config.EXPERIMENT_ID, config=config, mode=config.WANDB.MODE)
    np.set_printoptions(formatter={'float': '{: 0.4f}'.format}, suppress=True)
    torch.cuda.set_device(local_rank)
    if config.DIS:
        dist.init_process_group(
            "nccl", init_method='env://', rank=local_rank, world_size=config.WORLD_SIZE)
    seed = config.SEED + local_rank
    seed_torch(seed)
    cudnn.benchmark = True

    train_loader, val_loader = build_loader(config,
                                            config.DATASET.COARSE.SIZE,
                                            config.DATASET.COARSE.PROPRECESS_PATH,
                                            config.DATASET.COARSE.PROPRECESS_UL_PATH,
                                            config.MODEL.COARSE.POOL_OP_KERNEL_SIZES,
                                            config.DATASET.COARSE.NUM_EACH_EPOCH
                                            )
    model = build_coarse_model(config).cuda()
    if config.DIS:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], find_unused_parameters=True)
    logger.info(f'\n{model}\n')
    loss = build_loss(config.MODEL.DEEP_SUPERVISION,
                      config.MODEL.COARSE.POOL_OP_KERNEL_SIZES)
    optimizer = build_optimizer(config, model)
    lr_scheduler = build_scheduler(config, optimizer, len(train_loader))
    trainer = Trainer(config=config,
                      train_loader=train_loader,
                      val_loader=val_loader,
                      model=model,
                      loss=loss,
                      optimizer=optimizer,
                      lr_scheduler=lr_scheduler)
    trainer.train()


if __name__ == '__main__':
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "10000"
    _, config = parse_option()

    main(config)
