from torch.utils.data import DataLoader
from batchgenerators.utilities.file_and_folder_operations import *
from data.dataset_train import flare22_dataset
from sklearn.model_selection import train_test_split
from prefetch_generator import BackgroundGenerator
import torch
class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())

def build_loader(config,data_size, data_path,unlab_data_path, pool_op_kernel_sizes, num_each_epoch):
    series_ids_train = subfiles(data_path, join=False, suffix='npz')
    
    if config.DATASET.WITH_VAL:

        series_ids_train, series_ids_val = train_test_split(series_ids_train, test_size=config.DATASET.VAL_SPLIT,random_state=42)
        val_dataset = flare22_dataset(config,series_ids_val,data_size, data_path, pool_op_kernel_sizes,is_train=False)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset) if config.DIS else None
        val_loader = DataLoaderX(
            dataset=val_dataset,
            sampler=val_sampler ,
            batch_size = config.DATALOADER.BATCH_SIZE, 
            num_workers=config.DATALOADER.NUM_WORKERS,
            pin_memory= config.DATALOADER.PIN_MEMORY, 
            shuffle=False,
            drop_last=False
        )
    else:
        val_loader = None
    
    
    train_dataset = flare22_dataset(config, data_size, data_path, unlab_data_path,  pool_op_kernel_sizes, num_each_epoch,is_train=True)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset,shuffle=True) if config.DIS else None
    train_loader = DataLoaderX(
        train_dataset,
        sampler=train_sampler,
        batch_size = config.DATALOADER.BATCH_SIZE, 
        num_workers=config.DATALOADER.NUM_WORKERS,
        pin_memory= config.DATALOADER.PIN_MEMORY, 
        shuffle=True if train_sampler is None else False,
        drop_last=True
    )
    return train_loader,val_loader




