import numpy as np
import torch
import random
import os
from batchgenerators.utilities.file_and_folder_operations import *


def seed_torch(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def to_cuda(data, non_blocking=True):
    if isinstance(data, list):
        data = [i.cuda(non_blocking=non_blocking) for i in data]
    else:
        data = data.cuda(non_blocking=non_blocking)
    return data


def load_checkpoint(checkpoint_path):
    checkpoint_file = "final_checkpoint.pth"
    checkpoint = torch.load(
        join(checkpoint_path, checkpoint_file), map_location=torch.device('cpu'))
    return checkpoint
