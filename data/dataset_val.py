import numpy as np
from torch.utils.data import Dataset
from batchgenerators.utilities.file_and_folder_operations import *
from .utils import load_data,change_axes_of_image

class predict_dataset(Dataset):
    def __init__(self, config):
        super(predict_dataset, self).__init__()
        self.config = config
        self.data_path = config.DATASET.VAL_IMAGE_PATH
     
        self.is_nor_dir = self.config.DATASET.IS_NORMALIZATION_DIRECTION

        self.series_ids = subfiles(self.data_path, join=False, suffix='gz')
    def __len__(self):
        return len(self.series_ids)

    def __getitem__(self, idx):
        image_id = self.series_ids[idx].split("_")[1]
        raw_image, image_spacing, image_direction= load_data(join(self.data_path,self.series_ids[idx]))
        if self.is_nor_dir:
            raw_image = change_axes_of_image(raw_image, image_direction)
        return {'image_id': image_id,
                'raw_image': np.ascontiguousarray(raw_image),
                'raw_spacing': image_spacing,
                'image_direction': image_direction
                }
              
     