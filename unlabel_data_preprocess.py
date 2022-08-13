import os
import numpy as np
from batchgenerators.utilities.file_and_folder_operations import *
import shutil
import traceback
from multiprocessing import Pool, cpu_count
from config import get_config_no_args
from data.utils import crop_image_according_to_mask, load_data, clip_and_normalize_mean_std, resize_segmentation, change_axes_of_image, create_two_class_mask
from collections import OrderedDict
from skimage.transform import resize


def run_prepare_data(config, is_overwrite, is_multiprocessing=True):

    data_prepare = data_process(config, is_overwrite)
    if is_multiprocessing:
        pool = Pool(int(cpu_count() * 0.2))
        for data in data_prepare.data_list:
            try:
                pool.apply_async(data_prepare.process,  (data,))
            except Exception as err:
                traceback.print_exc()
                print('Create image/label throws exception %s, with series_id %s!' %
                      (err, data_prepare.data_info))

        pool.close()
        pool.join()
    else:
        for data in data_prepare.data_list:
            data_prepare.process(data)

class data_process(object):
    def __init__(self, config,  is_overwrite=False):
        self.config = config
        self.coarse_size = self.config.DATASET.COARSE.SIZE
        self.fine_size = self.config.DATASET.FINE.SIZE
        self.nor_dir = self.config.DATASET.IS_NORMALIZATION_DIRECTION
        self.extend_size = self.config.DATASET.EXTEND_SIZE

        self.image_path = config.DATASET.TRAIN_UNLABELED_IMAGE_PATH
        self.mask_path = config.DATASET.TRAIN_UNLABELED_MASK_PATH
        self.preprocess_coarse_path = config.DATASET.COARSE.PROPRECESS_UL_PATH
        self.preprocess_fine_path = config.DATASET.FINE.PROPRECESS_UL_PATH
        self.data_list = subfiles(self.image_path, join=False, suffix='nii.gz')
        if is_overwrite and isdir(self.preprocess_coarse_path):
            shutil.rmtree(self.preprocess_coarse_path)
        os.makedirs(self.preprocess_coarse_path, exist_ok=True)
        if is_overwrite and isdir(self.preprocess_fine_path):
            shutil.rmtree(self.preprocess_fine_path)
        os.makedirs(self.preprocess_fine_path, exist_ok=True)

    def process(self, image_id):

        data_id = image_id.split("_0000.nii.gz")[0]

        image, image_spacing, image_direction = load_data(
            join(self.image_path, data_id + "_0000.nii.gz"))
        mask, _, mask_direction = load_data(
            join(self.mask_path, data_id + ".nii.gz"))
        assert image_direction.all() == mask_direction.all()
        print(data_id, image.shape)
        if self.nor_dir:
            image = change_axes_of_image(image, image_direction)
            mask = change_axes_of_image(mask, mask_direction)
        data_info = OrderedDict()

        data_info["raw_shape"] = image.shape
        data_info["raw_spacing"] = image_spacing
        resize_spacing = image_spacing*image.shape/self.coarse_size
        data_info["resize_spacing"] = resize_spacing
        data_info["image_direction"] = image_direction
        with open(os.path.join(self.preprocess_coarse_path, "%s_info.pkl" % data_id), 'wb') as f:
            pickle.dump(data_info, f)

        image_resize = resize(image, self.coarse_size,
                              order=3, mode='edge', anti_aliasing=False)
        mask_resize = resize_segmentation(
            mask, self.coarse_size, order=0)
        mask_binary = create_two_class_mask(mask_resize)
        image_normal = clip_and_normalize_mean_std(image_resize)

        np.savez_compressed(os.path.join(self.preprocess_coarse_path, "%s.npz" %
                            data_id), data=image_normal[None, ...], seg=mask_binary[None, ...])

       
        margin = [int(self.extend_size / image_spacing[0]),
                  int(self.extend_size / image_spacing[1]),
                  int(self.extend_size / image_spacing[2])]
        crop_image, crop_mask = crop_image_according_to_mask(
            image, np.array(mask, dtype=int), margin)
        data_info_crop = OrderedDict()
        data_info_crop["raw_shape"] = image.shape
        data_info_crop["crop_shape"] = crop_image.shape
        data_info_crop["raw_spacing"] = image_spacing
        resize_crop_spacing = image_spacing*crop_image.shape/self.fine_size
        data_info_crop["resize_crop_spacing"] = resize_crop_spacing
        data_info_crop["image_direction"] = image_direction
        with open(os.path.join(self.preprocess_fine_path, "%s_info.pkl" % data_id), 'wb') as f:
            pickle.dump(data_info_crop, f)

        crop_image_resize = resize(
            crop_image, self.fine_size, order=3, mode='edge', anti_aliasing=False)
        crop_mask_resize = resize_segmentation(
            crop_mask, self.fine_size, order=0)
        crop_image_normal = clip_and_normalize_mean_std(crop_image_resize)
        np.savez_compressed(os.path.join(self.preprocess_fine_path, "%s.npz" % data_id),
                            data=crop_image_normal[None, ...], seg=crop_mask_resize[None, ...])

        print('End processing %s.' % data_id)

if __name__ == '__main__':
    config = get_config_no_args()
    run_prepare_data(config, False, False)
    
