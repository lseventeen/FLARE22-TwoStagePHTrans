import numpy as np
import random
from torch.utils.data import Dataset
from batchgenerators.utilities.file_and_folder_operations import *
from .data_augmentation import default_3D_augmentation_params,default_2D_augmentation_params,get_patch_size,DownsampleSegForDSTransform
from batchgenerators.transforms.abstract_transforms import Compose
from batchgenerators.transforms.color_transforms import BrightnessMultiplicativeTransform, \
    ContrastAugmentationTransform, BrightnessTransform
from batchgenerators.transforms.color_transforms import GammaTransform
from batchgenerators.transforms.noise_transforms import GaussianNoiseTransform, GaussianBlurTransform
from batchgenerators.transforms.resample_transforms import SimulateLowResolutionTransform
from batchgenerators.transforms.spatial_transforms import SpatialTransform, MirrorTransform
from batchgenerators.transforms.utility_transforms import NumpyToTensor
from data.utils import load_pickle
class flare22_dataset(Dataset):
    def __init__(self, config, data_size, data_path,  unlab_data_path, pool_op_kernel_sizes, num_each_epoch,is_train=True, is_deep_supervision=True):
        self.config=config
        self.data_path = data_path
        self.data_size = data_size
        self.unlab_data_path = unlab_data_path
        self.pool_op_kernel_sizes = pool_op_kernel_sizes
        self.num_each_epoch = num_each_epoch
        self.series_ids = subfiles(data_path, join=False, suffix='npz')
        self.unlab_series_ids = subfiles(unlab_data_path, join=False, suffix='npz')
        self.setup_DA_params()
        
        self.transforms = self.get_augmentation(
                    data_size,
                    self.data_aug_params,is_train=is_train,
                    deep_supervision_scales=self.deep_supervision_scales if is_deep_supervision else None
                )
    def __getitem__(self, idx):
        if idx < len(self.series_ids):
            data_id = self.series_ids[idx]
            data_info = load_pickle(join(self.data_path, data_id.split(".")[0] + "_info.pkl"))
            data_load = np.load(join(self.data_path,data_id))
        else:
            data_id = self.unlab_series_ids[random.randint(0,len(self.unlab_series_ids)-1)]
            data_info = load_pickle(join(self.unlab_data_path, data_id.split(".")[0] + "_info.pkl"))
            data_load = np.load(join(self.unlab_data_path,data_id))
        
        data_trans = self.transforms(**data_load)
        return data_trans, data_info

    def __len__(self):
        return self.num_each_epoch
        
  

    def setup_DA_params(self):
        if self.config.MODEL.DEEP_SUPERVISION:
            self.deep_supervision_scales = [[1, 1, 1]] + list(list(i) for i in 1 / np.cumprod(np.vstack(self.pool_op_kernel_sizes), axis=0))[:-1]
        self.data_aug_params = default_3D_augmentation_params
        self.data_aug_params['rotation_x'] = (
                -30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi)
        self.data_aug_params['rotation_y'] = (
                -30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi)
        self.data_aug_params['rotation_z'] = (
                -30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi)

        if self.config.DATASET.DA.DO_2D_AUG:
            if self.config.DATASET.DA.DO_ELASTIC:
                self.data_aug_params["elastic_deform_alpha"] = \
                    default_2D_augmentation_params["elastic_deform_alpha"]
                self.data_aug_params["elastic_deform_sigma"] = \
                    default_2D_augmentation_params["elastic_deform_sigma"]
                self.data_aug_params["rotation_x"] = default_2D_augmentation_params["rotation_x"]
                
        if self.config.DATASET.DA.DO_2D_AUG:
            self.basic_generator_patch_size = get_patch_size(self.data_size[1:],
                                                             self.data_aug_params['rotation_x'],
                                                             self.data_aug_params['rotation_y'],
                                                             self.data_aug_params['rotation_z'],
                                                             self.data_aug_params['scale_range'])
            self.basic_generator_patch_size = np.array(
                [self.data_size[0]] + list(self.basic_generator_patch_size))
        else:
            self.basic_generator_patch_size = get_patch_size(self.data_size, self.data_aug_params['rotation_x'],
                                                             self.data_aug_params['rotation_y'],
                                                             self.data_aug_params['rotation_z'],
                                                             self.data_aug_params['scale_range'])



    def get_augmentation(self, patch_size, params=default_3D_augmentation_params,is_train=True,border_val_seg=-1,
                            order_seg=1, order_data=3, deep_supervision_scales=None,):
        transforms = []
        if is_train:
           
            if self.config.DATASET.DA.DO_2D_AUG:
                ignore_axes = (1,)
           
                patch_size_spatial = patch_size[1:]
            else:
                patch_size_spatial = patch_size
                ignore_axes = None

            transforms.append(SpatialTransform(
                patch_size_spatial, patch_center_dist_from_border=None,
                do_elastic_deform=self.config.DATASET.DA.DO_ELASTIC, alpha=params.get("elastic_deform_alpha"),
                sigma=params.get("elastic_deform_sigma"),
                do_rotation=self.config.DATASET.DA.DO_ROTATION, angle_x=params.get("rotation_x"), angle_y=params.get("rotation_y"),
                angle_z=params.get("rotation_z"), p_rot_per_axis=params.get("rotation_p_per_axis"),
                do_scale=self.config.DATASET.DA.DO_SCALING, scale=params.get("scale_range"),
                border_mode_data=params.get("border_mode_data"), border_cval_data=0, order_data=order_data,
                border_mode_seg="constant", border_cval_seg=border_val_seg,
                order_seg=order_seg, random_crop=self.config.DATASET.DA.RANDOM_CROP, p_el_per_sample=params.get("p_eldef"),
                p_scale_per_sample=params.get("p_scale"), p_rot_per_sample=params.get("p_rot"),
                independent_scale_for_each_axis=params.get("independent_scale_factor_for_each_axis")
    ))

      
            transforms.append(GaussianNoiseTransform(p_per_sample=0.1))
            transforms.append(GaussianBlurTransform((0.5, 1.), different_sigma_per_channel=True, p_per_sample=0.2,
                                               p_per_channel=0.5))
            transforms.append(BrightnessMultiplicativeTransform(multiplier_range=(0.75, 1.25), p_per_sample=0.15))

            if self.config.DATASET.DA.DO_ADDITIVE_BRIGHTNESS:
                transforms.append(BrightnessTransform(params.get("additive_brightness_mu"),
                                                 params.get("additive_brightness_sigma"),
                                                 True, p_per_sample=params.get("additive_brightness_p_per_sample"),
                                                 p_per_channel=params.get("additive_brightness_p_per_channel")))

            transforms.append(ContrastAugmentationTransform(p_per_sample=0.15))
            transforms.append(SimulateLowResolutionTransform(zoom_range=(0.5, 1), per_channel=True,
                                                        p_per_channel=0.5,
                                                        order_downsample=0, order_upsample=3, p_per_sample=0.25,
                                                        ignore_axes=ignore_axes))
            transforms.append(
                GammaTransform(params.get("gamma_range"), True, True, retain_stats=params.get("gamma_retain_stats"),
                       p_per_sample=0.1))  # inverted gamma

            if self.config.DATASET.DA.DO_GAMMA:
                transforms.append(
                    GammaTransform(params.get("gamma_range"), False, True, retain_stats=params.get("gamma_retain_stats"),
                           p_per_sample=params["p_gamma"]))

            if self.config.DATASET.DA.DO_MIRROR:
                transforms.append(MirrorTransform(params.get("mirror_axes")))

        if deep_supervision_scales is not None:
            transforms.append(DownsampleSegForDSTransform(deep_supervision_scales, 0, input_key='seg',
                                                               output_key='seg'))

        transforms.append(NumpyToTensor(['data', 'seg'], 'float'))
        transforms = Compose(transforms)
        return transforms

    