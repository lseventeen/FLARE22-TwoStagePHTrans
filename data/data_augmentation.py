from copy import deepcopy
import numpy as np
from batchgenerators.augmentations.utils import resize_segmentation
from batchgenerators.transforms.abstract_transforms import AbstractTransform

default_3D_augmentation_params = {

    # "do_elastic": False,
    "elastic_deform_alpha": (0., 900.),
    "elastic_deform_sigma": (9., 13.),
    "p_eldef": 0.2,

    # "do_scaling": True,
    "scale_range": (0.85, 1.25),
    "independent_scale_factor_for_each_axis": False,
    "p_independent_scale_per_axis": 1,
    "p_scale": 0.2,

    # "do_rotation": True,
    "rotation_x": (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi),
    "rotation_y": (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi),
    "rotation_z": (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi),
    "rotation_p_per_axis": 1,
    "p_rot": 0.2,

    # "random_crop": False,
    "random_crop_dist_to_border": None,

    # "do_gamma": True,
    "gamma_retain_stats": True,
    "gamma_range": (0.7, 1.5),
    "p_gamma": 0.3,

    # "do_mirror": True,
    "mirror_axes": (0, 1, 2),

    "border_mode_data": "constant",

    # "do_additive_brightness": False,
    "additive_brightness_p_per_sample": 0.15,
    "additive_brightness_p_per_channel": 0.5,
    "additive_brightness_mu": 0.0,
    "additive_brightness_sigma": 0.1
}

default_2D_augmentation_params = deepcopy(default_3D_augmentation_params)

default_2D_augmentation_params["elastic_deform_alpha"] = (0., 200.)
default_2D_augmentation_params["elastic_deform_sigma"] = (9., 13.)
default_2D_augmentation_params["rotation_x"] = (
    -180. / 360 * 2. * np.pi, 180. / 360 * 2. * np.pi)
default_2D_augmentation_params["rotation_y"] = (
    -0. / 360 * 2. * np.pi, 0. / 360 * 2. * np.pi)
default_2D_augmentation_params["rotation_z"] = (
    -0. / 360 * 2. * np.pi, 0. / 360 * 2. * np.pi)


def get_patch_size(final_patch_size, rot_x, rot_y, rot_z, scale_range):
    if isinstance(rot_x, (tuple, list)):
        rot_x = max(np.abs(rot_x))
    if isinstance(rot_y, (tuple, list)):
        rot_y = max(np.abs(rot_y))
    if isinstance(rot_z, (tuple, list)):
        rot_z = max(np.abs(rot_z))
    rot_x = min(90 / 360 * 2. * np.pi, rot_x)
    rot_y = min(90 / 360 * 2. * np.pi, rot_y)
    rot_z = min(90 / 360 * 2. * np.pi, rot_z)
    from batchgenerators.augmentations.utils import rotate_coords_3d, rotate_coords_2d
    coords = np.array(final_patch_size)
    final_shape = np.copy(coords)
    if len(coords) == 3:
        final_shape = np.max(
            np.vstack((np.abs(rotate_coords_3d(coords, rot_x, 0, 0)), final_shape)), 0)
        final_shape = np.max(
            np.vstack((np.abs(rotate_coords_3d(coords, 0, rot_y, 0)), final_shape)), 0)
        final_shape = np.max(
            np.vstack((np.abs(rotate_coords_3d(coords, 0, 0, rot_z)), final_shape)), 0)
    elif len(coords) == 2:
        final_shape = np.max(
            np.vstack((np.abs(rotate_coords_2d(coords, rot_x)), final_shape)), 0)
    final_shape /= min(scale_range)
    return final_shape.astype(int)


class DownsampleSegForDSTransform(AbstractTransform):

    def __init__(self, ds_scales=(1, 0.5, 0.25), order=0, input_key="seg", output_key="seg", axes=None):
        self.axes = axes
        self.output_key = output_key
        self.input_key = input_key
        self.order = order
        self.ds_scales = ds_scales

    def __call__(self, **data_dict):
        data_dict[self.output_key] = downsample_seg_for_ds_transform(data_dict[self.input_key], self.ds_scales,
                                                                     self.order, self.axes)
        return data_dict


def downsample_seg_for_ds_transform(seg, ds_scales=((1, 1, 1), (0.5, 0.5, 0.5), (0.25, 0.25, 0.25)), order=0, axes=None):
    if axes is None:
        axes = list(range(1, len(seg.shape)))
    output = []
    for s in ds_scales:
        if all([i == 1 for i in s]):
            output.append(seg)
        else:
            new_shape = np.array(seg.shape).astype(float)
            for i, a in enumerate(axes):
                new_shape[a] *= s[i]
            new_shape = np.round(new_shape).astype(int)
            out_seg = np.zeros(new_shape, dtype=seg.dtype)
            for c in range(seg.shape[0]):
                out_seg[c] = resize_segmentation(seg[c], new_shape[1:], order)
            output.append(out_seg)
    return output
