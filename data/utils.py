import numpy as np
from batchgenerators.utilities.file_and_folder_operations import *
from skimage.transform import resize
import cv2
import shutil
import torch
import SimpleITK as sitk
import cc3d
import fastremap
import torch.nn.functional as F
from scipy.ndimage import binary_fill_holes

def load_pickle(file: str, mode: str = 'rb'):
    with open(file, mode) as f:
        a = pickle.load(f)
    return a


def load_data(data_path):

    data_itk = sitk.ReadImage(data_path)
    data_npy = sitk.GetArrayFromImage(data_itk)[None].astype(np.float32)
    data_spacing = np.array(data_itk.GetSpacing())[[2, 1, 0]]
    direction = data_itk.GetDirection()
    direction = np.array((direction[8], direction[4], direction[0]))
    return data_npy[0], data_spacing, direction


def change_axes_of_image(npy_image, orientation):
    if orientation[0] < 0:
        npy_image = np.flip(npy_image, axis=0)
    if orientation[1] > 0:
        npy_image = np.flip(npy_image, axis=1)
    if orientation[2] > 0:
        npy_image = np.flip(npy_image, axis=2)
    return npy_image


def clip_and_normalize_mean_std(image):
    mean = np.mean(image)
    std = np.std(image)

    image = (image - mean) / (std + 1e-5)
    return image


def resize_segmentation(segmentation, new_shape, order=3):
    tpe = segmentation.dtype
    unique_labels = np.unique(segmentation)
    assert len(segmentation.shape) == len(
        new_shape), "new shape must have same dimensionality as segmentation"
    if order == 0:
        return resize(segmentation.astype(float), new_shape, order, mode="edge", clip=True, anti_aliasing=False).astype(tpe)
    else:
        reshaped = np.zeros(new_shape, dtype=segmentation.dtype)

        for i, c in enumerate(unique_labels):
            mask = segmentation == c
            reshaped_multihot = resize(mask.astype(
                float), new_shape, order, mode="edge", clip=True, anti_aliasing=False)
            reshaped[reshaped_multihot >= 0.5] = c
        return reshaped

def maybe_to_torch(d):
    if isinstance(d, list):
        d = [maybe_to_torch(i) if not isinstance(
            i, torch.Tensor) else i for i in d]
    elif not isinstance(d, torch.Tensor):
        d = torch.from_numpy(d).float()
    return d


def create_two_class_mask(mask):

    mask = np.clip(mask, 0, 1)
    mask = binary_fill_holes(mask, origin=1,)
    return mask


def extract_topk_largest_candidates(npy_mask: np.array, label_unique, out_num_label: List) -> np.array:
    mask_shape = npy_mask.shape
    out_mask = np.zeros(
        [mask_shape[1], mask_shape[2], mask_shape[3]], np.uint8)
    for i in range(1, mask_shape[0]):
        t_mask = npy_mask[i].copy()
        keep_topk_largest_connected_object(
            t_mask, out_num_label, out_mask, label_unique[i])

    return out_mask


def keep_topk_largest_connected_object(npy_mask, k, out_mask, out_label):
    labels_out = cc3d.connected_components(npy_mask, connectivity=26)
    areas = {}
    for label, extracted in cc3d.each(labels_out, binary=True, in_place=True):
        areas[label] = fastremap.foreground(extracted)
    candidates = sorted(areas.items(), key=lambda item: item[1], reverse=True)
    
    for i in range(min(k, len(candidates))):
        out_mask[labels_out == int(candidates[i][0])] = out_label


def to_one_hot(seg, all_seg_labels=None):
    if all_seg_labels is None:
        all_seg_labels = np.unique(seg)
    result = np.zeros((len(all_seg_labels), *seg.shape), dtype=seg.dtype)
    for i, l in enumerate(all_seg_labels):
        result[i][seg == l] = 1
    return result


def input_downsample(x, input_size):
    x = F.interpolate(x, size=input_size, mode='trilinear',align_corners=False)
    mean = torch.mean(x)
    std = torch.std(x)
    x = (x - mean) / (1e-5 + std)
    return x


def output_upsample(x, output_size):
    x = F.interpolate(x, size=output_size,
                      mode='trilinear', align_corners=False)
    return x


def get_bbox_from_mask(mask, outside_value=0):
    mask_voxel_coords = np.where(mask != outside_value)
    minzidx = int(np.min(mask_voxel_coords[0]))
    maxzidx = int(np.max(mask_voxel_coords[0])) + 1
    minxidx = int(np.min(mask_voxel_coords[1]))
    maxxidx = int(np.max(mask_voxel_coords[1])) + 1
    minyidx = int(np.min(mask_voxel_coords[2]))
    maxyidx = int(np.max(mask_voxel_coords[2])) + 1
    return np.array([[minzidx, maxzidx], [minxidx, maxxidx], [minyidx, maxyidx]])


def crop_image_according_to_mask(npy_image, npy_mask, margin=None):
    if margin is None:
        margin = [20, 20, 20]

    bbox = get_bbox_from_mask(npy_mask)

    extend_bbox = np.concatenate(
        [np.max([[0, 0, 0], bbox[:, 0] - margin], axis=0)[:, np.newaxis],
         np.min([npy_image.shape, bbox[:, 1] + margin], axis=0)[:, np.newaxis]], axis=1)


    crop_mask = crop_to_bbox(npy_mask,extend_bbox)
    crop_image = crop_to_bbox(npy_image,extend_bbox)
       

    return crop_image, crop_mask


def crop_to_bbox(image, bbox):
    assert len(image.shape) == 3, "only supports 3d images"
    resizer = (slice(bbox[0][0], bbox[0][1]), slice(
        bbox[1][0], bbox[1][1]), slice(bbox[2][0], bbox[2][1]))
    return image[resizer]


def crop_image_according_to_bbox(npy_image, bbox, margin=None):
    if margin is None:
        margin = [20, 20, 20]

    image_shape = npy_image.shape
    extend_bbox = [[max(0, int(bbox[0][0]-margin[0])),
                   min(image_shape[0], int(bbox[0][1]+margin[0]))],
                   [max(0, int(bbox[1][0]-margin[1])),
                   min(image_shape[1], int(bbox[1][1]+margin[1]))],
                   [max(0, int(bbox[2][0]-margin[2])),
                   min(image_shape[2], int(bbox[2][1]+margin[2]))]]


    crop_image = crop_to_bbox(npy_image, extend_bbox)
   
  
    return crop_image, extend_bbox
