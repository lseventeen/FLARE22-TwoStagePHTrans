import argparse
from config import get_val_config
from models import build_coarse_model, build_fine_model
import os
import torch.backends.cudnn as cudnn
import numpy as np
import time
import torch
from torch.cuda.amp import autocast
import SimpleITK as sitk
from utils import to_cuda, load_checkpoint
from data import predict_dataset, DataLoaderX
from data.utils import change_axes_of_image, extract_topk_largest_candidates, to_one_hot, input_downsample, output_upsample, crop_image_according_to_bbox, get_bbox_from_mask
from batchgenerators.utilities.file_and_folder_operations import *
import torch.nn.functional as F

def parse_option():
    parser = argparse.ArgumentParser("FLARE2022_training")
    parser.add_argument('--cfg', type=str, metavar="FILE",
                        help='path to config file')
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )
    parser.add_argument('-smp', '--save_model_path', type=str,
                        default=None, help='path to model.pth')
    parser.add_argument('-dp', '--data_path', type=str,
                        default=None, help='path to validation image path')
    parser.add_argument('-op', '--output_path', type=str,
                        default=None, help='path to output image path')
    args = parser.parse_args()
    config = get_val_config(args)

    return args, config

class Inference:
    def __init__(self, config) -> None:
        self.config = config
        self.output_path = self.config.VAL_OUTPUT_PATH
        os.makedirs(config.VAL_OUTPUT_PATH, exist_ok=True)
        self.coarse_size = self.config.DATASET.COARSE.SIZE
        self.fine_size = self.config.DATASET.FINE.SIZE
        self.extend_size = self.config.DATASET.EXTEND_SIZE
        self.is_post_process = self.config.VAL.IS_POST_PROCESS
        self.is_nor_dir = self.config.DATASET.IS_NORMALIZATION_DIRECTION
        self.is_with_dataloader = self.config.VAL.IS_WITH_DATALOADER
        if self.is_with_dataloader:
            val_dataset = predict_dataset(config)
            self.val_loader = DataLoaderX(
                val_dataset,
                batch_size=1,
                num_workers=0,
                pin_memory=config.DATALOADER.PIN_MEMORY,
                shuffle=False,
            )
        else:
            self.val_loader = predict_dataset(config)
        cudnn.benchmark = True
       
    def run(self):
        torch.cuda.synchronize()
        t_start = time.time()
        with autocast():
            with torch.no_grad():
                for image_dict in self.val_loader:
                    image_dict = image_dict[0] if type(image_dict) is list else image_dict
                    if self.is_with_dataloader:
                        image_id = image_dict['image_id'][0]
                        raw_image = np.array(image_dict['raw_image'].squeeze(0))
                        raw_spacing = np.array(image_dict['raw_spacing'][0])
                        image_direction = np.array(image_dict['image_direction'][0])
                    else:
                        image_id = image_dict['image_id']
                        raw_image = image_dict['raw_image']
                        raw_spacing = image_dict['raw_spacing']
                        image_direction = image_dict['image_direction']
                    coarse_image = torch.from_numpy(
                        raw_image).unsqueeze(0).unsqueeze(0).float()
                    raw_image_shape = raw_image.shape
                    coarse_resize_factor = np.array(raw_image.shape) / np.array(self.coarse_size)
                    coarse_image = input_downsample(coarse_image, self.coarse_size)
                    coarse_image = self.coarse_predict(coarse_image, self.config.COARSE_MODEL_PATH)
                    coarse_pre = F.softmax(coarse_image, 1)
                    coarse_pre = coarse_pre.cpu().float()
                    torch.cuda.empty_cache()
                    coarse_mask = coarse_pre.argmax(1).squeeze(axis=0).numpy().astype(np.uint8)
                    lab_unique = np.unique(coarse_mask)
                    coarse_mask = to_one_hot(coarse_mask)
                    coarse_mask = extract_topk_largest_candidates(coarse_mask,lab_unique, 1)
                    coarse_bbox = get_bbox_from_mask(coarse_mask)
                    raw_bbox = [[int(coarse_bbox[0][0] * coarse_resize_factor[0]),
                                 int(coarse_bbox[0][1] * coarse_resize_factor[0])],
                                [int(coarse_bbox[1][0] * coarse_resize_factor[1]),
                                 int(coarse_bbox[1][1] * coarse_resize_factor[1])],
                                [int(coarse_bbox[2][0] * coarse_resize_factor[2]),
                                 int(coarse_bbox[2][1] * coarse_resize_factor[2])]]
                    margin = [self.extend_size / raw_spacing[i]
                              for i in range(3)]
                    crop_image, crop_fine_bbox = crop_image_according_to_bbox(
                        raw_image, raw_bbox, margin)
                    print(crop_fine_bbox)
                    crop_image_size = crop_image.shape
                    crop_image = torch.from_numpy(crop_image).unsqueeze(0).unsqueeze(0)
                    crop_image = input_downsample(crop_image, self.fine_size)
                    crop_image = self.fine_predict(crop_image, config.FINE_MODEL_PATH)
                    torch.cuda.empty_cache()
                    crop_image = output_upsample(crop_image, crop_image_size)
                    crop_image = F.softmax(crop_image, 1)
                    fine_mask = crop_image.argmax(1).squeeze(axis=0).numpy().astype(np.uint8)
                    if self.is_post_process:
                        lab_unique = np.unique(fine_mask)
                        fine_mask = to_one_hot(fine_mask)
                        fine_mask = extract_topk_largest_candidates(fine_mask,lab_unique, 1)
                    out_mask = np.zeros(raw_image_shape, np.uint8)
                    out_mask[crop_fine_bbox[0][0]:crop_fine_bbox[0][1],
                             crop_fine_bbox[1][0]:crop_fine_bbox[1][1],
                             crop_fine_bbox[2][0]:crop_fine_bbox[2][1]] = fine_mask
                    if self.is_nor_dir:
                        out_mask = change_axes_of_image(out_mask, image_direction)
                    sitk_image = sitk.GetImageFromArray(out_mask)
                    sitk.WriteImage(sitk_image, os.path.join(
                        self.output_path, "FLARETs_{}.nii.gz".format(image_id)), True)
                    print(f"{image_id} Done")

        torch.cuda.synchronize()
        t_end = time.time()
        average_time_usage = (t_end - t_start) * 1.0 / len(self.val_loader)
        print("Average time usage: {} s".format(average_time_usage))

    def coarse_predict(self, input, model_path):
        coarse_model_checkpoint = load_checkpoint(model_path)
        coarse_model = build_coarse_model(coarse_model_checkpoint["config"], True).eval()
        coarse_model.load_state_dict({k.replace('module.', ''): v for k, v in coarse_model_checkpoint['state_dict'].items()})
        self._set_requires_grad(coarse_model, False)
        coarse_model = coarse_model.cuda().half()
        input = to_cuda(input).half()
        out = coarse_model(input)
        coarse_model = coarse_model.cpu()
        return out.cpu().float()

    def fine_predict(self, input, model_path):
        fine_model_checkpoint = load_checkpoint(model_path)
        fine_model = build_fine_model(fine_model_checkpoint["config"], True).eval()
        fine_model.load_state_dict({k.replace('module.', ''): v for k, v in fine_model_checkpoint['state_dict'].items()})
        self._set_requires_grad(fine_model, False)
        fine_model = fine_model.cuda().half()
        input = to_cuda(input).half()
        out = fine_model(input)
        fine_model = fine_model.cpu()
        return out.cpu().float()

    @staticmethod
    def _set_requires_grad(model, requires_grad=False):
        for param in model.parameters():
            param.requires_grad = requires_grad

if __name__ == '__main__':
    torch.cuda.synchronize()
    t_start = time.time()
    _, config = parse_option()

    predict = Inference(config)
    predict.run()
    torch.cuda.synchronize()
    t_end = time.time()
    total_time = t_end - t_start
    print("Total_time: {} s".format(total_time))
 