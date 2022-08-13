import numpy as np
import torch
import torch.nn.functional as F

class AverageMeter(object):
    def __init__(self):
        self.initialized = False
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None

    def initialize(self, val, weight):
        self.val = val
        self.avg = val
        self.sum = np.multiply(val, weight)
        self.count = weight
        self.initialized = True

    def update(self, val, weight=1):
        if not self.initialized:
            self.initialize(val, weight)
        else:
            self.add(val, weight)

    def add(self, val, weight):
        self.val = val
        self.sum = np.add(self.sum, np.multiply(val, weight))
        self.count = self.count + weight
        self.avg = self.sum / self.count

    @property
    def value(self):
        return np.round(self.val, 4)

    @property
    def average(self):
        return np.round(self.avg, 4)


def run_online_evaluation(output, target):
    if isinstance(output, list):
        output = output[0]
    if isinstance(target, list):
        target = target[0]
    online_eval_foreground_dc = []
    online_eval_tp = []
    online_eval_fp = []
    online_eval_fn = []
    with torch.no_grad():
        num_classes = output.shape[1]
        output_softmax = F.softmax(output, 1)
        output_seg = output_softmax.argmax(1)
        target = target[:, 0]
        axes = tuple(range(1, len(target.shape)))
        tp_hard = torch.zeros(
            (target.shape[0], num_classes - 1)).to(output_seg.device.index)
        fp_hard = torch.zeros(
            (target.shape[0], num_classes - 1)).to(output_seg.device.index)
        fn_hard = torch.zeros(
            (target.shape[0], num_classes - 1)).to(output_seg.device.index)
        for c in range(1, num_classes):
            tp_hard[:, c - 1] = sum_tensor(
                (output_seg == c).float() * (target == c).float(), axes=axes)
            fp_hard[:, c - 1] = sum_tensor(
                (output_seg == c).float() * (target != c).float(), axes=axes)
            fn_hard[:, c - 1] = sum_tensor(
                (output_seg != c).float() * (target == c).float(), axes=axes)

        tp_hard = tp_hard.sum(0, keepdim=False).detach().cpu().numpy()
        fp_hard = fp_hard.sum(0, keepdim=False).detach().cpu().numpy()
        fn_hard = fn_hard.sum(0, keepdim=False).detach().cpu().numpy()

        online_eval_foreground_dc.append(
            list((2 * tp_hard) / (2 * tp_hard + fp_hard + fn_hard + 1e-8)))
        online_eval_tp.append(list(tp_hard))
        online_eval_fp.append(list(fp_hard))
        online_eval_fn.append(list(fn_hard))

        online_eval_tp = np.sum(online_eval_tp, 0)
        online_eval_fp = np.sum(online_eval_fp, 0)
        online_eval_fn = np.sum(online_eval_fn, 0)

        global_dc_per_class = [i for i in [2 * i / (2 * i + j + k) for i, j, k in
                                           zip(online_eval_tp, online_eval_fp, online_eval_fn)]
                               if not np.isnan(i)]
        average_global_dc = np.mean(global_dc_per_class)
    return average_global_dc


def sum_tensor(inp, axes, keepdim=False):
    axes = np.unique(axes).astype(int)
    if keepdim:
        for ax in axes:
            inp = inp.sum(int(ax), keepdim=True)
    else:
        for ax in sorted(axes, reverse=True):
            inp = inp.sum(int(ax))
    return inp
