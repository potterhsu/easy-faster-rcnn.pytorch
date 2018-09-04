import torch

from nms._ext import nms


class NMS(object):

    @staticmethod
    def suppress(sorted_bboxes, threshold: float):
        keep_indices = torch.LongTensor().cuda()
        nms.suppress(sorted_bboxes.contiguous(), threshold, keep_indices)
        return keep_indices
