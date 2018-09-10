import torch

from nms._ext import nms
from torch import Tensor


class NMS(object):

    @staticmethod
    def suppress(sorted_bboxes: Tensor, threshold: float) -> Tensor:
        keep_indices = torch.tensor([], dtype=torch.long).cuda()
        nms.suppress(sorted_bboxes.contiguous(), threshold, keep_indices)
        return keep_indices
