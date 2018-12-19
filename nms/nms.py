import torch

from nms._ext import nms
from torch import Tensor


class NMS(object):

    @staticmethod
    def suppress(sorted_bboxes: Tensor, threshold: float) -> Tensor:
        kept_indices = torch.tensor([], dtype=torch.long).cuda()
        nms.suppress(sorted_bboxes.contiguous(), threshold, kept_indices)
        return kept_indices
