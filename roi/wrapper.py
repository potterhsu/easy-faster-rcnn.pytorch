from enum import Enum

import torch
from torch import Tensor
from torch.nn import functional as F

from roi.align.crop_and_resize import CropAndResizeFunction


class Wrapper(object):

    class Mode(Enum):
        POOLING = 'pooling'
        ALIGN = 'align'

    OPTIONS = ['pooling', 'align']

    @staticmethod
    def apply(features: Tensor, proposal_bboxes: Tensor, proposal_batch_indices: Tensor, mode: Mode) -> Tensor:
        _, _, feature_map_height, feature_map_width = features.shape

        if mode == Wrapper.Mode.POOLING:
            pool = []
            for proposal_bbox in proposal_bboxes:  # TODO: modiify for batch
                start_x = max(min(round(proposal_bbox[0].item() / 16), feature_map_width - 1), 0)      # [0, feature_map_width)
                start_y = max(min(round(proposal_bbox[1].item() / 16), feature_map_height - 1), 0)     # (0, feature_map_height]
                end_x = max(min(round(proposal_bbox[2].item() / 16) + 1, feature_map_width), 1)        # [0, feature_map_width)
                end_y = max(min(round(proposal_bbox[3].item() / 16) + 1, feature_map_height), 1)       # (0, feature_map_height]
                roi_feature_map = features[..., start_y:end_y, start_x:end_x]
                pool.append(F.adaptive_max_pool2d(input=roi_feature_map, output_size=7))
            pool = torch.cat(pool, dim=0)
        elif mode == Wrapper.Mode.ALIGN:
            x1 = proposal_bboxes[:, 0::4] / 16.0
            y1 = proposal_bboxes[:, 1::4] / 16.0
            x2 = proposal_bboxes[:, 2::4] / 16.0
            y2 = proposal_bboxes[:, 3::4] / 16.0

            crops = CropAndResizeFunction(crop_height=7 * 2, crop_width=7 * 2)(
                features,
                torch.cat([y1 / (feature_map_height - 1), x1 / (feature_map_width - 1),
                           y2 / (feature_map_height - 1), x2 / (feature_map_width - 1)],
                          dim=1),
                proposal_batch_indices.int()
            )
            pool = F.max_pool2d(input=crops, kernel_size=2, stride=2)
        else:
            raise ValueError

        return pool

