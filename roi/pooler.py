from enum import Enum

import torch
from torch import Tensor
from torch.nn import functional as F

from support.layer.roi_align import ROIAlign


class Pooler(object):

    class Mode(Enum):
        POOLING = 'pooling'
        ALIGN = 'align'

    OPTIONS = ['pooling', 'align']

    @staticmethod
    def apply(features: Tensor, proposal_bboxes: Tensor, proposal_batch_indices: Tensor, mode: Mode) -> Tensor:
        _, _, feature_map_height, feature_map_width = features.shape
        scale = 1 / 16
        output_size = (7 * 2, 7 * 2)

        if mode == Pooler.Mode.POOLING:
            pool = []
            for (proposal_bbox, proposal_batch_index) in zip(proposal_bboxes, proposal_batch_indices):
                start_x = max(min(round(proposal_bbox[0].item() * scale), feature_map_width - 1), 0)      # [0, feature_map_width)
                start_y = max(min(round(proposal_bbox[1].item() * scale), feature_map_height - 1), 0)     # (0, feature_map_height]
                end_x = max(min(round(proposal_bbox[2].item() * scale) + 1, feature_map_width), 1)        # [0, feature_map_width)
                end_y = max(min(round(proposal_bbox[3].item() * scale) + 1, feature_map_height), 1)       # (0, feature_map_height]
                roi_feature_map = features[proposal_batch_index, :, start_y:end_y, start_x:end_x]
                pool.append(F.adaptive_max_pool2d(input=roi_feature_map, output_size=output_size))
            pool = torch.stack(pool, dim=0)
        elif mode == Pooler.Mode.ALIGN:
            pool = ROIAlign(output_size, spatial_scale=scale, sampling_ratio=0)(
                features,
                torch.cat([proposal_batch_indices.view(-1, 1).float(), proposal_bboxes], dim=1)
            )
        else:
            raise ValueError

        pool = F.max_pool2d(input=pool, kernel_size=2, stride=2)
        return pool

