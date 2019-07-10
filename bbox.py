from typing import List

import torch
from torch import Tensor


class BBox(object):

    def __init__(self, left: float, top: float, right: float, bottom: float):
        super().__init__()
        self.left = left
        self.top = top
        self.right = right
        self.bottom = bottom

    def __repr__(self) -> str:
        return 'BBox[l={:.1f}, t={:.1f}, r={:.1f}, b={:.1f}]'.format(
            self.left, self.top, self.right, self.bottom)

    def tolist(self) -> List[float]:
        return [self.left, self.top, self.right, self.bottom]

    @staticmethod
    def to_center_base(bboxes: Tensor) -> Tensor:
        return torch.stack([
            (bboxes[..., 0] + bboxes[..., 2]) / 2,
            (bboxes[..., 1] + bboxes[..., 3]) / 2,
            bboxes[..., 2] - bboxes[..., 0],
            bboxes[..., 3] - bboxes[..., 1]
        ], dim=-1)

    @staticmethod
    def from_center_base(center_based_bboxes: Tensor) -> Tensor:
        return torch.stack([
            center_based_bboxes[..., 0] - center_based_bboxes[..., 2] / 2,
            center_based_bboxes[..., 1] - center_based_bboxes[..., 3] / 2,
            center_based_bboxes[..., 0] + center_based_bboxes[..., 2] / 2,
            center_based_bboxes[..., 1] + center_based_bboxes[..., 3] / 2
        ], dim=-1)

    @staticmethod
    def calc_transformer(src_bboxes: Tensor, dst_bboxes: Tensor) -> Tensor:
        center_based_src_bboxes = BBox.to_center_base(src_bboxes)
        center_based_dst_bboxes = BBox.to_center_base(dst_bboxes)
        transformers = torch.stack([
            (center_based_dst_bboxes[..., 0] - center_based_src_bboxes[..., 0]) / center_based_src_bboxes[..., 2],
            (center_based_dst_bboxes[..., 1] - center_based_src_bboxes[..., 1]) / center_based_src_bboxes[..., 3],
            torch.log(center_based_dst_bboxes[..., 2] / center_based_src_bboxes[..., 2]),
            torch.log(center_based_dst_bboxes[..., 3] / center_based_src_bboxes[..., 3])
        ], dim=-1)
        return transformers

    @staticmethod
    def apply_transformer(src_bboxes: Tensor, transformers: Tensor) -> Tensor:
        center_based_src_bboxes = BBox.to_center_base(src_bboxes)
        center_based_dst_bboxes = torch.stack([
            transformers[..., 0] * center_based_src_bboxes[..., 2] + center_based_src_bboxes[..., 0],
            transformers[..., 1] * center_based_src_bboxes[..., 3] + center_based_src_bboxes[..., 1],
            torch.exp(transformers[..., 2]) * center_based_src_bboxes[..., 2],
            torch.exp(transformers[..., 3]) * center_based_src_bboxes[..., 3]
        ], dim=-1)
        dst_bboxes = BBox.from_center_base(center_based_dst_bboxes)
        return dst_bboxes

    @staticmethod
    def iou(source: Tensor, other: Tensor) -> Tensor:
        source, other = source.unsqueeze(dim=-2).repeat(1, 1, other.shape[-2], 1), \
                        other.unsqueeze(dim=-3).repeat(1, source.shape[-2], 1, 1)

        source_area = (source[..., 2] - source[..., 0]) * (source[..., 3] - source[..., 1])
        other_area = (other[..., 2] - other[..., 0]) * (other[..., 3] - other[..., 1])

        intersection_left = torch.max(source[..., 0], other[..., 0])
        intersection_top = torch.max(source[..., 1], other[..., 1])
        intersection_right = torch.min(source[..., 2], other[..., 2])
        intersection_bottom = torch.min(source[..., 3], other[..., 3])
        intersection_width = torch.clamp(intersection_right - intersection_left, min=0)
        intersection_height = torch.clamp(intersection_bottom - intersection_top, min=0)
        intersection_area = intersection_width * intersection_height

        return intersection_area / (source_area + other_area - intersection_area)

    @staticmethod
    def inside(bboxes: Tensor, left: float, top: float, right: float, bottom: float) -> Tensor:
        return ((bboxes[..., 0] >= left) * (bboxes[..., 1] >= top) *
                (bboxes[..., 2] <= right) * (bboxes[..., 3] <= bottom))

    @staticmethod
    def clip(bboxes: Tensor, left: float, top: float, right: float, bottom: float) -> Tensor:
        bboxes[..., [0, 2]] = bboxes[..., [0, 2]].clamp(min=left, max=right)
        bboxes[..., [1, 3]] = bboxes[..., [1, 3]].clamp(min=top, max=bottom)
        return bboxes
