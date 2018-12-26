from typing import Tuple, List

import numpy as np
import torch
from torch import nn, Tensor
from torch.nn import functional as F

from bbox import BBox
from nms.nms import NMS


class RegionProposalNetwork(nn.Module):

    def __init__(self, num_features_out: int, anchor_ratios: List[Tuple[int, int]], anchor_sizes: List[int], pre_nms_top_n: int, post_nms_top_n: int):
        super().__init__()

        self._features = nn.Sequential(
            nn.Conv2d(in_channels=num_features_out, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU()
        )

        self._anchor_ratios = anchor_ratios
        self._anchor_sizes = anchor_sizes

        num_anchor_ratios = len(self._anchor_ratios)
        num_anchor_sizes = len(self._anchor_sizes)
        num_anchors = num_anchor_ratios * num_anchor_sizes

        self._pre_nms_top_n = pre_nms_top_n
        self._post_nms_top_n = post_nms_top_n

        self._objectness = nn.Conv2d(in_channels=512, out_channels=num_anchors * 2, kernel_size=1)
        self._transformer = nn.Conv2d(in_channels=512, out_channels=num_anchors * 4, kernel_size=1)

    def forward(self, features: Tensor, image_width: int, image_height: int) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        anchor_bboxes = self._generate_anchors(image_width, image_height,
                                               num_x_anchors=features.shape[3], num_y_anchors=features.shape[2],
                                               anchor_ratios=self._anchor_ratios, anchor_sizes=self._anchor_sizes).cuda()

        features = self._features(features)
        objectnesses = self._objectness(features)
        transformers = self._transformer(features)

        objectnesses = objectnesses.permute(0, 2, 3, 1).contiguous().view(-1, 2)
        transformers = transformers.permute(0, 2, 3, 1).contiguous().view(-1, 4)

        proposal_bboxes = self._generate_proposals(anchor_bboxes, objectnesses, transformers, image_width, image_height)

        proposal_bboxes = proposal_bboxes[:self._pre_nms_top_n]
        kept_indices = NMS.suppress(proposal_bboxes, threshold=0.7)
        proposal_bboxes = proposal_bboxes[kept_indices]
        proposal_bboxes = proposal_bboxes[:self._post_nms_top_n]

        return anchor_bboxes, objectnesses, transformers, proposal_bboxes

    def sample(self, anchor_bboxes: Tensor, anchor_objectnesses: Tensor, anchor_transformers: Tensor, gt_bboxes: Tensor,
               image_width: int, image_height: int) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        anchor_bboxes = anchor_bboxes.cpu()
        gt_bboxes = gt_bboxes.cpu()

        # remove cross-boundary
        boundary = torch.tensor(BBox(0, 0, image_width, image_height).tolist(), dtype=torch.float)
        inside_indices = BBox.inside(anchor_bboxes, boundary.unsqueeze(dim=0)).squeeze().nonzero().view(-1)

        anchor_bboxes = anchor_bboxes[inside_indices]
        anchor_objectnesses = anchor_objectnesses[inside_indices]
        anchor_transformers = anchor_transformers[inside_indices]

        # find labels for each `anchor_bboxes`
        labels = torch.ones(len(anchor_bboxes), dtype=torch.long) * -1
        ious = BBox.iou(anchor_bboxes, gt_bboxes)
        anchor_max_ious, anchor_assignments = ious.max(dim=1)
        gt_max_ious, gt_assignments = ious.max(dim=0)
        anchor_additions = (ious == gt_max_ious).nonzero()[:, 0]
        labels[anchor_max_ious < 0.3] = 0
        labels[anchor_additions] = 1
        labels[anchor_max_ious >= 0.7] = 1

        # select 256 samples
        fg_indices = (labels == 1).nonzero().view(-1)
        bg_indices = (labels == 0).nonzero().view(-1)
        fg_indices = fg_indices[torch.randperm(len(fg_indices))[:min(len(fg_indices), 128)]]
        bg_indices = bg_indices[torch.randperm(len(bg_indices))[:256 - len(fg_indices)]]
        selected_indices = torch.cat([fg_indices, bg_indices])
        selected_indices = selected_indices[torch.randperm(len(selected_indices))]

        gt_anchor_objectnesses = labels[selected_indices]
        gt_bboxes = gt_bboxes[anchor_assignments[fg_indices]]
        anchor_bboxes = anchor_bboxes[fg_indices]
        gt_anchor_transformers = BBox.calc_transformer(anchor_bboxes, gt_bboxes)

        gt_anchor_objectnesses = gt_anchor_objectnesses.cuda()
        gt_anchor_transformers = gt_anchor_transformers.cuda()

        anchor_objectnesses = anchor_objectnesses[selected_indices]
        anchor_transformers = anchor_transformers[fg_indices]

        return anchor_objectnesses, anchor_transformers, gt_anchor_objectnesses, gt_anchor_transformers

    def loss(self, anchor_objectnesses: Tensor, anchor_transformers: Tensor, gt_anchor_objectnesses: Tensor, gt_anchor_transformers: Tensor) -> Tuple[Tensor, Tensor]:
        cross_entropy = F.cross_entropy(input=anchor_objectnesses, target=gt_anchor_objectnesses)

        # NOTE: The default of `reduction` is `elementwise_mean`, which is divided by N x 4 (number of all elements), here we replaced by N for better performance
        smooth_l1_loss = F.smooth_l1_loss(input=anchor_transformers, target=gt_anchor_transformers, reduction='sum')
        smooth_l1_loss /= len(gt_anchor_transformers)

        return cross_entropy, smooth_l1_loss

    def _generate_anchors(self, image_width: int, image_height: int, num_x_anchors: int, num_y_anchors: int, anchor_ratios: List[Tuple[int, int]], anchor_sizes: List[int]) -> Tensor:
        center_ys = np.linspace(start=0, stop=image_height, num=num_y_anchors + 2)[1:-1]
        center_xs = np.linspace(start=0, stop=image_width, num=num_x_anchors + 2)[1:-1]
        ratios = np.array(anchor_ratios)
        ratios = ratios[:, 0] / ratios[:, 1]
        sizes = np.array(anchor_sizes)

        # NOTE: it's important to let `center_ys` be the major index (i.e., move horizontally and then vertically) for consistency with 2D convolution

        # giving the string 'ij' returns a meshgrid with matrix indexing, i.e., with shape (#center_ys, #center_xs, #ratios)
        center_ys, center_xs, ratios, sizes = np.meshgrid(center_ys, center_xs, ratios, sizes, indexing='ij')

        center_ys = center_ys.reshape(-1)
        center_xs = center_xs.reshape(-1)
        ratios = ratios.reshape(-1)
        sizes = sizes.reshape(-1)

        widths = sizes * np.sqrt(1 / ratios)
        heights = sizes * np.sqrt(ratios)

        center_based_anchor_bboxes = np.stack((center_xs, center_ys, widths, heights), axis=1)
        center_based_anchor_bboxes = torch.from_numpy(center_based_anchor_bboxes).float()
        anchor_bboxes = BBox.from_center_base(center_based_anchor_bboxes)

        return anchor_bboxes

    def _generate_proposals(self, anchor_bboxes: Tensor, objectnesses: Tensor, transformers: Tensor, image_width: int, image_height: int) -> Tensor:
        proposal_score = objectnesses[:, 1]
        _, sorted_indices = torch.sort(proposal_score, dim=0, descending=True)

        sorted_transformers = transformers[sorted_indices]
        sorted_anchor_bboxes = anchor_bboxes[sorted_indices]

        proposal_bboxes = BBox.apply_transformer(sorted_anchor_bboxes, sorted_transformers.detach())
        proposal_bboxes = BBox.clip(proposal_bboxes, 0, 0, image_width, image_height)

        return proposal_bboxes
