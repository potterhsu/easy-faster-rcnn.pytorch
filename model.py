import os
import time
from typing import Union, Tuple, List

import torch
from torch import nn, Tensor
from torch.nn import functional as F

from backbone.interface import Interface as BackboneInterface
from bbox import BBox
from nms.nms import NMS
from rpn.region_proposal_network import RegionProposalNetwork


class Model(nn.Module):

    NUM_CLASSES = 21

    class ForwardInput:
        class Train(object):
            def __init__(self, image: Tensor, gt_classes: Tensor, gt_bboxes: Tensor):
                self.image = image
                self.gt_classes = gt_classes
                self.gt_bboxes = gt_bboxes

        class Eval(object):
            def __init__(self, image: Tensor):
                self.image = image

    class ForwardOutput:
        class Train(object):
            def __init__(self, anchor_objectness_loss: Tensor, anchor_transformer_loss: Tensor, proposal_class_loss: Tensor, proposal_transformer_loss: Tensor):
                self.anchor_objectness_loss = anchor_objectness_loss
                self.anchor_transformer_loss = anchor_transformer_loss
                self.proposal_class_loss = proposal_class_loss
                self.proposal_transformer_loss = proposal_transformer_loss

        class Eval(object):
            def __init__(self, proposal_bboxes: Tensor, proposal_classes: Tensor, proposal_transformers: Tensor):
                self.proposal_bboxes = proposal_bboxes
                self.proposal_classes = proposal_classes
                self.proposal_transformers = proposal_transformers

    def __init__(self, backbone: BackboneInterface):
        super().__init__()

        self.features = backbone.features()
        self._bn_modules = [it for it in self.features.modules() if isinstance(it, nn.BatchNorm2d)]

        self.rpn = RegionProposalNetwork()
        self.detection = Model.Detection()

        self._transformer_normalize_mean = torch.tensor([0., 0., 0., 0.], dtype=torch.float)
        self._transformer_normalize_std = torch.tensor([.1, .1, .2, .2], dtype=torch.float)

    def forward(self, forward_input: Union[ForwardInput.Train, ForwardInput.Eval]) -> Union[ForwardOutput.Train, ForwardOutput.Eval]:
        for bn_module in self._bn_modules:
            bn_module.eval()
            for parameter in bn_module.parameters():
                parameter.requires_grad = False

        image = forward_input.image.unsqueeze(dim=0)
        image_height, image_width = image.shape[2], image.shape[3]

        features = self.features(image)
        anchor_bboxes, anchor_objectnesses, anchor_transformers, proposal_bboxes = self.rpn.forward(features, image_width, image_height)

        if self.training:
            forward_input: Model.ForwardInput.Train

            anchor_objectnesses, anchor_transformers, gt_anchor_objectnesses, gt_anchor_transformers = self.rpn.sample(anchor_bboxes, anchor_objectnesses, anchor_transformers, forward_input.gt_bboxes, image_width, image_height)
            anchor_objectness_loss, anchor_transformer_loss = self.rpn.loss(anchor_objectnesses, anchor_transformers, gt_anchor_objectnesses, gt_anchor_transformers)

            proposal_bboxes, gt_proposal_classes, gt_proposal_transformers = self.sample(proposal_bboxes, forward_input.gt_classes, forward_input.gt_bboxes)
            proposal_classes, proposal_transformers = self.detection.forward(features, proposal_bboxes)
            proposal_class_loss, proposal_transformer_loss = self.loss(proposal_classes, proposal_transformers, gt_proposal_classes, gt_proposal_transformers)

            forward_output = Model.ForwardOutput.Train(anchor_objectness_loss, anchor_transformer_loss, proposal_class_loss, proposal_transformer_loss)
        else:
            proposal_classes, proposal_transformers = self.detection.forward(features, proposal_bboxes)
            forward_output = Model.ForwardOutput.Eval(proposal_bboxes, proposal_classes, proposal_transformers)

        return forward_output

    def sample(self, proposal_bboxes: Tensor, gt_classes: Tensor, gt_bboxes: Tensor):
        proposal_bboxes = proposal_bboxes.cpu()
        gt_classes = gt_classes.cpu()
        gt_bboxes = gt_bboxes.cpu()

        # find labels for each `proposal_bboxes`
        labels = torch.ones(len(proposal_bboxes), dtype=torch.long) * -1
        ious = BBox.iou(proposal_bboxes, gt_bboxes)
        proposal_max_ious, proposal_assignments = ious.max(dim=1)
        labels[proposal_max_ious < 0.5] = 0
        labels[proposal_max_ious >= 0.5] = gt_classes[proposal_assignments[proposal_max_ious >= 0.5]]

        # select 128 samples
        fg_indices = (labels > 0).nonzero().view(-1)
        bg_indices = (labels == 0).nonzero().view(-1)
        fg_indices = fg_indices[torch.randperm(len(fg_indices))[:min(len(fg_indices), 32)]]
        bg_indices = bg_indices[torch.randperm(len(bg_indices))[:128 - len(fg_indices)]]
        select_indices = torch.cat([fg_indices, bg_indices])
        select_indices = select_indices[torch.randperm(len(select_indices))]

        proposal_bboxes = proposal_bboxes[select_indices]
        gt_proposal_transformers = BBox.calc_transformer(proposal_bboxes, gt_bboxes[proposal_assignments[select_indices]])
        gt_proposal_classes = labels[select_indices]

        gt_proposal_transformers = (gt_proposal_transformers - self._transformer_normalize_mean) / self._transformer_normalize_std

        gt_proposal_transformers = gt_proposal_transformers.cuda()
        gt_proposal_classes = gt_proposal_classes.cuda()

        return proposal_bboxes, gt_proposal_classes, gt_proposal_transformers

    def loss(self, proposal_classes: Tensor, proposal_transformers: Tensor, gt_proposal_classes: Tensor, gt_proposal_transformers: Tensor):
        cross_entropy = F.cross_entropy(input=proposal_classes, target=gt_proposal_classes)

        proposal_transformers = proposal_transformers.view(-1, Model.NUM_CLASSES, 4)
        proposal_transformers = proposal_transformers[torch.arange(end=len(proposal_transformers), dtype=torch.long).cuda(), gt_proposal_classes]

        fg_indices = gt_proposal_classes.nonzero().view(-1)

        # NOTE: The default of `reduction` is `elementwise_mean`, which is divided by N x 4 (number of all elements), here we replaced by N for better performance
        smooth_l1_loss = F.smooth_l1_loss(input=proposal_transformers[fg_indices], target=gt_proposal_transformers[fg_indices], reduction='sum')
        smooth_l1_loss /= len(gt_proposal_transformers)

        return cross_entropy, smooth_l1_loss

    def save(self, path_to_checkpoints_dir: str, step: int) -> str:
        path_to_checkpoint = os.path.join(path_to_checkpoints_dir,
                                          'model-{:s}-{:d}.pth'.format(time.strftime('%Y%m%d%H%M'), step))
        torch.save(self.state_dict(), path_to_checkpoint)
        return path_to_checkpoint

    def load(self, path_to_checkpoint: str) -> 'Model':
        self.load_state_dict(torch.load(path_to_checkpoint))
        return self

    def detect(self, image: Tensor) -> Tuple[List[List[float]], List[int], List[float]]:
        forward_input = Model.ForwardInput.Eval(image)
        forward_output: Model.ForwardOutput.Eval = self.eval().forward(forward_input)

        proposal_bboxes = forward_output.proposal_bboxes
        proposal_classes = forward_output.proposal_classes
        proposal_transformers = forward_output.proposal_transformers

        proposal_transformers = proposal_transformers.view(-1, Model.NUM_CLASSES, 4)
        mean = self._transformer_normalize_mean.repeat(1, Model.NUM_CLASSES, 1).cuda()
        std = self._transformer_normalize_std.repeat(1, Model.NUM_CLASSES, 1).cuda()

        proposal_transformers = proposal_transformers * std - mean
        proposal_bboxes = proposal_bboxes.view(-1, 1, 4).repeat(1, Model.NUM_CLASSES, 1)
        detection_bboxes = BBox.apply_transformer(proposal_bboxes.view(-1, 4), proposal_transformers.view(-1, 4))

        detection_bboxes = detection_bboxes.view(-1, Model.NUM_CLASSES, 4)

        image_height, image_width = image.shape[1], image.shape[2]

        detection_bboxes[:, :, [0, 2]] = detection_bboxes[:, :, [0, 2]].clamp(min=0, max=image_width)
        detection_bboxes[:, :, [1, 3]] = detection_bboxes[:, :, [1, 3]].clamp(min=0, max=image_height)

        proposal_probs = F.softmax(proposal_classes, dim=1)

        detection_bboxes = detection_bboxes.cpu()
        proposal_probs = proposal_probs.cpu()

        bboxes = []
        labels = []
        probs = []

        for c in range(1, Model.NUM_CLASSES):
            detection_class_bboxes = detection_bboxes[:, c, :]
            proposal_class_probs = proposal_probs[:, c]

            _, sorted_indices = proposal_class_probs.sort(descending=True)
            detection_class_bboxes = detection_class_bboxes[sorted_indices]
            proposal_class_probs = proposal_class_probs[sorted_indices]

            keep_indices = NMS.suppress(detection_class_bboxes.cuda(), threshold=0.3)

            detection_class_bboxes = detection_class_bboxes[keep_indices]
            proposal_class_probs = proposal_class_probs[keep_indices]

            bboxes.extend(detection_class_bboxes.tolist())
            labels.extend([c] * len(keep_indices))
            probs.extend(proposal_class_probs.tolist())

        return bboxes, labels, probs

    class Detection(nn.Module):

        def __init__(self):
            super().__init__()

            self.fcs = nn.Sequential(
                nn.Linear(512 * 7 * 7, 4096),
                nn.ReLU(),
                nn.Linear(4096, 4096),
                nn.ReLU()
            )
            self._class = nn.Linear(4096, Model.NUM_CLASSES)
            self._transformer = nn.Linear(4096, Model.NUM_CLASSES * 4)

        def forward(self, features: Tensor, proposal_bboxes: Tensor) -> Tuple[Tensor, Tensor]:
            _, _, feature_map_height, feature_map_width = features.shape

            pool = []
            for proposal_bbox in proposal_bboxes:
                start_x = max(min(round(proposal_bbox[0].item() / 16), feature_map_width - 1), 0)      # [0, feature_map_width)
                start_y = max(min(round(proposal_bbox[1].item() / 16), feature_map_height - 1), 0)     # (0, feature_map_height]
                end_x = max(min(round(proposal_bbox[2].item() / 16) + 1, feature_map_width), 1)        # [0, feature_map_width)
                end_y = max(min(round(proposal_bbox[3].item() / 16) + 1, feature_map_height), 1)       # (0, feature_map_height]
                roi_feature_map = features[..., start_y:end_y, start_x:end_x]
                pool.append(F.adaptive_max_pool2d(roi_feature_map, 7))
            pool = torch.cat(pool, dim=0)   # pool has shape (128, 512, 7, 7)

            pool = pool.view(pool.shape[0], -1)
            h = self.fcs(pool)
            classes = self._class(h)
            transformers = self._transformer(h)
            return classes, transformers
