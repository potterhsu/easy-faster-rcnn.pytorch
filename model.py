import os
from typing import Union, Tuple, Callable, List

import torch
from torch import nn, Tensor
from torch.nn import functional as F
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

from backbone.base import Base as BackboneBase
from bbox import BBox
from nms.nms import NMS
from roi.wrapper import Wrapper as ROIWrapper
from rpn.region_proposal_network import RegionProposalNetwork


class Model(nn.Module):

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
            def __init__(self, detection_bboxes: Tensor, detection_labels: Tensor, detection_probs: Tensor):
                self.detection_bboxes = detection_bboxes
                self.detection_labels = detection_labels
                self.detection_probs = detection_probs

    def __init__(self, backbone: BackboneBase, num_classes: int, pooling_mode: ROIWrapper.Mode,
                 anchor_ratios: List[Tuple[int, int]], anchor_sizes: List[int], pre_nms_top_n: int, post_nms_top_n: int):
        super().__init__()

        self.features, pool_handler, hidden, hidden_handler, num_features_out, num_hidden_out = backbone.features()
        self._bn_modules = [it for it in self.features.modules() if isinstance(it, nn.BatchNorm2d)] + \
                           [it for it in hidden.modules() if isinstance(it, nn.BatchNorm2d)]

        self.num_classes = num_classes

        self.rpn = RegionProposalNetwork(num_features_out, anchor_ratios, anchor_sizes, pre_nms_top_n, post_nms_top_n)
        self.detection = Model.Detection(pooling_mode, pool_handler, hidden, hidden_handler, num_hidden_out, self.num_classes)

        self._transformer_normalize_mean = torch.tensor([0., 0., 0., 0.], dtype=torch.float).cuda()
        self._transformer_normalize_std = torch.tensor([.1, .1, .2, .2], dtype=torch.float).cuda()

    def forward(self, forward_input: Union[ForwardInput.Train, ForwardInput.Eval]) -> Union[ForwardOutput.Train, ForwardOutput.Eval]:
        # freeze batch normalization modules for each forwarding process just in case model was switched to `train` at any time
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
            detection_bboxes, detection_labels, detection_probs = self._generate_detections(proposal_bboxes, proposal_classes, proposal_transformers, image_width, image_height)
            forward_output = Model.ForwardOutput.Eval(detection_bboxes, detection_labels, detection_probs)

        return forward_output

    def sample(self, proposal_bboxes: Tensor, gt_classes: Tensor, gt_bboxes: Tensor):
        # find labels for each `proposal_bboxes`
        labels = torch.ones(len(proposal_bboxes), dtype=torch.long).cuda() * -1
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

        proposal_transformers = proposal_transformers.view(-1, self.num_classes, 4)
        proposal_transformers = proposal_transformers[torch.arange(end=len(proposal_transformers), dtype=torch.long).cuda(), gt_proposal_classes]

        fg_indices = gt_proposal_classes.nonzero().view(-1)

        # NOTE: The default of `reduction` is `elementwise_mean`, which is divided by N x 4 (number of all elements), here we replaced by N for better performance
        smooth_l1_loss = F.smooth_l1_loss(input=proposal_transformers[fg_indices], target=gt_proposal_transformers[fg_indices], reduction='sum')
        smooth_l1_loss /= len(gt_proposal_transformers)

        return cross_entropy, smooth_l1_loss

    def save(self, path_to_checkpoints_dir: str, step: int, optimizer: Optimizer, scheduler: _LRScheduler) -> str:
        path_to_checkpoint = os.path.join(path_to_checkpoints_dir, f'model-{step}.pth')
        checkpoint = {
            'state_dict': self.state_dict(),
            'step': step,
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict()
        }
        torch.save(checkpoint, path_to_checkpoint)
        return path_to_checkpoint

    def load(self, path_to_checkpoint: str, optimizer: Optimizer = None, scheduler: _LRScheduler = None) -> 'Model':
        checkpoint = torch.load(path_to_checkpoint)
        self.load_state_dict(checkpoint['state_dict'])
        step = checkpoint['step']
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if scheduler is not None:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        return step

    def _generate_detections(self, proposal_bboxes: Tensor, proposal_classes: Tensor, proposal_transformers: Tensor, image_width: int, image_height: int) -> Tuple[Tensor, Tensor, Tensor]:
        proposal_transformers = proposal_transformers.view(-1, self.num_classes, 4)
        mean = self._transformer_normalize_mean.repeat(1, self.num_classes, 1)
        std = self._transformer_normalize_std.repeat(1, self.num_classes, 1)

        proposal_transformers = proposal_transformers * std - mean
        proposal_bboxes = proposal_bboxes.view(-1, 1, 4).repeat(1, self.num_classes, 1)
        detection_bboxes = BBox.apply_transformer(proposal_bboxes.view(-1, 4), proposal_transformers.view(-1, 4))

        detection_bboxes = detection_bboxes.view(-1, self.num_classes, 4)

        detection_bboxes[:, :, [0, 2]] = detection_bboxes[:, :, [0, 2]].clamp(min=0, max=image_width)
        detection_bboxes[:, :, [1, 3]] = detection_bboxes[:, :, [1, 3]].clamp(min=0, max=image_height)

        proposal_probs = F.softmax(proposal_classes, dim=1)

        detection_bboxes = detection_bboxes.cpu()
        proposal_probs = proposal_probs.cpu()

        generated_bboxes = []
        generated_labels = []
        generated_probs = []

        for c in range(1, self.num_classes):
            detection_class_bboxes = detection_bboxes[:, c, :]
            proposal_class_probs = proposal_probs[:, c]

            _, sorted_indices = proposal_class_probs.sort(descending=True)
            detection_class_bboxes = detection_class_bboxes[sorted_indices]
            proposal_class_probs = proposal_class_probs[sorted_indices]

            keep_indices = NMS.suppress(detection_class_bboxes.cuda(), threshold=0.3)
            detection_class_bboxes = detection_class_bboxes[keep_indices]
            proposal_class_probs = proposal_class_probs[keep_indices]

            generated_bboxes.append(detection_class_bboxes)
            generated_labels.append(torch.ones(len(keep_indices)) * c)
            generated_probs.append(proposal_class_probs)

        generated_bboxes = torch.cat(generated_bboxes, dim=0)
        generated_labels = torch.cat(generated_labels, dim=0)
        generated_probs = torch.cat(generated_probs, dim=0)
        return generated_bboxes, generated_labels, generated_probs

    class Detection(nn.Module):

        def __init__(self, pooling_mode: ROIWrapper.Mode, pool_handler: Callable[[Tensor], Tensor], hidden: nn.Module, hidden_handler: Callable[[Tensor], Tensor], num_hidden_out: int, num_classes: int):
            super().__init__()
            self._pooling_mode = pooling_mode
            self.pool_handler = pool_handler
            self.hidden = hidden
            self.hidden_handler = hidden_handler
            self._class = nn.Linear(num_hidden_out, num_classes)
            self._transformer = nn.Linear(num_hidden_out, num_classes * 4)

        def forward(self, features: Tensor, proposal_bboxes: Tensor) -> Tuple[Tensor, Tensor]:
            pool = ROIWrapper.apply(features, proposal_bboxes, mode=self._pooling_mode)

            pool = self.pool_handler(pool)
            hidden = self.hidden(pool)
            hidden = self.hidden_handler(hidden)

            classes = self._class(hidden)
            transformers = self._transformer(hidden)
            return classes, transformers
