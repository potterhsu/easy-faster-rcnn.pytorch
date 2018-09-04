import os
import time
from typing import Union

import numpy as np
import torch
import torchvision.models
from torch import FloatTensor
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F

from bbox import BBox
from nms.nms import NMS
from rpn.region_proposal_network import RegionProposalNetwork


class Model(nn.Module):

    NUM_CLASSES = 21

    class ForwardInput:
        class Train(object):
            def __init__(self, image, gt_classes, gt_bboxes) -> None:
                self.image = image
                self.gt_classes = gt_classes
                self.gt_bboxes = gt_bboxes

        class Eval(object):
            def __init__(self, image) -> None:
                self.image = image

    class ForwardOutput:
        class Train(object):
            def __init__(self, anchor_objectness_loss, anchor_transformer_loss, proposal_class_loss, proposal_transformer_loss) -> None:
                self.anchor_objectness_loss = anchor_objectness_loss
                self.anchor_transformer_loss = anchor_transformer_loss
                self.proposal_class_loss = proposal_class_loss
                self.proposal_transformer_loss = proposal_transformer_loss

        class Eval(object):
            def __init__(self, proposal_bboxes, proposal_classes, proposal_transformers) -> None:
                self.proposal_bboxes = proposal_bboxes
                self.proposal_classes = proposal_classes
                self.proposal_transformers = proposal_transformers

    def __init__(self) -> None:
        super().__init__()

        vgg16 = torchvision.models.vgg16(pretrained=True)

        features = list(vgg16.features.children())
        for parameters in [feature.parameters() for i, feature in enumerate(features) if i < 10]:
            for parameter in parameters:
                parameter.requires_grad = False
        features.pop()

        self.features = nn.Sequential(*features)
        self.rpn = RegionProposalNetwork()
        self.head = Model.Head()

        self._transformer_normalize_mean = FloatTensor([0., 0., 0., 0.])
        self._transformer_normalize_std = FloatTensor([.1, .1, .2, .2])

    def forward(self, forward_input: Union[ForwardInput.Train, ForwardInput.Eval]) -> Union[ForwardOutput.Train, ForwardOutput.Eval]:
        image = Variable(forward_input.image, volatile=not self.training).unsqueeze(dim=0)
        image_height, image_width = image.shape[2], image.shape[3]

        features = self.features(image)
        anchor_bboxes, anchor_objectnesses, anchor_transformers, proposal_bboxes = self.rpn.forward(features, image_width, image_height)

        if self.training:
            forward_input: Model.ForwardInput.Train

            anchor_objectnesses, anchor_transformers, gt_anchor_objectnesses, gt_anchor_transformers = self.rpn.sample(anchor_bboxes, anchor_objectnesses, anchor_transformers, forward_input.gt_bboxes, image_width, image_height)
            anchor_objectness_loss, anchor_transformer_loss = self.rpn.loss(anchor_objectnesses, anchor_transformers, gt_anchor_objectnesses, gt_anchor_transformers)

            proposal_bboxes, gt_proposal_classes, gt_proposal_transformers = self.sample(proposal_bboxes, forward_input.gt_classes, forward_input.gt_bboxes)
            proposal_classes, proposal_transformers = self.head.forward(features, proposal_bboxes)
            proposal_class_loss, proposal_transformer_loss = self.loss(proposal_classes, proposal_transformers, gt_proposal_classes, gt_proposal_transformers)

            forward_output = Model.ForwardOutput.Train(anchor_objectness_loss, anchor_transformer_loss, proposal_class_loss, proposal_transformer_loss)
        else:
            proposal_classes, proposal_transformers = self.head.forward(features, proposal_bboxes)
            forward_output = Model.ForwardOutput.Eval(proposal_bboxes, proposal_classes, proposal_transformers)

        return forward_output

    def sample(self, proposal_bboxes, gt_classes, gt_bboxes):
        proposal_bboxes = proposal_bboxes.cpu()
        gt_classes = gt_classes.cpu()
        gt_bboxes = gt_bboxes.cpu()

        # find labels for each `proposal_bboxes`
        labels = torch.ones(len(proposal_bboxes)).long() * -1
        ious = BBox.iou(proposal_bboxes, gt_bboxes)
        proposal_max_ious, proposal_assignments = ious.max(dim=1)
        labels[proposal_max_ious < 0.5] = 0
        if len((proposal_max_ious >= 0.5).nonzero().squeeze()) > 0:
            labels[proposal_max_ious >= 0.5] = gt_classes[proposal_assignments[proposal_max_ious >= 0.5]]

        # select 128 samples
        fg_indices = (labels > 0).nonzero().squeeze()
        bg_indices = (labels == 0).nonzero().squeeze()
        if len(fg_indices) > 0:
            fg_indices = fg_indices[torch.randperm(len(fg_indices))[:min(len(fg_indices), 32)]]
        if len(bg_indices) > 0:
            bg_indices = bg_indices[torch.randperm(len(bg_indices))[:128 - len(fg_indices)]]
        select_indices = torch.cat([fg_indices, bg_indices])
        select_indices = select_indices[torch.randperm(len(select_indices))]

        proposal_bboxes = proposal_bboxes[select_indices]
        gt_proposal_transformers = BBox.calc_transformer(proposal_bboxes, gt_bboxes[proposal_assignments[select_indices]])
        gt_proposal_classes = labels[select_indices]

        gt_proposal_transformers = (gt_proposal_transformers - self._transformer_normalize_mean) / self._transformer_normalize_std
        gt_proposal_transformers = Variable(gt_proposal_transformers).cuda()
        gt_proposal_classes = Variable(gt_proposal_classes).cuda()

        return proposal_bboxes, gt_proposal_classes, gt_proposal_transformers

    def loss(self, proposal_classes, proposal_transformers, gt_proposal_classes, gt_proposal_transformers):
        cross_entropy = F.cross_entropy(input=proposal_classes, target=gt_proposal_classes)

        proposal_transformers = proposal_transformers.view(-1, Model.NUM_CLASSES, 4)
        proposal_transformers = proposal_transformers[torch.arange(0, len(proposal_transformers)).long().cuda(), gt_proposal_classes]

        fg_indices = np.where(gt_proposal_classes.data.cpu().numpy() > 0)[0]
        in_weight = np.zeros((len(proposal_transformers), 4))
        in_weight[fg_indices] = 1
        in_weight = Variable(FloatTensor(in_weight)).cuda()

        proposal_transformers = proposal_transformers * in_weight
        gt_proposal_transformers = gt_proposal_transformers * in_weight

        # NOTE: The default of `size_average` is `True`, which is divided by N x 4 (number of all elements), here we replaced by N for better performance
        smooth_l1_loss = F.smooth_l1_loss(input=proposal_transformers, target=gt_proposal_transformers, size_average=False)
        smooth_l1_loss /= len(gt_proposal_transformers)

        return cross_entropy, smooth_l1_loss

    def save(self, path_to_checkpoints_dir, step):
        path_to_checkpoint = os.path.join(path_to_checkpoints_dir,
                                          'model-{:s}-{:d}.pth'.format(time.strftime('%Y%m%d%H%M'), step))
        torch.save(self.state_dict(), path_to_checkpoint)
        return path_to_checkpoint

    def load(self, path_to_checkpoint):
        self.load_state_dict(torch.load(path_to_checkpoint))
        return self

    def detect(self, image):
        forward_input = Model.ForwardInput.Eval(image)
        forward_output: Model.ForwardOutput.Eval = self.eval().forward(forward_input)

        proposal_bboxes = forward_output.proposal_bboxes
        proposal_classes = forward_output.proposal_classes.data
        proposal_transformers = forward_output.proposal_transformers.data

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

        proposal_probs = F.softmax(Variable(proposal_classes), dim=1).data

        bboxes = []
        labels = []
        probs = []

        for c in range(1, Model.NUM_CLASSES):
            detection_class_bboxes = detection_bboxes[:, c, :]
            proposal_class_probs = proposal_probs[:, c]

            selected_indices = (proposal_class_probs > 0.05).nonzero().squeeze()
            if len(selected_indices) > 0:
                detection_class_bboxes = detection_class_bboxes[selected_indices]
                proposal_class_probs = proposal_class_probs[selected_indices]

            _, sorted_indices = proposal_class_probs.sort(descending=True)
            detection_class_bboxes = detection_class_bboxes[sorted_indices]
            proposal_class_probs = proposal_class_probs[sorted_indices]

            keep_indices = NMS.suppress(detection_class_bboxes, threshold=0.3)

            detection_class_bboxes = detection_class_bboxes[keep_indices]
            proposal_class_probs = proposal_class_probs[keep_indices]

            bboxes.extend(detection_class_bboxes.tolist())
            labels.extend([c] * len(keep_indices))
            probs.extend(proposal_class_probs.tolist())

        return bboxes, labels, probs

    class Head(nn.Module):

        def __init__(self):
            super().__init__()

            self.fcs = nn.Sequential(
                nn.Linear(512 * 7 * 7, 4096),
                nn.ReLU(),
                nn.Linear(4096, 4096),
                nn.ReLU(),
            )
            self._class = nn.Linear(4096, Model.NUM_CLASSES)
            self._transformer = nn.Linear(4096, Model.NUM_CLASSES * 4)

        def forward(self, features, proposal_bboxes):
            proposal_bboxes = Variable(proposal_bboxes)

            _, _, feature_map_height, feature_map_width = features.size()

            pool = []
            for proposal_bbox in proposal_bboxes:
                start_x = max(min(round(proposal_bbox[0].data[0] / 16), feature_map_width - 1), 0)      # [0, feature_map_width)
                start_y = max(min(round(proposal_bbox[1].data[0] / 16), feature_map_height - 1), 0)     # (0, feature_map_height]
                end_x = max(min(round(proposal_bbox[2].data[0] / 16) + 1, feature_map_width), 1)        # [0, feature_map_width)
                end_y = max(min(round(proposal_bbox[3].data[0] / 16) + 1, feature_map_height), 1)       # (0, feature_map_height]
                roi_feature_map = features[..., start_y:end_y, start_x:end_x]
                pool.append(F.adaptive_max_pool2d(roi_feature_map, 7))
            pool = torch.cat(pool, dim=0)   # pool has shape (128, 512, 7, 7)

            pool = pool.view(pool.shape[0], -1)
            h = self.fcs(pool)
            classes = self._class(h)
            transformers = self._transformer(h)
            return classes, transformers
