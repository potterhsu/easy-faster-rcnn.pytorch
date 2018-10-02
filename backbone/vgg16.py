import torchvision
from torch import nn

from backbone.interface import Interface


class Vgg16(Interface):

    def __init__(self, pretrained: bool):
        super().__init__(pretrained)

    def features(self):
        vgg16 = torchvision.models.vgg16(pretrained=self._pretrained)

        # list(vgg16.features.children()) consists of following modules
        #   [0] = Conv2d, [1] = ReLU,
        #   [2] = Conv2d, [3] = ReLU,
        #   [4] = MaxPool2d,
        #   [5] = Conv2d, [6] = ReLU,
        #   [7] = Conv2d, [8] = ReLU,
        #   [9] = MaxPool2d,
        #   [10] = Conv2d, [11] = ReLU,
        #   ...
        #   [28] = Conv2d, [29] = ReLU,
        #   [30] = MaxPool2d
        features = list(vgg16.features.children())[:-1]

        for parameters in [feature.parameters() for i, feature in enumerate(features) if i < 10]:
            for parameter in parameters:
                parameter.requires_grad = False

        features = nn.Sequential(*features)
        return features
