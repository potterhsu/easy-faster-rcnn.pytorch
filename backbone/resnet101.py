import torchvision
from torch import nn

from backbone.interface import Interface


class Resnet101(Interface):

    def __init__(self, pretrained: bool):
        super().__init__(pretrained)

    def features(self):
        resnet101 = torchvision.models.resnet101(pretrained=self._pretrained)

        # list(resnet101.children()) consists of following modules
        #   [0] = Conv2d, [1] = BatchNorm2d, [2] = ReLU, [3] = MaxPool2d,
        #   [4] = Sequential(Bottleneck...), [5] = Sequential(Bottleneck...),
        #   [6] = Sequential(Bottleneck...), [7] = Sequential(Bottleneck...),
        #   [8] = AvgPool2d, [9] = Linear
        features = list(resnet101.children())[:-2]

        for parameters in [feature.parameters() for i, feature in enumerate(features) if i <= 6]:
            for parameter in parameters:
                parameter.requires_grad = False

        features.append(nn.ConvTranspose2d(in_channels=2048, out_channels=512, kernel_size=3, stride=2, padding=1))
        features.append(nn.ReLU())

        features = nn.Sequential(*features)
        return features
