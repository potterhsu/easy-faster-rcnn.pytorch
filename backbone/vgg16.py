from typing import Tuple, Callable

import torchvision
from torch import nn, Tensor

import backbone.base


class Vgg16(backbone.base.Base):

    def __init__(self, pretrained: bool):
        super().__init__(pretrained)

    def features(self) -> Tuple[nn.Module, Callable[[Tensor], Tensor], nn.Module, Callable[[Tensor], Tensor], int, int]:
        vgg16 = torchvision.models.vgg16(pretrained=self._pretrained)

        # vgg16.features consists of following modules
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
        features = vgg16.features[:-1]
        num_features_out = 512

        # vgg16.classifier consists of following modules
        #   [0] = Linear, [1] = ReLU, [2] = Dropout,
        #   [3] = Linear, [4] = ReLU, [5] = Dropout,
        #   [6] = Linear
        hidden = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(),
            # nn.Dropout(),  # remove `Dropout` for better performance
            nn.Linear(4096, 4096),
            nn.ReLU(),
            # nn.Dropout()  # remove `Dropout` for better performance
        )
        num_hidden_out = 4096

        for parameters in [feature.parameters() for i, feature in enumerate(features) if i < 10]:
            for parameter in parameters:
                parameter.requires_grad = False

        return features, self.pool_handler, hidden, self.hidden_handler, num_features_out, num_hidden_out

    def pool_handler(self, pool: Tensor) -> Tensor:
        return pool.view(pool.shape[0], -1)

    def hidden_handler(self, hidden: Tensor) -> Tensor:
        return hidden


