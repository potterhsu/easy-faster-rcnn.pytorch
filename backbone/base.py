from typing import Tuple, Callable, Type

from torch import nn, Tensor


class Base(object):

    OPTIONS = ['vgg16', 'resnet101']

    @staticmethod
    def from_name(name: str) -> Type['Base']:
        if name == 'vgg16':
            from backbone.vgg16 import Vgg16
            return Vgg16
        elif name == 'resnet101':
            from backbone.resnet101 import ResNet101
            return ResNet101
        else:
            raise ValueError

    def __init__(self, pretrained: bool):
        super().__init__()
        self._pretrained = pretrained

    def features(self) -> Tuple[nn.Module, Callable[[Tensor], Tensor], nn.Module, Callable[[Tensor], Tensor], int, int]:
        raise NotImplementedError

    def pool_handler(self, pool: Tensor) -> Tensor:
        raise NotImplementedError

    def hidden_handler(self, hidden: Tensor) -> Tensor:
        raise NotImplementedError
