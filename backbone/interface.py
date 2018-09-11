from typing import Type


class Interface(object):

    @staticmethod
    def from_name(name: str) -> Type['Interface']:
        if name == 'vgg16':
            from backbone.vgg16 import Vgg16
            return Vgg16
        elif name == 'resnet101':
            from backbone.resnet101 import Resnet101
            return Resnet101
        else:
            raise ValueError

    def __init__(self, pretrained: bool):
        super().__init__()
        self._pretrained = pretrained

    def features(self):
        raise NotImplementedError
