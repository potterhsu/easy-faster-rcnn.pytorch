import ast
from typing import Tuple, List

from roi.pooler import Pooler


class Config(object):

    IMAGE_MIN_SIDE: float = 600.0
    IMAGE_MAX_SIDE: float = 1000.0

    ANCHOR_RATIOS: List[Tuple[int, int]] = [(1, 2), (1, 1), (2, 1)]
    ANCHOR_SIZES: List[int] = [128, 256, 512]
    POOLER_MODE: Pooler.Mode = Pooler.Mode.ALIGN

    @classmethod
    def describe(cls):
        text = '\nConfig:\n'
        attrs = [attr for attr in dir(cls) if not callable(getattr(cls, attr)) and not attr.startswith('__')]
        text += '\n'.join(['\t{:s} = {:s}'.format(attr, str(getattr(cls, attr))) for attr in attrs]) + '\n'

        return text

    @classmethod
    def setup(cls, image_min_side: float = None, image_max_side: float = None,
              anchor_ratios: List[Tuple[int, int]] = None, anchor_sizes: List[int] = None, pooler_mode: str = None):
        if image_min_side is not None:
            cls.IMAGE_MIN_SIDE = image_min_side
        if image_max_side is not None:
            cls.IMAGE_MAX_SIDE = image_max_side

        if anchor_ratios is not None:
            cls.ANCHOR_RATIOS = ast.literal_eval(anchor_ratios)
        if anchor_sizes is not None:
            cls.ANCHOR_SIZES = ast.literal_eval(anchor_sizes)
        if pooler_mode is not None:
            cls.POOLER_MODE = Pooler.Mode(pooler_mode)
