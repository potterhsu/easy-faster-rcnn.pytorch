from typing import List, Tuple

from config.config import Config


class EvalConfig(Config):

    RPN_PRE_NMS_TOP_N: int = 6000
    RPN_POST_NMS_TOP_N: int = 300

    @classmethod
    def setup(cls, image_min_side: float = None, image_max_side: float = None,
              anchor_ratios: List[Tuple[int, int]] = None, anchor_sizes: List[int] = None, pooler_mode: str = None,
              rpn_pre_nms_top_n: int = None, rpn_post_nms_top_n: int = None):
        super().setup(image_min_side, image_max_side, anchor_ratios, anchor_sizes, pooler_mode)

        if rpn_pre_nms_top_n is not None:
            cls.RPN_PRE_NMS_TOP_N = rpn_pre_nms_top_n
        if rpn_post_nms_top_n is not None:
            cls.RPN_POST_NMS_TOP_N = rpn_post_nms_top_n
