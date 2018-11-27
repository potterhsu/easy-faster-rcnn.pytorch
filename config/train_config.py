from typing import List, Tuple

from config.config import Config


class TrainConfig(Config):

    RPN_PRE_NMS_TOP_N: int = 12000
    RPN_POST_NMS_TOP_N: int = 2000

    LEARNING_RATE: float = 0.001
    MOMENTUM: float = 0.9
    WEIGHT_DECAY: float = 0.0005
    STEP_LR_SIZE: int = 50000
    STEP_LR_GAMMA: float = 0.1

    NUM_STEPS_TO_DISPLAY: int = 20
    NUM_STEPS_TO_SNAPSHOT: int = 10000
    NUM_STEPS_TO_FINISH: int = 70000

    @classmethod
    def setup(cls, image_min_side: float = None, image_max_side: float = None,
              anchor_ratios: List[Tuple[int, int]] = None, anchor_sizes: List[int] = None, pooling_mode: str = None,
              rpn_pre_nms_top_n: int = None, rpn_post_nms_top_n: int = None,
              learning_rate: float = None, momentum: float = None, weight_decay: float = None,
              step_lr_size: int = None, step_lr_gamma: float = None,
              num_steps_to_display: int = None, num_steps_to_snapshot: int = None, num_steps_to_finish: int = None):
        super().setup(image_min_side, image_max_side, anchor_ratios, anchor_sizes, pooling_mode)

        if rpn_pre_nms_top_n is not None:
            cls.RPN_PRE_NMS_TOP_N = rpn_pre_nms_top_n
        if rpn_post_nms_top_n is not None:
            cls.RPN_POST_NMS_TOP_N = rpn_post_nms_top_n

        if learning_rate is not None:
            cls.LEARNING_RATE = learning_rate
        if momentum is not None:
            cls.MOMENTUM = momentum
        if weight_decay is not None:
            cls.WEIGHT_DECAY = weight_decay
        if step_lr_size is not None:
            cls.STEP_LR_SIZE = step_lr_size
        if step_lr_gamma is not None:
            cls.STEP_LR_GAMMA = step_lr_gamma

        if num_steps_to_display is not None:
            cls.NUM_STEPS_TO_DISPLAY = num_steps_to_display
        if num_steps_to_snapshot is not None:
            cls.NUM_STEPS_TO_SNAPSHOT = num_steps_to_snapshot
        if num_steps_to_finish is not None:
            cls.NUM_STEPS_TO_FINISH = num_steps_to_finish
