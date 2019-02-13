import ast
from typing import List, Tuple

from config.config import Config


class TrainConfig(Config):

    RPN_PRE_NMS_TOP_N: int = 12000
    RPN_POST_NMS_TOP_N: int = 2000

    ANCHOR_SMOOTH_L1_LOSS_BETA: float = 1.0
    PROPOSAL_SMOOTH_L1_LOSS_BETA: float = 1.0

    BATCH_SIZE: int = 1
    LEARNING_RATE: float = 0.001
    MOMENTUM: float = 0.9
    WEIGHT_DECAY: float = 0.0005
    STEP_LR_SIZES: List[int] = [50000, 70000]
    STEP_LR_GAMMA: float = 0.1
    WARM_UP_FACTOR: float = 0.3333
    WARM_UP_NUM_ITERS: int = 500

    NUM_STEPS_TO_DISPLAY: int = 20
    NUM_STEPS_TO_SNAPSHOT: int = 10000
    NUM_STEPS_TO_FINISH: int = 90000

    @classmethod
    def setup(cls, image_min_side: float = None, image_max_side: float = None,
              anchor_ratios: List[Tuple[int, int]] = None, anchor_sizes: List[int] = None, pooler_mode: str = None,
              rpn_pre_nms_top_n: int = None, rpn_post_nms_top_n: int = None,
              anchor_smooth_l1_loss_beta: float = None, proposal_smooth_l1_loss_beta: float = None,
              batch_size: int = None, learning_rate: float = None, momentum: float = None, weight_decay: float = None,
              step_lr_sizes: List[int] = None, step_lr_gamma: float = None,
              warm_up_factor: float = None, warm_up_num_iters: int = None,
              num_steps_to_display: int = None, num_steps_to_snapshot: int = None, num_steps_to_finish: int = None):
        super().setup(image_min_side, image_max_side, anchor_ratios, anchor_sizes, pooler_mode)

        if rpn_pre_nms_top_n is not None:
            cls.RPN_PRE_NMS_TOP_N = rpn_pre_nms_top_n
        if rpn_post_nms_top_n is not None:
            cls.RPN_POST_NMS_TOP_N = rpn_post_nms_top_n

        if anchor_smooth_l1_loss_beta is not None:
            cls.ANCHOR_SMOOTH_L1_LOSS_BETA = anchor_smooth_l1_loss_beta
        if proposal_smooth_l1_loss_beta is not None:
            cls.PROPOSAL_SMOOTH_L1_LOSS_BETA = proposal_smooth_l1_loss_beta

        if batch_size is not None:
            cls.BATCH_SIZE = batch_size
        if learning_rate is not None:
            cls.LEARNING_RATE = learning_rate
        if momentum is not None:
            cls.MOMENTUM = momentum
        if weight_decay is not None:
            cls.WEIGHT_DECAY = weight_decay
        if step_lr_sizes is not None:
            cls.STEP_LR_SIZES = ast.literal_eval(step_lr_sizes)
        if step_lr_gamma is not None:
            cls.STEP_LR_GAMMA = step_lr_gamma
        if warm_up_factor is not None:
            cls.WARM_UP_FACTOR = warm_up_factor
        if warm_up_num_iters is not None:
            cls.WARM_UP_NUM_ITERS = warm_up_num_iters

        if num_steps_to_display is not None:
            cls.NUM_STEPS_TO_DISPLAY = num_steps_to_display
        if num_steps_to_snapshot is not None:
            cls.NUM_STEPS_TO_SNAPSHOT = num_steps_to_snapshot
        if num_steps_to_finish is not None:
            cls.NUM_STEPS_TO_FINISH = num_steps_to_finish
