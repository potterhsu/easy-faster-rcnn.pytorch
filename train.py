import argparse
import os
import time
from collections import deque
from typing import Optional

import uuid
from torch import optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

from backbone.base import Base as BackboneBase
from config.train_config import TrainConfig as Config
from dataset.base import Base as DatasetBase
from logger import Logger as Log
from model import Model
from roi.wrapper import Wrapper as ROIWrapper


def _train(dataset_name: str, backbone_name: str, path_to_data_dir: str, path_to_checkpoints_dir: str, path_to_resuming_checkpoint: Optional[str]):
    dataset = DatasetBase.from_name(dataset_name)(path_to_data_dir, DatasetBase.Mode.TRAIN, Config.IMAGE_MIN_SIDE, Config.IMAGE_MAX_SIDE)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=8, pin_memory=True)

    Log.i('Found {:d} samples'.format(len(dataset)))

    backbone = BackboneBase.from_name(backbone_name)(pretrained=True)
    model = Model(backbone, dataset.num_classes(), pooling_mode=Config.POOLING_MODE,
                  anchor_ratios=Config.ANCHOR_RATIOS, anchor_sizes=Config.ANCHOR_SIZES,
                  rpn_pre_nms_top_n=Config.RPN_PRE_NMS_TOP_N, rpn_post_nms_top_n=Config.RPN_POST_NMS_TOP_N).cuda()
    optimizer = optim.SGD(model.parameters(), lr=Config.LEARNING_RATE,
                          momentum=Config.MOMENTUM, weight_decay=Config.WEIGHT_DECAY)
    scheduler = StepLR(optimizer, step_size=Config.STEP_LR_SIZE, gamma=Config.STEP_LR_GAMMA)

    step = 0
    time_checkpoint = time.time()
    losses = deque(maxlen=100)
    should_stop = False

    num_steps_to_display = Config.NUM_STEPS_TO_DISPLAY
    num_steps_to_snapshot = Config.NUM_STEPS_TO_SNAPSHOT
    num_steps_to_finish = Config.NUM_STEPS_TO_FINISH

    if path_to_resuming_checkpoint is not None:
        step = model.load(path_to_resuming_checkpoint, optimizer, scheduler)
        Log.i(f'Model has been restored from file: {path_to_resuming_checkpoint}')

    Log.i('Start training')

    while not should_stop:
        for batch_index, (_, image_batch, _, bboxes_batch, labels_batch) in enumerate(dataloader):
            assert image_batch.shape[0] == 1, 'only batch size of 1 is supported'

            image = image_batch[0].cuda()
            bboxes = bboxes_batch[0].cuda()
            labels = labels_batch[0].cuda()

            forward_input = Model.ForwardInput.Train(image, gt_classes=labels, gt_bboxes=bboxes)
            forward_output: Model.ForwardOutput.Train = model.train().forward(forward_input)

            loss = forward_output.anchor_objectness_loss + forward_output.anchor_transformer_loss + \
                forward_output.proposal_class_loss + forward_output.proposal_transformer_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            losses.append(loss.item())
            step += 1

            if step == num_steps_to_finish:
                should_stop = True

            if step % num_steps_to_display == 0:
                elapsed_time = time.time() - time_checkpoint
                time_checkpoint = time.time()
                steps_per_sec = num_steps_to_display / elapsed_time
                avg_loss = sum(losses) / len(losses)
                lr = scheduler.get_lr()[0]
                Log.i(f'[Step {step}] Avg. Loss = {avg_loss:.6f}, Learning Rate = {lr} ({steps_per_sec:.2f} steps/sec)')

            if step % num_steps_to_snapshot == 0 or should_stop:
                path_to_checkpoint = model.save(path_to_checkpoints_dir, step, optimizer, scheduler)
                Log.i(f'Model has been saved to {path_to_checkpoint}')

            if should_stop:
                break

    Log.i('Done')


if __name__ == '__main__':
    def main():
        parser = argparse.ArgumentParser()
        parser.add_argument('-s', '--dataset', type=str, choices=DatasetBase.OPTIONS, required=True, help='name of dataset')
        parser.add_argument('-b', '--backbone', type=str, choices=BackboneBase.OPTIONS, required=True, help='name of backbone model')
        parser.add_argument('-d', '--data_dir', type=str, default='./data', help='path to data directory')
        parser.add_argument('-o', '--outputs_dir', type=str, default='./outputs', help='path to outputs directory')
        parser.add_argument('-r', '--resume_checkpoint', type=str, help='path to resuming checkpoint')
        parser.add_argument('--image_min_side', type=float, help='default: {:g}'.format(Config.IMAGE_MIN_SIDE))
        parser.add_argument('--image_max_side', type=float, help='default: {:g}'.format(Config.IMAGE_MAX_SIDE))
        parser.add_argument('--anchor_ratios', type=str, help='default: "{!s}"'.format(Config.ANCHOR_RATIOS))
        parser.add_argument('--anchor_sizes', type=str, help='default: "{!s}"'.format(Config.ANCHOR_SIZES))
        parser.add_argument('--pooling_mode', type=str, choices=ROIWrapper.OPTIONS, help='default: {.value:s}'.format(Config.POOLING_MODE))
        parser.add_argument('--rpn_pre_nms_top_n', type=int, help='default: {:d}'.format(Config.RPN_PRE_NMS_TOP_N))
        parser.add_argument('--rpn_post_nms_top_n', type=int, help='default: {:d}'.format(Config.RPN_POST_NMS_TOP_N))
        parser.add_argument('--learning_rate', type=float, help='default: {:g}'.format(Config.LEARNING_RATE))
        parser.add_argument('--momentum', type=float, help='default: {:g}'.format(Config.MOMENTUM))
        parser.add_argument('--weight_decay', type=float, help='default: {:g}'.format(Config.WEIGHT_DECAY))
        parser.add_argument('--step_lr_size', type=float, help='default: {:d}'.format(Config.STEP_LR_SIZE))
        parser.add_argument('--step_lr_gamma', type=float, help='default: {:g}'.format(Config.STEP_LR_GAMMA))
        parser.add_argument('--num_steps_to_display', type=int, help='default: {:d}'.format(Config.NUM_STEPS_TO_DISPLAY))
        parser.add_argument('--num_steps_to_snapshot', type=int, help='default: {:d}'.format(Config.NUM_STEPS_TO_SNAPSHOT))
        parser.add_argument('--num_steps_to_finish', type=int, help='default: {:d}'.format(Config.NUM_STEPS_TO_FINISH))
        args = parser.parse_args()

        dataset_name = args.dataset
        backbone_name = args.backbone
        path_to_data_dir = args.data_dir
        path_to_outputs_dir = args.outputs_dir
        path_to_resuming_checkpoint = args.resume_checkpoint

        path_to_checkpoints_dir = os.path.join(path_to_outputs_dir, 'checkpoints-{:s}-{:s}-{:s}-{:s}'.format(
            time.strftime('%Y%m%d%H%M%S'), dataset_name, backbone_name, str(uuid.uuid4()).split('-')[0]))
        os.makedirs(path_to_checkpoints_dir)

        Config.setup(image_min_side=args.image_min_side, image_max_side=args.image_max_side,
                     anchor_ratios=args.anchor_ratios, anchor_sizes=args.anchor_sizes, pooling_mode=args.pooling_mode,
                     rpn_pre_nms_top_n=args.rpn_pre_nms_top_n, rpn_post_nms_top_n=args.rpn_post_nms_top_n,
                     learning_rate=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay,
                     step_lr_size=args.step_lr_size, step_lr_gamma=args.step_lr_gamma,
                     num_steps_to_display=args.num_steps_to_display, num_steps_to_snapshot=args.num_steps_to_snapshot, num_steps_to_finish=args.num_steps_to_finish)

        Log.initialize(os.path.join(path_to_checkpoints_dir, 'train.log'))
        Log.i('Arguments:')
        for k, v in vars(args).items():
            Log.i(f'\t{k} = {v}')
        Log.i(Config.describe())

        _train(dataset_name, backbone_name, path_to_data_dir, path_to_checkpoints_dir, path_to_resuming_checkpoint)

    main()
