import argparse
import os
import random

from PIL import ImageDraw
from torchvision.transforms import transforms
from dataset.base import Base as DatasetBase
from backbone.base import Base as BackboneBase
from bbox import BBox
from model import Model
from roi.wrapper import Wrapper as ROIWrapper
from config.eval_config import EvalConfig as Config


def _infer(path_to_input_image: str, path_to_output_image: str, path_to_checkpoint: str, dataset_name: str, backbone_name: str, prob_thresh: float):
    image = transforms.Image.open(path_to_input_image)
    dataset_class = DatasetBase.from_name(dataset_name)
    image_tensor, scale = dataset_class.preprocess(image, Config.IMAGE_MIN_SIDE, Config.IMAGE_MAX_SIDE)

    backbone = BackboneBase.from_name(backbone_name)(pretrained=False)
    model = Model(backbone, dataset_class.num_classes(), pooling_mode=Config.POOLING_MODE,
                  anchor_ratios=Config.ANCHOR_RATIOS, anchor_sizes=Config.ANCHOR_SIZES,
                  pre_nms_top_n=Config.RPN_PRE_NMS_TOP_N, post_nms_top_n=Config.RPN_POST_NMS_TOP_N).cuda()
    model.load(path_to_checkpoint)

    forward_input = Model.ForwardInput.Eval(image_tensor.cuda())
    forward_output: Model.ForwardOutput.Eval = model.eval().forward(forward_input)

    detection_bboxes = forward_output.detection_bboxes / scale
    detection_labels = forward_output.detection_labels
    detection_probs = forward_output.detection_probs

    draw = ImageDraw.Draw(image)

    for bbox, label, prob in zip(detection_bboxes.tolist(), detection_labels.tolist(), detection_probs.tolist()):
        if prob < prob_thresh:
            continue

        color = random.choice(['red', 'green', 'blue', 'yellow', 'purple', 'white'])
        bbox = BBox(left=bbox[0], top=bbox[1], right=bbox[2], bottom=bbox[3])
        category = dataset_class.LABEL_TO_CATEGORY_DICT[label]

        draw.rectangle(((bbox.left, bbox.top), (bbox.right, bbox.bottom)), outline=color)
        draw.text((bbox.left, bbox.top), text=f'{category:s} {prob:.3f}', fill=color)

    image.save(path_to_output_image)
    print(f'Output image is saved to {path_to_output_image}')


if __name__ == '__main__':
    def main():
        parser = argparse.ArgumentParser()
        parser.add_argument('input', type=str, help='path to input image')
        parser.add_argument('output', type=str, help='path to output result image')
        parser.add_argument('-c', '--checkpoint', type=str, required=True, help='path to checkpoint')
        parser.add_argument('-s', '--dataset', type=str, choices=DatasetBase.OPTIONS, required=True, help='name of dataset')
        parser.add_argument('-b', '--backbone', type=str, choices=BackboneBase.OPTIONS, required=True, help='name of backbone model')
        parser.add_argument('-p', '--probability_threshold', type=float, default=0.6, help='threshold of detection probability')
        parser.add_argument('--image_min_side', type=float, help='default: {:g}'.format(Config.IMAGE_MIN_SIDE))
        parser.add_argument('--image_max_side', type=float, help='default: {:g}'.format(Config.IMAGE_MAX_SIDE))
        parser.add_argument('--anchor_ratios', type=str, help='default: "{!s}"'.format(Config.ANCHOR_RATIOS))
        parser.add_argument('--anchor_sizes', type=str, help='default: "{!s}"'.format(Config.ANCHOR_SIZES))
        parser.add_argument('--pooling_mode', type=str, choices=ROIWrapper.OPTIONS, help='default: {.value:s}'.format(Config.POOLING_MODE))
        parser.add_argument('--rpn_pre_nms_top_n', type=int, help='default: {:d}'.format(Config.RPN_PRE_NMS_TOP_N))
        parser.add_argument('--rpn_post_nms_top_n', type=int, help='default: {:d}'.format(Config.RPN_POST_NMS_TOP_N))
        args = parser.parse_args()

        path_to_input_image = args.input
        path_to_output_image = args.output
        path_to_checkpoint = args.checkpoint
        dataset_name = args.dataset
        backbone_name = args.backbone
        prob_thresh = args.probability_threshold

        os.makedirs(os.path.join(os.path.curdir, os.path.dirname(path_to_output_image)), exist_ok=True)

        Config.setup(image_min_side=args.image_min_side, image_max_side=args.image_max_side,
                     anchor_ratios=args.anchor_ratios, anchor_sizes=args.anchor_sizes, pooling_mode=args.pooling_mode,
                     rpn_pre_nms_top_n=args.rpn_pre_nms_top_n, rpn_post_nms_top_n=args.rpn_post_nms_top_n)

        _infer(path_to_input_image, path_to_output_image, path_to_checkpoint, dataset_name, backbone_name, prob_thresh)

    main()
