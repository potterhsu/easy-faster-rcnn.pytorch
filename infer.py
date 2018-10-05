import argparse
import random

from PIL import ImageDraw
from torchvision.transforms import transforms

from backbone.interface import Interface
from bbox import BBox
from dataset import Dataset
from model import Model


def _infer(path_to_input_image: str, path_to_output_image: str, path_to_checkpoint: str, backbone_name: str):
    image = transforms.Image.open(path_to_input_image)
    image_tensor, scale = Dataset.preprocess(image)

    backbone = Interface.from_name(backbone_name)(pretrained=False)
    model = Model(backbone).cuda()
    model.load(path_to_checkpoint)

    forward_input = Model.ForwardInput.Eval(image_tensor.cuda())
    forward_output: Model.ForwardOutput.Eval = model.eval().forward(forward_input)

    detection_bboxes = forward_output.detection_bboxes / scale
    detection_labels = forward_output.detection_labels
    detection_probs = forward_output.detection_probs

    draw = ImageDraw.Draw(image)

    for bbox, label, prob in zip(detection_bboxes.tolist(), detection_labels.tolist(), detection_probs.tolist()):
        if prob < 0.6:
            continue

        color = random.choice(['red', 'green', 'blue', 'yellow', 'purple', 'white'])
        bbox = BBox(left=bbox[0], top=bbox[1], right=bbox[2], bottom=bbox[3])
        category = Dataset.LABEL_TO_CATEGORY_DICT[label]

        draw.rectangle(((bbox.left, bbox.top), (bbox.right, bbox.bottom)), outline=color)
        draw.text((bbox.left, bbox.top), text=f'{category:s} {prob:.3f}', fill=color)

    image.save(path_to_output_image)


if __name__ == '__main__':
    def main():
        parser = argparse.ArgumentParser()
        parser.add_argument('input', type=str, help='path to input image')
        parser.add_argument('output', type=str, help='path to output result image')
        parser.add_argument('-c', '--checkpoint', help='path to checkpoint')
        parser.add_argument('-b', '--backbone', choices=['vgg16', 'resnet101'], required=True, help='name of backbone model')
        args = parser.parse_args()

        path_to_input_image = args.input
        path_to_output_image = args.output
        path_to_checkpoint = args.checkpoint
        backbone_name = args.backbone

        _infer(path_to_input_image, path_to_output_image, path_to_checkpoint, backbone_name)

    main()
