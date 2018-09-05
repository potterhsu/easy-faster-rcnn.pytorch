import argparse
import random

from PIL import ImageDraw
from torchvision.transforms import transforms

from bbox import BBox
from dataset import Dataset
from model import Model


def _infer(path_to_input_image, path_to_output_image, path_to_checkpoint):
    image = transforms.Image.open(path_to_input_image)
    image_tensor, scale = Dataset.preprosess(image)

    model = Model().cuda()
    model.load(path_to_checkpoint)
    bboxes, labels, probs = model.detect(image_tensor.cuda())

    draw = ImageDraw.Draw(image)

    for bbox, label, prob in zip(bboxes, labels, probs):
        if prob < 0.6:
            continue

        bbox = [it / scale for it in bbox]
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
        args = parser.parse_args()

        path_to_input_image = args.input
        path_to_output_image = args.output
        path_to_checkpoint = args.checkpoint

        _infer(path_to_input_image, path_to_output_image, path_to_checkpoint)

    main()
