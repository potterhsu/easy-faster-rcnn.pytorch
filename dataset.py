import os
import random
import xml.etree.ElementTree as ET
from enum import Enum
from typing import List, Tuple

import PIL
import torch.utils.data
from PIL import Image, ImageOps
from torch import Tensor
from torchvision import transforms

from bbox import BBox


class Dataset(torch.utils.data.Dataset):

    class Mode(Enum):
        TRAIN = 'train'
        TEST = 'test'

    class Annotation(object):
        class Object(object):
            def __init__(self, name: str, difficult: bool, bbox: BBox):
                super().__init__()
                self.name = name
                self.difficult = difficult
                self.bbox = bbox

            def __repr__(self) -> str:
                return 'Object[name={:s}, difficult={!s}, bbox={!s}]'.format(
                    self.name, self.difficult, self.bbox)

        def __init__(self, filename: str, objects: List[Object]):
            super().__init__()
            self.filename = filename
            self.objects = objects

    CATEGORY_TO_LABEL_DICT = {
        'background': 0,
        'aeroplane': 1, 'bicycle': 2, 'bird': 3, 'boat': 4, 'bottle': 5,
        'bus': 6, 'car': 7, 'cat': 8, 'chair': 9, 'cow': 10,
        'diningtable': 11, 'dog': 12, 'horse': 13, 'motorbike': 14, 'person': 15,
        'pottedplant': 16, 'sheep': 17, 'sofa': 18, 'train': 19, 'tvmonitor': 20
    }

    LABEL_TO_CATEGORY_DICT = {v: k for k, v in CATEGORY_TO_LABEL_DICT.items()}

    def __init__(self, path_to_data_dir: str, mode: Mode):
        super().__init__()

        self._mode = mode

        path_to_voc2007_dir = os.path.join(path_to_data_dir, 'VOCdevkit', 'VOC2007')
        path_to_imagesets_main_dir = os.path.join(path_to_voc2007_dir, 'ImageSets', 'Main')
        path_to_annotations_dir = os.path.join(path_to_voc2007_dir, 'Annotations')
        self._path_to_jpeg_images_dir = os.path.join(path_to_voc2007_dir, 'JPEGImages')

        if self._mode == Dataset.Mode.TRAIN:
            path_to_image_ids_txt = os.path.join(path_to_imagesets_main_dir, 'trainval.txt')
        elif self._mode == Dataset.Mode.TEST:
            path_to_image_ids_txt = os.path.join(path_to_imagesets_main_dir, 'test.txt')
        else:
            raise ValueError('invalid mode')

        with open(path_to_image_ids_txt, 'r') as f:
            lines = f.readlines()
            self._image_ids = [line.rstrip() for line in lines]

        self._image_id_to_annotation_dict = {}
        for image_id in self._image_ids:
            path_to_annotation_xml = os.path.join(path_to_annotations_dir, f'{image_id}.xml')
            tree = ET.ElementTree(file=path_to_annotation_xml)
            root = tree.getroot()

            self._image_id_to_annotation_dict[image_id] = Dataset.Annotation(
                filename=next(root.iterfind('filename')).text,
                objects=[Dataset.Annotation.Object(name=next(tag_object.iterfind('name')).text,
                                                   difficult=next(tag_object.iterfind('difficult')).text == '1',
                                                   bbox=BBox(
                                                       left=float(next(tag_object.iterfind('bndbox/xmin')).text),
                                                       top=float(next(tag_object.iterfind('bndbox/ymin')).text),
                                                       right=float(next(tag_object.iterfind('bndbox/xmax')).text),
                                                       bottom=float(next(tag_object.iterfind('bndbox/ymax')).text))
                                                   )
                         for tag_object in root.iterfind('object')]
            )

    def __len__(self) -> int:
        return len(self._image_id_to_annotation_dict)

    def __getitem__(self, index: int) -> Tuple[str, Tensor, float, Tensor, Tensor]:
        image_id = self._image_ids[index]
        annotation = self._image_id_to_annotation_dict[image_id]

        bboxes = [obj.bbox.tolist() for obj in annotation.objects if not obj.difficult]
        labels = [Dataset.CATEGORY_TO_LABEL_DICT[obj.name] for obj in annotation.objects if not obj.difficult]

        bboxes = torch.tensor(bboxes, dtype=torch.float)
        labels = torch.tensor(labels, dtype=torch.long)

        image = Image.open(os.path.join(self._path_to_jpeg_images_dir, annotation.filename))

        # random flip on only training mode
        if self._mode == Dataset.Mode.TRAIN and random.random() > 0.5:
            image = ImageOps.mirror(image)
            bboxes[:, [0, 2]] = image.width - bboxes[:, [2, 0]]  # index 0 and 2 represent `left` and `right` respectively

        image, scale = Dataset.preprocess(image)
        bboxes *= scale

        return image_id, image, scale, bboxes, labels

    @staticmethod
    def preprocess(image: PIL.Image.Image) -> Tuple[Tensor, float]:
        # resize according to the rules:
        #   1. scale shorter edge to 600
        #   2. after scaling, if longer edge > 1000, scale longer edge to 1000
        scale_for_shorter_edge = 600.0 / min(image.width, image.height)
        longer_edge_after_scaling = max(image.width, image.height) * scale_for_shorter_edge
        scale_for_longer_edge = (1000.0 / longer_edge_after_scaling) if longer_edge_after_scaling > 1000 else 1
        scale = scale_for_shorter_edge * scale_for_longer_edge

        transform = transforms.Compose([
            transforms.Resize((round(image.height * scale), round(image.width * scale))),  # interpolation `BILINEAR` is applied by default
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        image = transform(image)

        return image, scale


if __name__ == '__main__':
    def main():
        dataset = Dataset(path_to_data_dir='data', mode=Dataset.Mode.TRAIN)
        image_id, image, scale, bboxes, labels = dataset[0]
        print('image_id:', image_id)
        print('image.shape:', image.shape)
        print('scale:', scale)
        print('bboxes.shape:', bboxes.shape)
        print('labels.shape:', labels.shape)

    main()
