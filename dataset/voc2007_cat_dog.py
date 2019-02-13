import os
import random
import xml.etree.ElementTree as ET
from typing import List, Tuple

import numpy as np
import torch.utils.data
from PIL import Image, ImageOps
from torch import Tensor

from bbox import BBox
from dataset.base import Base
from voc_eval import voc_eval


class VOC2007CatDog(Base):

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
        'cat': 1, 'dog': 2
    }

    LABEL_TO_CATEGORY_DICT = {v: k for k, v in CATEGORY_TO_LABEL_DICT.items()}

    def __init__(self, path_to_data_dir: str, mode: Base.Mode, image_min_side: float, image_max_side: float):
        super().__init__(path_to_data_dir, mode, image_min_side, image_max_side)

        path_to_voc2007_dir = os.path.join(self._path_to_data_dir, 'VOCdevkit', 'VOC2007')
        path_to_imagesets_main_dir = os.path.join(path_to_voc2007_dir, 'ImageSets', 'Main')
        path_to_annotations_dir = os.path.join(path_to_voc2007_dir, 'Annotations')
        self._path_to_jpeg_images_dir = os.path.join(path_to_voc2007_dir, 'JPEGImages')

        if self._mode == VOC2007CatDog.Mode.TRAIN:
            path_to_image_ids_txt = os.path.join(path_to_imagesets_main_dir, 'trainval.txt')
        elif self._mode == VOC2007CatDog.Mode.EVAL:
            path_to_image_ids_txt = os.path.join(path_to_imagesets_main_dir, 'test.txt')
        else:
            raise ValueError('invalid mode')

        with open(path_to_image_ids_txt, 'r') as f:
            lines = f.readlines()
            image_ids = [line.rstrip() for line in lines]

        self._image_id_to_annotation_dict = {}
        self._image_ratios = []

        for image_id in image_ids:
            path_to_annotation_xml = os.path.join(path_to_annotations_dir, f'{image_id}.xml')
            tree = ET.ElementTree(file=path_to_annotation_xml)
            root = tree.getroot()

            annotation = VOC2007CatDog.Annotation(
                filename=root.find('filename').text,
                objects=[VOC2007CatDog.Annotation.Object(
                    name=next(tag_object.iterfind('name')).text,
                    difficult=next(tag_object.iterfind('difficult')).text == '1',
                    bbox=BBox(  # convert to 0-based pixel index
                        left=float(next(tag_object.iterfind('bndbox/xmin')).text) - 1,
                        top=float(next(tag_object.iterfind('bndbox/ymin')).text) - 1,
                        right=float(next(tag_object.iterfind('bndbox/xmax')).text) - 1,
                        bottom=float(next(tag_object.iterfind('bndbox/ymax')).text) - 1
                    )
                ) for tag_object in root.iterfind('object')]
            )
            annotation.objects = [obj for obj in annotation.objects if obj.name in ['cat', 'dog'] and not obj.difficult]

            if len(annotation.objects) > 0:
                self._image_id_to_annotation_dict[image_id] = annotation

                width = int(root.find('size/width').text)
                height = int(root.find('size/height').text)
                ratio = float(width / height)
                self._image_ratios.append(ratio)

        self._image_ids = list(self._image_id_to_annotation_dict.keys())

    def __len__(self) -> int:
        return len(self._image_id_to_annotation_dict)

    def __getitem__(self, index: int) -> Tuple[str, Tensor, Tensor, Tensor, Tensor]:
        image_id = self._image_ids[index]
        annotation = self._image_id_to_annotation_dict[image_id]

        bboxes = [obj.bbox.tolist() for obj in annotation.objects]
        labels = [VOC2007CatDog.CATEGORY_TO_LABEL_DICT[obj.name] for obj in annotation.objects]

        bboxes = torch.tensor(bboxes, dtype=torch.float)
        labels = torch.tensor(labels, dtype=torch.long)

        image = Image.open(os.path.join(self._path_to_jpeg_images_dir, annotation.filename))

        # random flip on only training mode
        if self._mode == VOC2007CatDog.Mode.TRAIN and random.random() > 0.5:
            image = ImageOps.mirror(image)
            bboxes[:, [0, 2]] = image.width - bboxes[:, [2, 0]]  # index 0 and 2 represent `left` and `right` respectively

        image, scale = VOC2007CatDog.preprocess(image, self._image_min_side, self._image_max_side)
        scale = torch.tensor(scale, dtype=torch.float)
        bboxes *= scale

        return image_id, image, scale, bboxes, labels

    def evaluate(self, path_to_results_dir: str, image_ids: List[str], bboxes: List[List[float]], classes: List[int], probs: List[float]) -> Tuple[float, str]:
        self._write_results(path_to_results_dir, image_ids, bboxes, classes, probs)

        path_to_voc2007_dir = os.path.join(self._path_to_data_dir, 'VOCdevkit', 'VOC2007')
        path_to_main_dir = os.path.join(path_to_voc2007_dir, 'ImageSets', 'Main')
        path_to_annotations_dir = os.path.join(path_to_voc2007_dir, 'Annotations')

        class_to_ap_dict = {}
        for c in range(1, VOC2007CatDog.num_classes()):
            category = VOC2007CatDog.LABEL_TO_CATEGORY_DICT[c]
            try:
                path_to_cache_dir = os.path.join('caches', 'voc2007-cat-dog')
                os.makedirs(path_to_cache_dir, exist_ok=True)
                _, _, ap = voc_eval(detpath=os.path.join(path_to_results_dir, 'comp3_det_test_{:s}.txt'.format(category)),
                                    annopath=os.path.join(path_to_annotations_dir, '{:s}.xml'),
                                    imagesetfile=os.path.join(path_to_main_dir, 'test.txt'),
                                    classname=category,
                                    cachedir=path_to_cache_dir,
                                    ovthresh=0.5,
                                    use_07_metric=True)
            except IndexError:
                ap = 0

            class_to_ap_dict[c] = ap

        mean_ap = np.mean([v for k, v in class_to_ap_dict.items()]).item()

        detail = ''
        for c in range(1, VOC2007CatDog.num_classes()):
            detail += '{:d}: {:s} AP = {:.4f}\n'.format(c, VOC2007CatDog.LABEL_TO_CATEGORY_DICT[c], class_to_ap_dict[c])

        return mean_ap, detail

    def _write_results(self, path_to_results_dir: str, image_ids: List[str], bboxes: List[List[float]], classes: List[int], probs: List[float]):
        class_to_txt_files_dict = {}
        for c in range(1, VOC2007CatDog.num_classes()):
            class_to_txt_files_dict[c] = open(os.path.join(path_to_results_dir, 'comp3_det_test_{:s}.txt'.format(VOC2007CatDog.LABEL_TO_CATEGORY_DICT[c])), 'w')

        for image_id, bbox, cls, prob in zip(image_ids, bboxes, classes, probs):
            class_to_txt_files_dict[cls].write('{:s} {:f} {:f} {:f} {:f} {:f}\n'.format(image_id, prob,
                                                                                        bbox[0], bbox[1], bbox[2], bbox[3]))

        for _, f in class_to_txt_files_dict.items():
            f.close()

    @property
    def image_ratios(self) -> List[float]:
        return self._image_ratios

    @staticmethod
    def num_classes() -> int:
        return 3
