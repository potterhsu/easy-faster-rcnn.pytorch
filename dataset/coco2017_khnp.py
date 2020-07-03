import json
import os
import pickle
import random
import sys
from io import StringIO
from typing import List, Tuple, Dict

import torch
import torch.utils.data.dataset
from PIL import Image, ImageOps
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torch import Tensor
from torchvision.datasets import CocoDetection
from tqdm import tqdm

from bbox import BBox
from dataset.base import Base
from dataset.coco2017 import COCO2017

import numpy as np
import cv2
from dataset.pallete_aug import heatMapConvert

class COCO2017KHNP(Base):

    class Annotation(object):
        class Object(object):
            def __init__(self, bbox: BBox, label: int):
                super().__init__()
                self.bbox = bbox
                self.label = label

            def __repr__(self):# -> str:
                return 'Object[label={:d}, bbox={!s}]'.format(
                    self.label, self.bbox)

        def __init__(self, filename: str, objects: List[Object]):
            super().__init__()
            self.filename = filename
            self.objects = objects

    # CATEGORY_TO_LABEL_DICT = {
    #     'background': 0,
    #     'Motor': 1, 
    #     'TB': 2, 
    #     'BTCG_TR': 3, 
    #     'BTCG_BUSBAR': 4, 
    #     'RT_TR': 5, 
    #     'ESWP_B': 6, 
    #     'Pt_1': 7, 
    #     'CB': 8, 
    #     'E': 9, 
    #     'FUSE': 10, 
    #     'TRDR': 11,
    #     'BUSHING':12,
    #     'LT_ARRESTER':13,
    #     'INSULATOR':14
    # }
    CATEGORY_TO_LABEL_DICT = {
        'BTCG_BUSBAR': 0, 
        'BTCG_TR': 1, 
        'BUSHING': 2, 
        'CB': 3, 
        'E': 4, 
        'ESWP_B': 5, 
        'FUSE': 6, 
        'INSULATOR': 7, 
        'LT_ARRESTER': 8, 
        'Motor': 9, 
        'Pt_1': 10, 
        'RT_TR': 11, 
        'TB': 12, 
        'TRDR': 13
    }

    LABEL_TO_CATEGORY_DICT = {v: k for k, v in CATEGORY_TO_LABEL_DICT.items()}

    def __init__(self, path_to_data_dir: str, mode: Base.Mode, image_min_side: float, image_max_side: float):
        super().__init__(path_to_data_dir, mode, image_min_side, image_max_side)

        path_to_coco_dir = os.path.join(self._path_to_data_dir, 'COCO2')
        path_to_annotations_dir = os.path.join(path_to_coco_dir, 'annotations')
        path_to_caches_dir = os.path.join('caches', 'coco2017-khnp', f'{self._mode.value}')
        path_to_image_ids_pickle = os.path.join(path_to_caches_dir, 'image-ids.pkl')
        path_to_image_id_dict_pickle = os.path.join(path_to_caches_dir, 'image-id-dict.pkl')
        path_to_image_ratios_pickle = os.path.join(path_to_caches_dir, 'image-ratios.pkl')

        if self._mode == COCO2017KHNP.Mode.TRAIN:
            path_to_jpeg_images_dir = os.path.join(path_to_coco_dir, 'train2017')
            path_to_annotation = os.path.join(path_to_annotations_dir, 'instances_train2017.json')
        elif self._mode == COCO2017KHNP.Mode.EVAL:
            path_to_jpeg_images_dir = os.path.join(path_to_coco_dir, 'val2017')
            path_to_annotation = os.path.join(path_to_annotations_dir, 'instances_val2017.json')
        else:
            raise ValueError('invalid mode')

        coco_dataset = CocoDetection(root=path_to_jpeg_images_dir, annFile=path_to_annotation)

        if os.path.exists(path_to_image_ids_pickle) and os.path.exists(path_to_image_id_dict_pickle):
            print('loading cache files...')

            with open(path_to_image_ids_pickle, 'rb') as f:
                self._image_ids = pickle.load(f)

            with open(path_to_image_id_dict_pickle, 'rb') as f:
                self._image_id_to_annotation_dict = pickle.load(f)

            with open(path_to_image_ratios_pickle, 'rb') as f:
                self._image_ratios = pickle.load(f)
        #else:
        if True:
            print('generating cache files...')

            os.makedirs(path_to_caches_dir, exist_ok=True)

            self._image_id_to_annotation_dict: Dict[str, COCO2017KHNP.Annotation] = {}
            self._image_ratios = []

            for idx, (image, annotation) in enumerate(tqdm(coco_dataset)):
                if len(annotation) > 0:
                    image_id = str(annotation[0]['image_id'])  # all image_id in annotation are the same
                    annotation = COCO2017KHNP.Annotation(
                        filename=os.path.join(path_to_jpeg_images_dir, '{:012d}.csv'.format(int(image_id))),
                        objects=[COCO2017KHNP.Annotation.Object(
                            bbox=BBox(  # `ann['bbox']` is in the format [left, top, width, height]
                                left=ann['bbox'][0],
                                top=ann['bbox'][1],
                                right=ann['bbox'][2]+ann['bbox'][0],
                                bottom=ann['bbox'][3] + ann['bbox'][1]
                            ),
                            label=ann['category_id'])
                            for ann in annotation]
                    )

                    if len(annotation.objects) > 0:
                        self._image_id_to_annotation_dict[image_id] = annotation

                        ratio = float(image.width / image.height)
                        self._image_ratios.append(ratio)

            self._image_ids = list(self._image_id_to_annotation_dict.keys())

            with open(path_to_image_ids_pickle, 'wb') as f:
                pickle.dump(self._image_ids, f)

            with open(path_to_image_id_dict_pickle, 'wb') as f:
                pickle.dump(self._image_id_to_annotation_dict, f)

            with open(path_to_image_ratios_pickle, 'wb') as f:
                pickle.dump(self.image_ratios, f)

    def __len__(self):# -> int:
        return len(self._image_id_to_annotation_dict)

    def __getitem__(self, index: int):# -> Tuple[str, Tensor, Tensor, Tensor, Tensor]:
        image_id = self._image_ids[index]
        annotation = self._image_id_to_annotation_dict[image_id]

        bboxes = [obj.bbox.tolist() for obj in annotation.objects]
        labels = [COCO2017KHNP.CATEGORY_TO_LABEL_DICT[COCO2017KHNP.LABEL_TO_CATEGORY_DICT[obj.label]] for obj in annotation.objects]  # mapping from original `COCO2017` dataset

        bboxes = torch.tensor(bboxes, dtype=torch.float)
        labels = torch.tensor(labels, dtype=torch.long)

        #image = Image.open(annotation.filename).convert('RGB')  # for some grayscale images
        image = np.loadtxt(annotation.filename,delimiter=',')
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        image = cv2.applyColorMap(image, cv2.COLORMAP_INFERNO)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        
        # random flip on only training mode
        if self._mode == COCO2017KHNP.Mode.TRAIN and random.random() > 0.5:
            image = ImageOps.mirror(image)
            bboxes[:, [0, 2]] = image.width - bboxes[:, [2, 0]]  # index 0 and 2 represent `left` and `right` respectively

        image, scale = COCO2017KHNP.preprocess(image, self._image_min_side, self._image_max_side)
        scale = torch.tensor(scale, dtype=torch.float)
        bboxes *= scale

        return image_id, image, scale, bboxes, labels

    def evaluate(self, path_to_results_dir: str, image_ids: List[str], bboxes: List[List[float]], classes: List[int], probs: List[float]):# -> Tuple[float, str]:
        self._write_results(path_to_results_dir, image_ids, bboxes, classes, probs)

        annType = 'bbox'
        path_to_coco_dir = os.path.join(self._path_to_data_dir, 'COCO2')
        path_to_annotations_dir = os.path.join(path_to_coco_dir, 'annotations')
        path_to_annotation = os.path.join(path_to_annotations_dir, 'instances_val2017.json')

        cocoGt = COCO(path_to_annotation)
        cocoDt = cocoGt.loadRes(os.path.join(path_to_results_dir, 'results.json'))

        cocoEval = COCOeval(cocoGt, cocoDt, annType)
        cocoEval.evaluate()
        cocoEval.accumulate()

        original_stdout = sys.stdout
        string_stdout = StringIO()
        sys.stdout = string_stdout
        cocoEval.summarize()
        sys.stdout = original_stdout

        mean_ap = cocoEval.stats[0].item()  # stats[0] records AP@[0.5:0.95]
        detail = string_stdout.getvalue()

        return mean_ap, detail

    def _write_results(self, path_to_results_dir: str, image_ids: List[str], bboxes: List[List[float]], classes: List[int], probs: List[float]):
        results = []
        for image_id, bbox, cls, prob in zip(image_ids, bboxes, classes, probs):
            results.append(
                {
                    'image_id': int(image_id),  # COCO evaluation requires `image_id` to be type `int`
                    'category_id': COCO2017KHNP.CATEGORY_TO_LABEL_DICT[COCO2017KHNP.LABEL_TO_CATEGORY_DICT[cls]],  # mapping to original `COCO2017` dataset
                    'bbox': [   # format [left, top, width, height] is expected
                        bbox[0],
                        bbox[1],
                        bbox[2] - bbox[0],
                        bbox[3] - bbox[1]
                    ],
                    'score': prob
                }
            )

        with open(os.path.join(path_to_results_dir, 'results.json'), 'w') as f:
            json.dump(results, f)

    @property
    def image_ratios(self):# -> List[float]:
        return self._image_ratios

    @staticmethod
    def num_classes():# -> int:
        return 14
