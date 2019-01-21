from enum import Enum
from typing import Tuple, List, Type

import PIL
import torch.utils.data.dataset
from PIL import Image
from torch import Tensor
from torchvision.transforms import transforms


class Base(torch.utils.data.dataset.Dataset):

    class Mode(Enum):
        TRAIN = 'train'
        EVAL = 'eval'

    OPTIONS = ['voc2007', 'coco2017', 'voc2007-cat-dog', 'coco2017-person', 'coco2017-car', 'coco2017-animal']

    @staticmethod
    def from_name(name: str) -> Type['Base']:
        if name == 'voc2007':
            from dataset.voc2007 import VOC2007
            return VOC2007
        elif name == 'coco2017':
            from dataset.coco2017 import COCO2017
            return COCO2017
        elif name == 'voc2007-cat-dog':
            from dataset.voc2007_cat_dog import VOC2007CatDog
            return VOC2007CatDog
        elif name == 'coco2017-person':
            from dataset.coco2017_person import COCO2017Person
            return COCO2017Person
        elif name == 'coco2017-car':
            from dataset.coco2017_car import COCO2017Car
            return COCO2017Car
        elif name == 'coco2017-animal':
            from dataset.coco2017_animal import COCO2017Animal
            return COCO2017Animal
        else:
            raise ValueError

    def __init__(self, path_to_data_dir: str, mode: Mode, image_min_side: float, image_max_side: float):
        self._path_to_data_dir = path_to_data_dir
        self._mode = mode
        self._image_min_side = image_min_side
        self._image_max_side = image_max_side

    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, index: int) -> Tuple[str, Tensor, float, Tensor, Tensor]:
        raise NotImplementedError

    def evaluate(self, path_to_results_dir: str, image_ids: List[str], bboxes: List[List[float]], classes: List[int], probs: List[float]) -> Tuple[float, str]:
        raise NotImplementedError

    def _write_results(self, path_to_results_dir: str, image_ids: List[str], bboxes: List[List[float]], classes: List[int], probs: List[float]):
        raise NotImplementedError

    @staticmethod
    def num_classes() -> int:
        raise NotImplementedError

    @staticmethod
    def preprocess(image: PIL.Image.Image, image_min_side: float, image_max_side: float) -> Tuple[Tensor, float]:
        # resize according to the rules:
        #   1. scale shorter side to IMAGE_MIN_SIDE
        #   2. after scaling, if longer side > IMAGE_MAX_SIDE, scale longer side to IMAGE_MAX_SIDE
        scale_for_shorter_side = image_min_side / min(image.width, image.height)
        longer_side_after_scaling = max(image.width, image.height) * scale_for_shorter_side
        scale_for_longer_side = (image_max_side / longer_side_after_scaling) if longer_side_after_scaling > image_max_side else 1
        scale = scale_for_shorter_side * scale_for_longer_side

        transform = transforms.Compose([
            transforms.Resize((round(image.height * scale), round(image.width * scale))),  # interpolation `BILINEAR` is applied by default
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        image = transform(image)

        return image, scale
