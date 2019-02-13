import random
from enum import Enum
from typing import Tuple, List, Type, Iterator

import PIL
import torch.utils.data.dataset
import torch.utils.data.sampler
from PIL import Image
from torch import Tensor
from torch.nn import functional as F
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

    def __getitem__(self, index: int) -> Tuple[str, Tensor, Tensor, Tensor, Tensor]:
        raise NotImplementedError

    def evaluate(self, path_to_results_dir: str, image_ids: List[str], bboxes: List[List[float]], classes: List[int], probs: List[float]) -> Tuple[float, str]:
        raise NotImplementedError

    def _write_results(self, path_to_results_dir: str, image_ids: List[str], bboxes: List[List[float]], classes: List[int], probs: List[float]):
        raise NotImplementedError

    @property
    def image_ratios(self) -> List[float]:
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

    @staticmethod
    def padding_collate_fn(batch: List[Tuple[str, Tensor, Tensor, Tensor, Tensor]]) -> Tuple[List[str], Tensor, Tensor, Tensor, Tensor]:
        image_id_batch, image_batch, scale_batch, bboxes_batch, labels_batch = zip(*batch)

        max_image_width = max([it.shape[2] for it in image_batch])
        max_image_height = max([it.shape[1] for it in image_batch])
        max_bboxes_length = max([len(it) for it in bboxes_batch])
        max_labels_length = max([len(it) for it in labels_batch])

        padded_image_batch = []
        padded_bboxes_batch = []
        padded_labels_batch = []

        for image in image_batch:
            padded_image = F.pad(input=image, pad=(0, max_image_width - image.shape[2], 0, max_image_height - image.shape[1]))  # pad has format (left, right, top, bottom)
            padded_image_batch.append(padded_image)

        for bboxes in bboxes_batch:
            padded_bboxes = torch.cat([bboxes, torch.zeros(max_bboxes_length - len(bboxes), 4).to(bboxes)])
            padded_bboxes_batch.append(padded_bboxes)

        for labels in labels_batch:
            padded_labels = torch.cat([labels, torch.zeros(max_labels_length - len(labels)).to(labels)])
            padded_labels_batch.append(padded_labels)

        image_id_batch = list(image_id_batch)
        padded_image_batch = torch.stack(padded_image_batch, dim=0)
        scale_batch = torch.stack(scale_batch, dim=0)
        padded_bboxes_batch = torch.stack(padded_bboxes_batch, dim=0)
        padded_labels_batch = torch.stack(padded_labels_batch, dim=0)

        return image_id_batch, padded_image_batch, scale_batch, padded_bboxes_batch, padded_labels_batch

    class NearestRatioRandomSampler(torch.utils.data.sampler.Sampler):

        def __init__(self, image_ratios: List[float], num_neighbors: int):
            super().__init__(data_source=None)
            self._image_ratios = image_ratios
            self._num_neighbors = num_neighbors

        def __len__(self) -> int:
            return len(self._image_ratios)

        def __iter__(self) -> Iterator[int]:
            image_ratios = torch.tensor(self._image_ratios)
            tall_indices = (image_ratios < 1).nonzero().view(-1)
            fat_indices = (image_ratios >= 1).nonzero().view(-1)

            tall_indices_length = len(tall_indices)
            fat_indices_length = len(fat_indices)

            tall_indices = tall_indices[torch.randperm(tall_indices_length)]
            fat_indices = fat_indices[torch.randperm(fat_indices_length)]

            num_tall_remainder = tall_indices_length % self._num_neighbors
            num_fat_remainder = fat_indices_length % self._num_neighbors

            tall_indices = tall_indices[:tall_indices_length - num_tall_remainder]
            fat_indices = fat_indices[:fat_indices_length - num_fat_remainder]

            tall_indices = tall_indices.view(-1, self._num_neighbors)
            fat_indices = fat_indices.view(-1, self._num_neighbors)
            merge_indices = torch.cat([tall_indices, fat_indices], dim=0)
            merge_indices = merge_indices[torch.randperm(len(merge_indices))].view(-1)

            return iter(merge_indices.tolist())
