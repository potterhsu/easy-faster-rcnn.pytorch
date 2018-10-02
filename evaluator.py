import os
from typing import Dict, List

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import Dataset
from model import Model
from voc_eval import voc_eval


class Evaluator(object):
    def __init__(self, dataset: Dataset, path_to_data_dir: str, path_to_results_dir: str):
        super().__init__()
        self.dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=8, pin_memory=True)
        self._path_to_data_dir = path_to_data_dir
        self._path_to_results_dir = path_to_results_dir
        os.makedirs(self._path_to_results_dir, exist_ok=True)

    def evaluate(self, model: Model) -> Dict[int, float]:
        all_image_ids, all_pred_bboxes, all_pred_labels, all_pred_probs = [], [], [], []

        with torch.no_grad():
            for batch_index, (image_id_batch, image_batch, scale_batch, _, _) in enumerate(tqdm(self.dataloader)):
                image_id = image_id_batch[0]
                image = image_batch[0].cuda()
                scale = scale_batch[0].item()

                pred_bboxes, pred_labels, pred_probs = model.detect(image)
                pred_bboxes = [[it / scale for it in bbox] for bbox in pred_bboxes]

                all_pred_bboxes.extend(pred_bboxes)
                all_pred_labels.extend(pred_labels)
                all_pred_probs.extend(pred_probs)
                all_image_ids.extend([image_id] * len(pred_labels))

        self._write_results(all_image_ids, all_pred_bboxes, all_pred_labels, all_pred_probs)

        path_to_voc2007_dir = os.path.join(self._path_to_data_dir, 'VOCdevkit', 'VOC2007')
        path_to_main_dir = os.path.join(path_to_voc2007_dir, 'ImageSets', 'Main')
        path_to_annotations_dir = os.path.join(path_to_voc2007_dir, 'Annotations')

        label_to_ap_dict = {}
        for c in range(1, Model.NUM_CLASSES):
            category = Dataset.LABEL_TO_CATEGORY_DICT[c]
            try:
                _, _, ap = voc_eval(detpath=os.path.join(self._path_to_results_dir, 'comp3_det_test_{:s}.txt'.format(category)),
                                    annopath=os.path.join(path_to_annotations_dir, '{:s}.xml'),
                                    imagesetfile=os.path.join(path_to_main_dir, 'test.txt'),
                                    classname=category,
                                    cachedir='cache',
                                    ovthresh=0.5,
                                    use_07_metric=True)
            except IndexError:
                ap = 0

            label_to_ap_dict[c] = ap

        return label_to_ap_dict

    def _write_results(self, image_ids: List[str], bboxes: List[List[float]], labels: List[int], probs: List[float]):
        label_to_txt_files_dict = {}
        for c in range(1, Model.NUM_CLASSES):
            label_to_txt_files_dict[c] = open(os.path.join(self._path_to_results_dir, 'comp3_det_test_{:s}.txt'.format(Dataset.LABEL_TO_CATEGORY_DICT[c])), 'w')

        for image_id, bbox, label, prob in zip(image_ids, bboxes, labels, probs):
            label_to_txt_files_dict[label].write('{:s} {:f} {:f} {:f} {:f} {:f}\n'.format(image_id, prob,
                                                                                          bbox[0], bbox[1], bbox[2], bbox[3]))

        for _, f in label_to_txt_files_dict.items():
            f.close()
