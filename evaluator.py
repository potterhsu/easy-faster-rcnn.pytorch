from typing import Tuple

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset.base import Base as DatasetBase
from model import Model


class Evaluator(object):
    def __init__(self, dataset: DatasetBase, path_to_data_dir: str, path_to_results_dir: str):
        super().__init__()
        self._dataset = dataset
        self._dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=8, pin_memory=True)
        self._path_to_data_dir = path_to_data_dir
        self._path_to_results_dir = path_to_results_dir

    def evaluate(self, model: Model) -> Tuple[float, str]:
        all_image_ids, all_detection_bboxes, all_detection_labels, all_detection_probs = [], [], [], []

        with torch.no_grad():
            for batch_index, (image_id_batch, image_batch, scale_batch, _, _) in enumerate(tqdm(self._dataloader)):
                image_id = image_id_batch[0]
                image = image_batch[0].cuda()
                scale = scale_batch[0].item()

                forward_input = Model.ForwardInput.Eval(image)
                forward_output: Model.ForwardOutput.Eval = model.eval().forward(forward_input)

                detection_bboxes = forward_output.detection_bboxes / scale
                detection_labels = forward_output.detection_labels
                detection_probs = forward_output.detection_probs

                selected_indices = (detection_probs > 0.05).nonzero().view(-1)
                detection_bboxes = detection_bboxes[selected_indices]
                detection_labels = detection_labels[selected_indices]
                detection_probs = detection_probs[selected_indices]

                all_detection_bboxes.extend(detection_bboxes.tolist())
                all_detection_labels.extend(detection_labels.tolist())
                all_detection_probs.extend(detection_probs.tolist())
                all_image_ids.extend([image_id] * len(detection_bboxes))

        mean_ap, detail = self._dataset.evaluate(self._path_to_results_dir, all_image_ids, all_detection_bboxes, all_detection_labels, all_detection_probs)
        return mean_ap, detail
