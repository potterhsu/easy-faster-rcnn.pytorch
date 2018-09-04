import argparse

import numpy as np

from dataset import Dataset
from evaluator import Evaluator
from model import Model


def _eval(checkpoint, path_to_data_dir, path_to_results_dir):
    dataset = Dataset(path_to_data_dir, Dataset.Mode.TEST)
    evaluator = Evaluator(dataset, path_to_data_dir, path_to_results_dir)

    model = Model().cuda()
    model.load(checkpoint)

    label_to_ap_dict = evaluator.evaluate(model)
    mean_ap = np.mean([v for k, v in label_to_ap_dict.items()])

    for c in range(1, Model.NUM_CLASSES):
        print('{:d}: {:s} AP = {:.4f}'.format(c, Dataset.LABEL_TO_CATEGORY_DICT[c], label_to_ap_dict[c]))

    print('mAP = {:.4f}'.format(mean_ap))


if __name__ == '__main__':
    def main():
        parser = argparse.ArgumentParser()
        parser.add_argument('checkpoint', type=str, help='path to evaluate checkpoint, e.g.: ./checkpoints/model-100.pth')
        parser.add_argument('-d', '--data_dir', default='./data', help='path to data directory')
        parser.add_argument('-r', '--results_dir', default='./results', help='path to results directory')
        args = parser.parse_args()

        checkpoint = args.checkpoint
        path_to_data_dir = args.data_dir
        path_to_results_dir = args.results_dir

        _eval(checkpoint, path_to_data_dir, path_to_results_dir)

    main()
