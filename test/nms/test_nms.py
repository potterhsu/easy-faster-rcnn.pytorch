import os
import time
import unittest

import numpy as np
import torch

from support.layer.nms import nms


class TestNMS(unittest.TestCase):
    def _run_nms(self, bboxes, scores):
        start = time.time()
        threshold = 0.7
        kept_indices = nms(bboxes, scores, threshold)
        print('%s in %.3fs, %d -> %d' % (self.id(), time.time() - start, len(bboxes), len(kept_indices)))
        return kept_indices

    def test_nms_empty(self):
        bboxes = torch.tensor([], dtype=torch.float).cuda()
        scores = torch.tensor([], dtype=torch.float).cuda()
        kept_indices = self._run_nms(bboxes, scores)
        self.assertEqual(len(kept_indices), 0)

    def test_nms_single(self):
        bboxes = torch.tensor([[5, 5, 10, 10]], dtype=torch.float).cuda()
        scores  = torch.tensor([0.8], dtype=torch.float).cuda()
        kept_indices = self._run_nms(bboxes, scores)
        self.assertEqual(len(kept_indices), 1)
        self.assertListEqual(kept_indices.tolist(), [0])

    def test_nms_small(self):
        bboxes = torch.tensor([[5, 5, 10, 10], [5, 5, 10, 10], [5, 5, 30, 30]], dtype=torch.float).cuda()
        scores = torch.tensor([0.6, 0.9, 0.4], dtype=torch.float).cuda()
        kept_indices = self._run_nms(bboxes, scores)
        self.assertEqual(len(kept_indices), 2)
        self.assertListEqual(kept_indices.tolist(), [1, 2])

    def test_nms_large(self):
        # detections format: [[left, top, right, bottom, score], ...], which (right, bottom) is included in area
        cur_dir = os.path.dirname(os.path.abspath(__file__))
        detections = np.load(os.path.join(cur_dir, 'nms-large-input.npy'))
        detections = torch.tensor(detections, dtype=torch.float).cuda()
        bboxes = detections[:, 0:4]
        scores = detections[:, 4]

        kept_indices = self._run_nms(bboxes, scores)
        self.assertEqual(len(kept_indices), 1934)

        expect = np.load(os.path.join(cur_dir, 'nms-large-output.npy'))
        self.assertListEqual(sorted(kept_indices.tolist()),
                             sorted(expect.tolist()))


if __name__ == '__main__':
    assert torch.cuda.is_available(), 'NMS module requires CUDA support'
    torch.tensor([]).cuda()  # dummy for initializing GPU
    unittest.main()
