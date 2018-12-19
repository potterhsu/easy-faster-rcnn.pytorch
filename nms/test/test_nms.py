import os
import time
import unittest

import numpy as np
import torch

from nms.nms import NMS


class TestNMS(unittest.TestCase):
    def _run_nms(self, bboxes):
        start = time.time()
        kept_indices = NMS.suppress(bboxes.contiguous(), threshold=0.7)
        print('%s in %.3fs, %d -> %d' % (self.id(), time.time() - start, len(bboxes), len(kept_indices)))
        return kept_indices

    def test_nms_empty(self):
        bboxes = torch.FloatTensor().cuda()
        kept_indices = self._run_nms(bboxes)
        self.assertEqual(len(kept_indices), 0)

    def test_nms_single(self):
        bboxes = torch.FloatTensor([[5, 5, 10, 10]]).cuda()
        kept_indices = self._run_nms(bboxes)
        self.assertEqual(len(kept_indices), 1)
        self.assertListEqual(kept_indices.tolist(), [0])

    def test_nms_small(self):
        # bboxes = torch.FloatTensor([[5, 5, 10, 10], [5, 5, 10, 10], [5, 5, 30, 30]]).cuda()
        bboxes = torch.FloatTensor([[5, 5, 10, 10], [5, 5, 30, 30]]).cuda()
        kept_indices = self._run_nms(bboxes)
        self.assertEqual(len(kept_indices), 2)
        # self.assertListEqual(kept_indices.tolist(), [0, 2])
        self.assertListEqual(kept_indices.tolist(), [0, 1])

    def test_nms_large(self):
        # detections format: [[left, top, right, bottom, score], ...], which (right, bottom) is included in area
        detections = np.load(os.path.join('nms', 'test', 'nms-large-input.npy'))
        bboxes = torch.FloatTensor(detections).cuda()
        sorted_indices = torch.sort(bboxes[:, 4], dim=0, descending=True)[1]
        bboxes = bboxes[:, 0:4][sorted_indices]

        # point of (right, bottom) in our bbox definition is not included in area
        bboxes[:, 2] += 1
        bboxes[:, 3] += 1

        kept_indices = self._run_nms(bboxes)
        kept_indices_for_detection = sorted_indices[kept_indices]
        self.assertEqual(len(kept_indices_for_detection), 1934)

        expect = np.load(os.path.join('nms', 'test', 'nms-large-output.npy'))
        self.assertListEqual(kept_indices_for_detection.tolist(), expect.tolist())


if __name__ == '__main__':
    assert torch.cuda.is_available(), 'NMS module requires CUDA support'
    torch.FloatTensor().cuda()  # dummy for initializing GPU
    unittest.main()
