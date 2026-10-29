"""Run with: python -m unittest discover -s tests -p test_denoising_data.py"""
import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image

from src.data.datasets import GaussianDenoisingDataset
from src.utils.inference import predict_image


class DenoisingTests(unittest.TestCase):
    def test_noise_and_separate_numeric_names(self):
        with tempfile.TemporaryDirectory() as directory:
            for name in ('0001.png', '000001.png'):
                Image.new('RGB', (32, 32), (128, 128, 128)).save(Path(directory) / name)
            Path(directory, '.DS_Store').touch()
            ds = GaussianDenoisingDataset(directory, sigma=25, train=False)
            self.assertEqual(len(ds), 2)
            self.assertTrue(torch.equal(ds[0]['input'], ds[0]['input']))
            self.assertFalse(torch.equal(ds[0]['input'], ds[1]['input']))
            train = GaussianDenoisingDataset(directory, train=True, patch_size=16)
            self.assertEqual(train[0]['gt'].shape, (3, 16, 16))
            self.assertFalse(torch.equal(train[0]['input'], train[0]['input']))
            clean = GaussianDenoisingDataset(directory, sigma=0, train=False)[0]
            self.assertTrue(torch.equal(clean['input'], clean['gt']))

    def test_grayscale_noise(self):
        with tempfile.TemporaryDirectory() as directory:
            Image.new('L', (33, 49), 128).save(Path(directory) / 'test001.png')
            item = GaussianDenoisingDataset(directory, train=False)[0]
            self.assertTrue(item['grayscale'])
            self.assertTrue(torch.equal(item['input'][0], item['input'][1]))

    def test_tiling_preserves_shape_and_covers_edges(self):
        class Identity(torch.nn.Module):
            def forward(self, x, task_idx):
                return x
        for shape in ((1, 3, 19, 81), (1, 3, 71, 113), (1, 3, 16, 16)):
            image = torch.rand(shape)
            output = predict_image(Identity(), image, -1, 32, 8)
            self.assertTrue(torch.allclose(output, image, atol=1e-6))


if __name__ == '__main__':
    unittest.main()
