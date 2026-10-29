import tempfile
import unittest
from pathlib import Path

from PIL import Image
from src.data.datasets import pair_gopro_sequences, RestorationDataset


class GoProTests(unittest.TestCase):
    def test_repeated_frame_names_stay_in_sequence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for sequence, color in [('sequence_a', 30), ('sequence_b', 220)]:
                for folder in ('blur', 'blur_gamma', 'sharp'):
                    target = root / sequence / folder
                    target.mkdir(parents=True)
                    Image.new('RGB', (32, 32), (color,) * 3).save(target / '000001.png')
            inputs, targets = pair_gopro_sequences(directory)
            self.assertEqual(len(inputs), 2)
            self.assertTrue(all(Path(a).parent.parent == Path(b).parent.parent
                                for a, b in zip(inputs, targets)))
            ds = RestorationDataset(inputs, targets, patch_size=16)
            self.assertEqual(tuple(ds[0]['input'].shape), (3, 16, 16))
            gamma, _ = pair_gopro_sequences(directory, 'blur_gamma')
            self.assertEqual(Path(gamma[0]).parent.name, 'blur_gamma')
            (root / 'sequence_a' / 'sharp' / '000001.png').unlink()
            with self.assertRaises(RuntimeError):
                pair_gopro_sequences(directory)


if __name__ == '__main__':
    unittest.main()
