from pathlib import Path
import tempfile
import unittest

from src.data.datasets import pair_images


class ImagePairingTests(unittest.TestCase):
    def pair(self, inputs, targets):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for folder, names in [('input', inputs), ('gt', targets)]:
                (root / folder).mkdir()
                for name in names:
                    (root / folder / name).touch()
            a, b = pair_images(str(root), str(root))
            return [(Path(x).name, Path(y).name) for x, y in zip(a, b)]

    def test_zero_padded_distinct_images_use_exact_names(self):
        self.assertEqual(self.pair(['00001.jpg', '1.jpg'], ['1.jpg', '00001.jpg']),
                         [('00001.jpg', '00001.jpg'), ('1.jpg', '1.jpg')])

    def test_stems_allow_different_extensions(self):
        self.assertEqual(self.pair(['001.png', '1.png'], ['1.jpg', '001.jpg']),
                         [('001.png', '001.jpg'), ('1.png', '1.jpg')])

    def test_unique_numeric_ids_allow_many_degradations(self):
        self.assertEqual(self.pair(['1_0.8_0.1.png', '1_0.9_0.2.png'], ['0001.jpg']),
                         [('1_0.8_0.1.png', '0001.jpg'), ('1_0.9_0.2.png', '0001.jpg')])

    def test_ambiguous_numeric_ids_fail(self):
        with self.assertRaisesRegex(RuntimeError, '歧义'):
            self.pair(['hazy1.png'], ['0001.jpg', '1.jpg'])

    def test_incomplete_matches_fail(self):
        with self.assertRaises(RuntimeError):
            self.pair(['1.png', '2.png'], ['1.png'])
        with self.assertRaisesRegex(RuntimeError, '未配对'):
            self.pair(['1.png'], ['1.png', '2.png'])

    def test_unrelated_equal_counts_do_not_sort_pair(self):
        with self.assertRaises(RuntimeError):
            self.pair(['aaa.png'], ['bbb.png'])


if __name__ == '__main__':
    unittest.main()
