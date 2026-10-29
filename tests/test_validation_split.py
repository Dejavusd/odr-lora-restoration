import unittest
from unittest.mock import patch

import torch
from src.scripts.validate_effective_basis import grouped_split, check_merge


class GroupedValidationTests(unittest.TestCase):
    def test_repeated_targets_never_cross_partition(self):
        rows = [dict(gt_pixel_sha256=f'group-{i//6}') for i in range(120)]
        train, val = grouped_split(rows)
        self.assertEqual(len(train), 108)
        self.assertEqual(len(val), 12)
        self.assertEqual(set(train) | set(val), set(range(120)))
        self.assertFalse(set(train) & set(val))
        self.assertFalse({rows[i]['gt_pixel_sha256'] for i in train}
                         & {rows[i]['gt_pixel_sha256'] for i in val})
        self.assertEqual((train, val), grouped_split(rows))

    def test_one_group_cannot_be_split_safely(self):
        with self.assertRaises(ValueError):
            grouped_split([dict(gt_pixel_sha256='same')]*6)

    def test_merge_check_disables_tf32_and_restores_backend(self):
        before = (torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32)
        def inspect(*args):
            self.assertFalse(torch.backends.cudnn.allow_tf32)
            self.assertFalse(torch.backends.cuda.matmul.allow_tf32)
            return dict(passed=True)
        with patch('src.scripts.validate_effective_basis._check_merge', side_effect=inspect):
            self.assertTrue(check_merge(None, None, None)['passed'])
        self.assertEqual(before, (torch.backends.cudnn.allow_tf32,
                                  torch.backends.cuda.matmul.allow_tf32))

    def test_merge_failure_also_restores_backend(self):
        before = (torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32)
        with patch('src.scripts.validate_effective_basis._check_merge', side_effect=RuntimeError('test')):
            with self.assertRaisesRegex(RuntimeError, 'test'):
                check_merge(None, None, None)
        self.assertEqual(before, (torch.backends.cudnn.allow_tf32,
                                  torch.backends.cuda.matmul.allow_tf32))


if __name__ == '__main__':
    unittest.main()
