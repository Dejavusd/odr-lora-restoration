import copy
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

from src.data.clean_protocol import prepare_splits, load_manifest, sha256_file
from src.data.datasets import RestorationDataset
from src.scripts.train_clean_sequence import configure_method
from src.trainers.clean_sequence_trainer import CleanSequenceTrainer, validate_resume, run_protocol
from src.utils.training_state import atomic_json, atomic_write, capture_rng, restore_rng, seed_all


class StubLPIPS(torch.nn.Module):
    """Unit-test dependency stub only; real-data GPU smoke uses real LPIPS/VGG."""
    def forward(self, x, y):
        return (x-y).abs().mean().reshape(1)


class StubLoss(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.perceptual = torch.nn.Module()
        self.perceptual.use_lpips = True
        self.perceptual.lpips = StubLPIPS()

    def forward(self, x, y):
        return (x-y).abs().mean()


class CleanSequenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        generator = np.random.default_rng(27)
        tasks = [('lowlight', 'LSRW'), ('rainh', 'RainH'), ('dehaze', 'Haze4K')]
        for _, dataset in tasks:
            for split, groups in [('train', 4), ('test', 1)]:
                base = self.root/'data'/dataset/split
                for folder in ('input', 'gt'):
                    (base/folder).mkdir(parents=True)
                for group in range(groups):
                    gt = generator.integers(20, 230, (16, 16, 3), dtype=np.uint8)
                    for variant in range(2):
                        name = f'{group}_{variant}.png'
                        Image.fromarray(gt).save(base/'gt'/name)
                        Image.fromarray((gt.astype(np.float32)*(.5+.1*variant)).astype(np.uint8)).save(base/'input'/name)
        self.cfg = dict(
            experiment=dict(output_dir=str(self.root/'run'), seed=42, incremental_seed=42,
                            device='cpu', strict_fp32=True, memory_fraction=.5),
            data=dict(root=str(self.root/'data'), split_manifest=str(self.root/'splits.json'),
                      split_seed=31, validation_fraction=.25, num_workers=0,
                      tasks=[dict(name=n, dataset=d, patch_size=16, batch_size=2) for n,d in tasks]),
            model=dict(base_channels=4, num_blocks=[1,1], kernel_size=3, share_rank=2,
                       private_rank_base=2, rank_gamma=1.5, rank_min=2, rank_max=4),
            odr_lora=dict(spatial_blocks=[1,1], spatial_sample_ratio=.5, freq_level=1,
                          freq_decomp='dwt', adaptive_lambda_min=.5, adaptive_lambda_max=1.,
                          skip_orthogonal=True, share_subspace=True, gpm_max_basis=2,
                          gpm_statistics='frequency'),
            train=dict(epochs_per_task=3, max_steps_per_epoch=1, lr=2e-4, weight_decay=1e-4,
                       grad_clip=1., calibration_batches=1, gpm_batches=1, eta_min=1e-6),
            eval=dict(validation_interval=1, max_images=1, tile_size=0, tile_overlap=0), smoke_test=False)
        self.cfg = configure_method(self.cfg, 'effective_r16')
        self.manifest = prepare_splits(self.cfg, workers=1)
        self.stub = patch('src.trainers.incremental_trainer.TotalLoss', StubLoss)
        self.stub.start()
        torch.set_num_threads(2)

    def tearDown(self):
        self.stub.stop()
        self.temp.cleanup()

    def config_at(self, name):
        cfg = copy.deepcopy(self.cfg)
        cfg['experiment']['output_dir'] = str(self.root/name)
        return cfg

    def test_grouped_manifest_and_file_hash_verification(self):
        manifest = load_manifest(self.cfg, verify_hashes=True)
        for item in manifest['tasks'].values():
            self.assertEqual((item['train_pairs'], item['validation_pairs'], item['test_pairs']), (6,2,2))
        self.assertEqual(manifest, self.manifest)

    def test_manifest_detects_data_mutation(self):
        path = Path(self.manifest['tasks']['lowlight']['train_records'][0]['input'])
        path.write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, '数据发生改变'):
            load_manifest(self.cfg)

    def test_manifest_detects_group_leak(self):
        item = self.manifest['tasks']['lowlight']
        index = item['validation_indices'][0]
        item['train_records'][index]['gt_group'] = item['train_records'][item['train_indices'][0]]['gt_group']
        atomic_json(self.cfg['data']['split_manifest'], self.manifest)
        with self.assertRaisesRegex(ValueError, '泄漏'):
            load_manifest(self.cfg)

    def test_strict_size_never_silently_resizes_gt(self):
        row = self.manifest['tasks']['lowlight']['train_records'][0]
        Image.new('RGB', (17,16)).save(row['gt'])
        data = RestorationDataset([row['input']], [row['gt']], augment=False, strict_size=True)
        with self.assertRaisesRegex(ValueError, '尺寸不一致'):
            data[0]

    def test_atomic_failure_preserves_previous_checkpoint(self):
        path = self.root/'state.pth'
        path.write_bytes(b'previous complete state')
        def fail(handle):
            handle.write(b'partial')
            raise OSError('simulated disk failure')
        with self.assertRaises(OSError):
            atomic_write(path, fail)
        self.assertEqual(path.read_bytes(), b'previous complete state')
        self.assertFalse(list(self.root.glob('state.pth.*.tmp')))

    def test_rng_roundtrip(self):
        seed_all(17)
        state = capture_rng()
        expected = (random.random(), np.random.rand(), torch.rand(3))
        restore_rng(state)
        self.assertEqual(random.random(), expected[0])
        self.assertEqual(np.random.rand(), expected[1])
        self.assertTrue(torch.equal(torch.rand(3), expected[2]))

    def test_epoch_resume_matches_uninterrupted_training(self):
        cfg_a, cfg_b = self.config_at('continuous'), self.config_at('interrupted')
        seed_all(42)
        full = CleanSequenceTrainer(cfg_a, self.manifest)
        full.run(stop_after='lowlight')
        seed_all(42)
        partial = CleanSequenceTrainer(cfg_b, self.manifest)
        save = partial.save
        def interrupted(path, idx, epoch, *args, **kwargs):
            save(path, idx, epoch, *args, **kwargs)
            if Path(path).name == 'latest.pth' and epoch == 1:
                raise RuntimeError('simulated disconnect')
        with patch.object(partial, 'save', side_effect=interrupted):
            with self.assertRaisesRegex(RuntimeError, 'simulated disconnect'):
                partial.run(stop_after='lowlight')
        latest = self.root/'interrupted/latest.pth'
        saved = torch.load(latest, weights_only=False)
        self.assertTrue(saved['in_progress'])
        self.assertIsNotNone(saved['scheduler'])
        resumed = CleanSequenceTrainer(cfg_b, self.manifest)
        resumed.run(stop_after='lowlight', resume_path=latest)
        self.assertEqual(full.history, resumed.history)
        for name, value in full.model.state_dict().items():
            self.assertTrue(torch.equal(value, resumed.model.state_dict()[name]), name)
        best_score = max(row['validation']['psnr'] for row in full.history)
        self.assertEqual(full.best_psnr, best_score)
        final = torch.load(self.root/'interrupted/task_0_final.pth', weights_only=False)
        self.assertTrue(final['gpm_updated'])
        self.assertFalse(final['in_progress'])

    def test_task0_branches_and_three_task_export(self):
        cfg = self.config_at('source')
        seed_all(42)
        source = CleanSequenceTrainer(cfg, self.manifest)
        source.run(stop_after='lowlight')
        checkpoint = self.root/'source/task_0_final.pth'
        branch_cfg = configure_method(self.config_at('branch'), 'current_basis')
        branch = CleanSequenceTrainer(branch_cfg, self.manifest)
        branch.run(resume_path=checkpoint)
        final = torch.load(self.root/'branch/task_2_final.pth', weights_only=False)
        self.assertEqual(final['num_tasks'], 3)
        self.assertTrue(final['gpm_updated'])
        self.assertEqual(set(final['performance_matrix']), {'0','1','2'})
        exported = torch.load(self.root/'branch/merged.pth', weights_only=False)
        self.assertTrue(exported['inference_only'])
        with self.assertRaises(ValueError):
            validate_resume(branch_cfg, exported, sha256_file(self.cfg['data']['split_manifest']))
        for idx in (1,2):
            transition = json.loads((self.root/f'branch/task_{idx}_gpm_transition.json').read_text())
            self.assertEqual(transition['gpm_forward_max_error'], 0.)
            self.assertTrue((self.root/f'branch/task_{idx}_protection.json').exists())

    def test_resume_rejects_old_and_probe_models(self):
        sha = sha256_file(self.cfg['data']['split_manifest'])
        for state in [{}, dict(clean_protocol_version=1, probe_only=True),
                      dict(clean_protocol_version=1, smoke_test=True)]:
            with self.assertRaises(ValueError):
                validate_resume(self.cfg, state, sha)

    def test_only_completed_task0_may_change_method(self):
        sha = sha256_file(self.cfg['data']['split_manifest'])
        saved = dict(clean_protocol_version=1, split_sha256=sha, run_protocol=run_protocol(self.cfg),
                     task_idx=0, in_progress=True, smoke_test=False, gpm_updated=False)
        changed = configure_method(self.cfg, 'current_basis')
        with self.assertRaisesRegex(ValueError, '训练协议'):
            validate_resume(changed, saved, sha)
        saved.update(in_progress=False, gpm_updated=True)
        validate_resume(changed, saved, sha)
        saved['task_idx'] = 1
        with self.assertRaises(ValueError):
            validate_resume(changed, saved, sha)

    def test_resume_requires_identical_split_and_updated_memory(self):
        sha = sha256_file(self.cfg['data']['split_manifest'])
        saved = dict(clean_protocol_version=1, split_sha256=sha, run_protocol=run_protocol(self.cfg),
                     task_idx=0, in_progress=False, smoke_test=False, gpm_updated=False)
        with self.assertRaisesRegex(ValueError, 'GPM'):
            validate_resume(self.cfg, saved, sha)
        saved['gpm_updated'] = True
        with self.assertRaisesRegex(ValueError, 'SHA256'):
            validate_resume(self.cfg, saved, 'different')

    def test_preparer_rejects_train_test_gt_overlap(self):
        row = self.manifest['tasks']['lowlight']['train_records'][0]
        target = Path(self.manifest['tasks']['lowlight']['test_records'][0]['gt'])
        target.write_bytes(Path(row['gt']).read_bytes())
        cfg = copy.deepcopy(self.cfg)
        cfg['data']['split_manifest'] = str(self.root/'other_split.json')
        with self.assertRaisesRegex(ValueError, '训练/测试存在'):
            prepare_splits(cfg, workers=1)


if __name__ == '__main__':
    unittest.main()
