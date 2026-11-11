import copy
import json
import os
from pathlib import Path
import unittest

import torch

from src.data.clean_protocol import sha256_file
from src.data.five_task_protocol import load_extension, prepare_extension
from src.models.task_memory import bank_hash, sketch_hash
from src.scripts.train_task_memory_extension import CODE_FILES, ROOT
from src.scripts.summarize_task_memory_extension import summarize_extensions
from src.trainers.clean_sequence_trainer import CleanSequenceTrainer, validate_resume
from src.trainers.task_balanced_trainer import TaskBalancedTrainer, task_balanced_config
from src.trainers.task_memory_extension_trainer import (
    TaskMemoryExtensionTrainer, task_memory_extension_config, validate_task_memory_source)
from src.utils.training_state import atomic_json
from tests import test_clean_sequence as fixtures
from tests import test_five_tasks as data_fixtures


class GuardedExtension(TaskMemoryExtensionTrainer):
    active_task = None

    def train_task(self, task, *args, **kwargs):
        self.active_task = task
        try:
            return super().train_task(task, *args, **kwargs)
        finally:
            self.active_task = None

    def loader(self, task, role):
        if role == 'train' and (task != self.active_task or task not in ('denoise', 'deblur')):
            raise AssertionError('Old TRAIN accessed by task-memory extension')
        return super().loader(task, role)


class TaskMemoryExtensionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        f = cls.fixture = fixtures.CleanSequenceTests()
        f.setUp()
        data_fixtures.FiveTaskTests.extra_data(f)
        cls.extended = f.root/'five.json'
        prepare_extension(f.cfg['data']['split_manifest'], cls.extended, 1)
        cfg = f.config_at('stable_source')
        cfg['model']['architecture'] = 'stable_v1'
        cfg['train']['epochs_per_task'] = 1
        first = CleanSequenceTrainer(cfg, f.manifest)
        first.run(stop_after='lowlight')
        source0 = Path(first.output_dir)/'task_0_final.pth'
        cls.sources = {}
        for mode in ('balanced', 'mean_pca'):
            cfg = task_balanced_config(source0, f.root/mode, mode, epochs=1, batch_size=2)
            trainer = TaskBalancedTrainer(cfg, f.manifest)
            trainer.run(resume_path=source0)
            cls.sources[mode] = Path(trainer.output_dir)/'task_2_final.pth'
        cls.source = cls.sources['balanced']

    @classmethod
    def tearDownClass(cls):
        cls.fixture.tearDown()

    def config(self, name, mode='balanced'):
        cfg = task_memory_extension_config(self.sources[mode], self.extended,
                                            self.fixture.root/name, epochs=2, batch_size=2)
        cfg['eval']['validation_interval'] = 1
        for task in cfg['data']['tasks'][3:]:
            task['patch_size'] = 16
        return cfg

    def test_source_validation_and_unchanged_import_for_both_modes(self):
        for mode in self.sources:
            source = self.sources[mode]
            source_sha = sha256_file(source)
            original = torch.load(source, weights_only=False)
            cfg = self.config('import_'+mode, mode)
            self.assertEqual(cfg['task_memory']['mode'], mode)
            trainer = GuardedExtension(cfg, load_extension(cfg))
            boundary = trainer.restore(source)
            self.assertEqual(trainer.history, original['history'])
            self.assertEqual(trainer.performance_matrix, original['performance_matrix'])
            self.assertEqual(bank_hash(trainer.task_sketches), original['task_sketches_sha256'])
            for name, value in original['model'].items():
                self.assertTrue(torch.equal(value, trainer.model.state_dict()[name]), name)
            self.assertEqual(boundary['task_idx'], 2)
            self.assertFalse(boundary['in_progress'])
            self.assertEqual(sha256_file(source), source_sha)
        original = torch.load(self.source, weights_only=False)
        for field, value in [('in_progress', True), ('smoke_test', True), ('gpm_updated', False),
                             ('task_idx', 1), ('task_memory_extension_version', 1)]:
            bad = copy.deepcopy(original)
            bad[field] = value
            with self.assertRaises(ValueError):
                validate_task_memory_source(bad)
        bad = copy.deepcopy(original)
        del bad['task_sketches'][next(iter(bad['task_sketches']))]['0']
        with self.assertRaisesRegex(ValueError, 'SHA256'):
            validate_task_memory_source(bad)
        for kind in ('empty', 'nan', 'oversized'):
            bad = copy.deepcopy(original)
            key = next(k for k in bad['model'] if k.endswith('.dual_gpms.global._basis'))
            basis = bad['model'][key]
            if kind == 'empty': bad['model'][key] = basis[:, :0]
            elif kind == 'nan': bad['model'][key] = basis*float('nan')
            else: bad['model'][key] = torch.ones(basis.shape[0], 33)
            with self.assertRaisesRegex(ValueError, 'source GPM'):
                validate_task_memory_source(bad)

    def test_no_old_training_and_bit_exact_epoch_resume(self):
        cfg = self.config('full_extension')
        full = GuardedExtension(cfg, load_extension(cfg))
        full.run(stop_after='denoise', resume_path=self.source)
        original = torch.load(self.source, weights_only=False)
        for name, entries in original['task_sketches'].items():
            for task, entry in entries.items():
                self.assertEqual(sketch_hash(entry), sketch_hash(full.task_sketches[name][task]))
            self.assertEqual(set(full.task_sketches[name]), {'0', '1', '2', '3'})
        interrupted_cfg = self.config('interrupted_extension')
        partial = GuardedExtension(interrupted_cfg, load_extension(interrupted_cfg))
        save = partial.save
        def disconnect(path, idx, epoch, *args, **kwargs):
            save(path, idx, epoch, *args, **kwargs)
            if Path(path).name == 'latest.pth' and idx == 3 and epoch == 1:
                raise RuntimeError('epoch disconnect')
        partial.save = disconnect
        with self.assertRaisesRegex(RuntimeError, 'epoch disconnect'):
            partial.run(stop_after='denoise', resume_path=self.source)
        latest = self.fixture.root/'interrupted_extension/latest.pth'
        state = torch.load(latest, weights_only=False)
        for change in ('mode', 'batch', 'seed', 'source'):
            bad = copy.deepcopy(interrupted_cfg)
            if change == 'mode': bad['task_memory']['mode'] = 'mean_pca'
            elif change == 'batch': bad['data']['tasks'][3]['batch_size'] += 1
            elif change == 'seed': bad['experiment']['incremental_seed'] += 1
            else: bad['task_memory_extension']['source_sha256'] = 'wrong'
            with self.assertRaisesRegex(ValueError, '训练协议'):
                validate_resume(bad, state, state['split_sha256'])
        resumed = GuardedExtension(interrupted_cfg, load_extension(interrupted_cfg))
        resumed.run(stop_after='denoise', resume_path=latest)
        self.assertEqual(full.history, resumed.history)
        self.assertEqual(bank_hash(full.task_sketches), bank_hash(resumed.task_sketches))
        self.assertEqual(full.performance_matrix, resumed.performance_matrix)
        for name, value in full.model.state_dict().items():
            self.assertTrue(torch.equal(value, resumed.model.state_dict()[name]), name)

    def test_boundary_interruption_and_optional_gopro_continuation(self):
        cfg = self.config('boundary_reference')
        full = GuardedExtension(cfg, load_extension(cfg))
        full.run(stop_after='denoise', resume_path=self.source)
        interrupted_cfg = self.config('boundary_interrupted')
        partial = GuardedExtension(interrupted_cfg, load_extension(interrupted_cfg))
        finish = partial.finalize_memory
        def disconnect(idx):
            finish(idx)
            raise RuntimeError('memory disconnect')
        partial.finalize_memory = disconnect
        with self.assertRaisesRegex(RuntimeError, 'memory disconnect'):
            partial.run(stop_after='denoise', resume_path=self.source)
        resumed = GuardedExtension(interrupted_cfg, load_extension(interrupted_cfg))
        resumed.run(stop_after='denoise', resume_path=Path(partial.output_dir)/'latest.pth')
        self.assertEqual(full.history, resumed.history)
        self.assertEqual(bank_hash(full.task_sketches), bank_hash(resumed.task_sketches))
        for name, value in full.model.state_dict().items():
            self.assertTrue(torch.equal(value, resumed.model.state_dict()[name]), name)
        tail_cfg = self.config('gopro_tail')
        tail = GuardedExtension(tail_cfg, load_extension(tail_cfg))
        tail.run(stop_after='deblur', resume_path=Path(full.output_dir)/'task_3_final.pth')
        self.assertEqual(set(tail.performance_matrix), {'0', '1', '2', '3', '4'})
        self.assertTrue(all(set(entries) == {'0', '1', '2', '3', '4'} for entries in tail.task_sketches.values()))

    def test_reject_changed_parent_manifest_and_invalid_code_paths(self):
        for path in CODE_FILES:
            self.assertTrue((ROOT/path).is_file(), path)
        manifest = json.loads(self.extended.read_text())
        manifest['parent_sha256'] = 'changed'
        bad_path = self.fixture.root/'bad_parent.json'
        atomic_json(bad_path, manifest)
        with self.assertRaisesRegex(ValueError, 'parent split'):
            task_memory_extension_config(self.source, bad_path, self.fixture.root/'bad_output')
        cfg = self.config('corrupt_resume')
        trainer = GuardedExtension(cfg, load_extension(cfg))
        state = trainer.restore(self.source)
        state['task_sketches'][next(iter(state['task_sketches']))]['1']['factor'][0, 0] += 1
        path = self.fixture.root/'corrupt_extension.pth'
        torch.save(state, path)
        with self.assertRaisesRegex(ValueError, 'SHA256'):
            GuardedExtension(cfg, load_extension(cfg)).restore(path)
        state['task_balanced_version'] = 0
        torch.save(state, path)
        with self.assertRaisesRegex(ValueError, 'task-summary version'):
            GuardedExtension(cfg, load_extension(cfg)).restore(path)

    @unittest.skipUnless(os.environ.get('ODR_GPU_EXTENSION_TEST') == '1', 'Opt-in GPU resume check')
    def test_gpu_epoch_resume_and_memory_devices(self):
        self.assertTrue(torch.cuda.is_available())
        cfg = self.config('gpu_extension')
        cfg['experiment']['device'] = 'cuda'
        full = GuardedExtension(cfg, load_extension(cfg))
        full.run(stop_after='denoise', resume_path=self.source)
        partial_cfg = copy.deepcopy(cfg)
        partial_cfg['experiment']['output_dir'] = str(self.fixture.root/'gpu_interrupted')
        partial = GuardedExtension(partial_cfg, load_extension(partial_cfg))
        save = partial.save
        def disconnect(path, idx, epoch, *args, **kwargs):
            save(path, idx, epoch, *args, **kwargs)
            if Path(path).name == 'latest.pth' and idx == 3 and epoch == 1:
                raise RuntimeError('GPU disconnect')
        partial.save = disconnect
        with self.assertRaisesRegex(RuntimeError, 'GPU disconnect'):
            partial.run(stop_after='denoise', resume_path=self.source)
        resumed = GuardedExtension(partial_cfg, load_extension(partial_cfg))
        resumed.run(stop_after='denoise', resume_path=Path(partial.output_dir)/'latest.pth')
        self.assertEqual(full.history, resumed.history)
        for name, value in full.model.state_dict().items():
            self.assertTrue(torch.equal(value, resumed.model.state_dict()[name]), name)
        for module in resumed.model.modules():
            if hasattr(module, 'dual_gpms'):
                self.assertEqual(module.dual_gpms['global'].basis.device, module.weight.device)
        self.assertTrue(all(e['factor'].device.type == 'cpu'
                            for entries in resumed.task_sketches.values() for e in entries.values()))

    def test_summary_reports_own_boundary_drop_and_rejects_unmatched_runs(self):
        paths = []
        for mode in ('balanced', 'mean_pca'):
            cfg = self.config('summary_extension_'+mode, mode)
            cfg['eval']['max_images'] = None
            trainer = GuardedExtension(cfg, load_extension(cfg))
            trainer.run(stop_after='denoise', resume_path=self.sources[mode])
            path = Path(trainer.output_dir)
            paths.append(path)
            atomic_json(path/'run_manifest.json', dict(protocol=trainer.protocol, split_sha256=trainer.split_sha,
                source_code_sha256={f: sha256_file(ROOT/f) for f in CODE_FILES}))
        report = summarize_extensions(paths)
        self.assertEqual(set(report['runs']), {'balanced', 'mean_pca'})
        for mode in ('balanced', 'mean_pca'):
            self.assertIn('denoise_BSD68_psnr', report['runs'][mode]['metrics'])
            state = torch.load(self.sources[mode], weights_only=False)
            before = sum(v['psnr'] for v in state['performance_matrix']['2'].values())/3
            metrics = report['runs'][mode]['metrics']
            self.assertAlmostEqual(metrics['old_mean_drop'], before-metrics['old_mean_psnr'])
        with self.assertRaisesRegex(ValueError, 'Both'):
            summarize_extensions(paths[:1])
        run = json.loads((paths[1]/'run_manifest.json').read_text())
        run['protocol']['experiment']['incremental_seed'] += 1
        atomic_json(paths[1]/'run_manifest.json', run)
        with self.assertRaisesRegex(ValueError, 'Unmatched'):
            summarize_extensions(paths)


if __name__ == '__main__':
    unittest.main()
