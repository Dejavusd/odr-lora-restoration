import copy
from pathlib import Path
import unittest

import numpy as np
import torch

from src.data.clean_protocol import sha256_file
from src.models.task_memory import bank_hash
from src.scripts.summarize_task_replicates import summarize_runs
from src.trainers.clean_sequence_trainer import CleanSequenceTrainer, run_protocol, validate_resume
from src.trainers.memory_source_trainer import non_memory_fingerprint
from src.trainers.task_balanced_trainer import TaskBalancedTrainer, task_balanced_config
from src.trainers.task_replicate_trainer import TaskReplicateTrainer, task_replicate_config, validate_rainh_source
from src.utils.training_state import atomic_json
from tests import test_clean_sequence as fixtures


class GuardedReplicate(TaskReplicateTrainer):
    def loader(self, task, role):
        if role == 'train' and task != 'dehaze':
            raise AssertionError('Old TRAIN accessed by Haze4K replicate')
        return super().loader(task, role)

    def bootstrap_memory(self):
        raise AssertionError('Replication must not rebuild old summaries from images')


class TaskReplicateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        f = cls.fixture = fixtures.CleanSequenceTests()
        f.setUp()
        cfg = f.config_at('replicate_origin')
        cfg['model']['architecture'] = 'stable_v1'
        cfg['train']['epochs_per_task'] = 1
        first = CleanSequenceTrainer(cfg, f.manifest); first.run(stop_after='lowlight')
        source0 = Path(first.output_dir)/'task_0_final.pth'
        config = task_balanced_config(source0, f.root/'replicate_rainh', epochs=2, batch_size=2)
        config['eval']['validation_interval'] = 1
        original = TaskBalancedTrainer(config, f.manifest)
        original.run(stop_after='rainh', resume_path=source0)
        cls.source = Path(original.output_dir)/'task_1_final.pth'
        cls.source_state = torch.load(cls.source, weights_only=False)

    @classmethod
    def tearDownClass(cls):
        cls.fixture.tearDown()

    def config(self, name, mode='balanced', seed=7):
        return task_replicate_config(self.source, self.fixture.root/name, mode, seed, epochs=2, batch_size=2)

    def test_reselect_same_physical_weights_snapshots_without_images(self):
        models, banks = [], []
        initial_sha = sha256_file(self.source)
        for mode in ('balanced', 'mean_pca'):
            trainer = GuardedReplicate(self.config('reselect_'+mode, mode), self.fixture.manifest)
            state = trainer.restore(self.source)
            models.append(non_memory_fingerprint(trainer.model)); banks.append(bank_hash(trainer.task_sketches))
            self.assertEqual(state['task_idx'], 1)
            self.assertTrue(state['gpm_updated'])
            self.assertEqual(trainer.history, [])
        self.assertEqual(models[0], models[1]); self.assertEqual(banks[0], banks[1])
        self.assertEqual(initial_sha, sha256_file(self.source))
        invalid = copy.deepcopy(self.source_state); invalid['smoke_test'] = True
        with self.assertRaises(ValueError): validate_rainh_source(invalid)
        invalid = copy.deepcopy(self.source_state); invalid['task_idx'] = 2
        with self.assertRaises(ValueError): validate_rainh_source(invalid)
        invalid = copy.deepcopy(self.source_state)
        del invalid['task_sketches'][next(iter(invalid['task_sketches']))]['0']
        with self.assertRaises(ValueError): validate_rainh_source(invalid)

    def test_seed42_reproduces_original_haze_entry(self):
        cfg = copy.deepcopy(self.source_state['run_protocol'])
        cfg['experiment']['output_dir'] = str(self.fixture.root/'parity_original')
        original = TaskBalancedTrainer(cfg, self.fixture.manifest)
        original.run(resume_path=self.source)
        replica = GuardedReplicate(self.config('parity_replica', seed=42), self.fixture.manifest)
        replica.run(resume_path=self.source)
        self.assertEqual([r for r in original.history if r['task']=='dehaze'], replica.history)
        self.assertEqual(bank_hash(original.task_sketches), bank_hash(replica.task_sketches))
        for name, value in original.model.state_dict().items():
            self.assertTrue(torch.equal(value, replica.model.state_dict()[name]), name)

    def test_seeded_epoch_resume_and_seed_mode_rejection(self):
        cfg = self.config('seed_reference')
        full = GuardedReplicate(cfg, self.fixture.manifest); full.run(resume_path=self.source)
        interrupted_cfg = self.config('seed_interrupted')
        partial = GuardedReplicate(interrupted_cfg, self.fixture.manifest)
        save = partial.save
        def disconnect(path, idx, epoch, *args, **kwargs):
            save(path, idx, epoch, *args, **kwargs)
            if Path(path).name=='latest.pth' and idx==2:
                raise RuntimeError('disconnect')
        partial.save = disconnect
        with self.assertRaisesRegex(RuntimeError, 'disconnect'):
            partial.run(resume_path=self.source)
        latest = self.fixture.root/'seed_interrupted/latest.pth'
        state = torch.load(latest, weights_only=False)
        for changed in (self.config('seed_interrupted', seed=2026),
                        self.config('seed_interrupted', mode='mean_pca')):
            with self.assertRaisesRegex(ValueError, '训练协议'):
                validate_resume(changed, state, state['split_sha256'])
        resumed = GuardedReplicate(interrupted_cfg, self.fixture.manifest)
        resumed.run(resume_path=latest)
        self.assertEqual(full.history, resumed.history)
        self.assertEqual(bank_hash(full.task_sketches), bank_hash(resumed.task_sketches))
        for name, value in full.model.state_dict().items():
            self.assertTrue(torch.equal(value, resumed.model.state_dict()[name]), name)

    def test_summary_paired_statistics_and_rejections(self):
        paths = []; manifests = {}
        core = ('src/models/task_memory.py', 'src/models/odr_lora.py', 'src/models/restoration_net.py',
                'src/trainers/task_balanced_trainer.py', 'src/trainers/memory_source_trainer.py',
                'src/trainers/clean_sequence_trainer.py')
        for seed, gain in [(7, .1), (2026, .3)]:
            for mode in ('balanced', 'mean_pca'):
                cfg = self.config(f'summary_{seed}_{mode}', mode, seed)
                trainer = GuardedReplicate(cfg, self.fixture.manifest); trainer.restore(self.source)
                path = Path(trainer.output_dir); paths.append(path)
                protocol = run_protocol(cfg)
                run = dict(protocol=protocol, split_sha256=trainer.split_sha,
                           source_code_sha256={k: sha256_file(k) for k in core})
                manifests[path] = run
                atomic_json(path/'run_manifest.json', run)
                atomic_json(path/'status.json', dict(status='complete', stopped_after='dehaze', smoke_test=False))
                matrix = copy.deepcopy(self.source_state['performance_matrix'])
                matrix['2'] = {t: dict(psnr=v+(gain if mode=='balanced' else 0), ssim=.5, lpips=.3, num=2)
                               for t,v in zip(('lowlight','rainh','dehaze'), (18.,16.,20.))}
                atomic_json(path/'continual_metrics.json', dict(matrix=matrix, latest=dict(stage=2,
                    acc_psnr=float(np.mean([r['psnr'] for r in matrix['2'].values()])), forgetting_psnr=2.)))
        report = summarize_runs(paths, self.source)
        self.assertEqual(report['n'], 2)
        self.assertAlmostEqual(report['paired_difference_stats']['mean_psnr']['mean'], .2)
        self.assertAlmostEqual(report['paired_difference_stats']['mean_psnr']['sample_std'], np.std([.1,.3],ddof=1))
        with self.assertRaisesRegex(ValueError, 'Every seed'):
            summarize_runs(paths[:-1], self.source)
        changed = copy.deepcopy(manifests[paths[-1]])
        changed['source_code_sha256']['src/models/task_memory.py'] = 'changed'
        atomic_json(paths[-1]/'run_manifest.json', changed)
        with self.assertRaisesRegex(ValueError, 'code changed'):
            summarize_runs(paths, self.source)
        changed = copy.deepcopy(manifests[paths[-1]]); changed['protocol']['smoke_test'] = True
        atomic_json(paths[-1]/'run_manifest.json', changed)
        with self.assertRaisesRegex(ValueError, 'Smoke'):
            summarize_runs(paths, self.source)


if __name__ == '__main__': unittest.main()
