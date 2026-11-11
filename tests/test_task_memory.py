import copy
import json
from pathlib import Path
import unittest

import torch

from src.models.task_memory import TaskSketchAccumulator, select_task_memory, bank_hash, sketch_hash
from src.trainers.clean_sequence_trainer import CleanSequenceTrainer, validate_resume
from src.trainers.memory_source_trainer import non_memory_fingerprint
from src.trainers.task_balanced_trainer import TaskBalancedTrainer, task_balanced_config
from tests import test_clean_sequence as fixtures


def entry(factor):
    return dict(factor=factor.float(), columns=100, total_energy=float(factor.square().sum()))


class TaskMemoryTests(unittest.TestCase):
    def test_worst_first_coverage_and_capacity(self):
        bank = {'0': entry(torch.diag(torch.tensor([.6, .4, 0, 0, 0]).sqrt())),
                '1': entry(torch.diag(torch.tensor([0, 0, .34, .33, .33]).sqrt()))}
        mean, a = select_task_memory(bank, 2, 'mean_pca')
        balanced, b = select_task_memory(bank, 2, 'balanced')
        self.assertEqual(mean.shape, balanced.shape)
        self.assertLess(a['minimum_sketch_coverage'], 1e-6)
        self.assertGreater(b['minimum_sketch_coverage'], .33)
        torch.testing.assert_close(balanced.T @ balanced, torch.eye(2), atol=1e-6, rtol=1e-6)
        # Better worst-task coverage may sacrifice mean coverage; no global guarantee.
        self.assertLess(b['mean_sketch_coverage'], a['mean_sketch_coverage'])

    def test_single_task_equal_modes_and_shared_direction(self):
        bank = {'0': entry(torch.diag(torch.tensor([3., 2., 1., .5])))}
        a, _ = select_task_memory(bank, 2, 'mean_pca')
        b, _ = select_task_memory(bank, 2, 'balanced')
        self.assertTrue(torch.equal(a, b))
        shared = torch.tensor([[1.], [0.], [0.]])
        basis, info = select_task_memory({'0': entry(shared), '1': entry(shared)}, 3)
        self.assertEqual(basis.shape, (3, 1))
        self.assertAlmostEqual(info['minimum_sketch_coverage'], 1.)

    def test_task_energy_and_column_count_do_not_set_task_weights(self):
        bank = {'0': entry(torch.diag(torch.tensor([3., 2., .1]))),
                '1': entry(torch.tensor([[0., 0.], [2., 0.], [0., 3.]]))}
        baseline, _ = select_task_memory(bank, 2)
        modified = copy.deepcopy(bank)
        modified['1']['factor'] *= 100
        modified['1']['total_energy'] *= 10000
        modified['1']['columns'] *= 10
        result, _ = select_task_memory(modified, 2)
        torch.testing.assert_close(baseline @ baseline.T, result @ result.T, atol=1e-6, rtol=1e-6)

    def test_five_tasks_share_directions_without_five_private_bases(self):
        bank = {}
        for task in range(5):
            factor = torch.zeros(6, 2)
            factor[0, 0] = .8**.5
            factor[task+1, 1] = .2**.5
            bank[str(task)] = entry(factor)
        basis, info = select_task_memory(bank, 3)
        self.assertEqual(basis.shape, (6, 3))
        self.assertEqual(len(info['tasks']), 5)
        self.assertGreater(info['minimum_sketch_coverage'], .79)

    def test_streaming_sketch_bounded_and_covariance(self):
        torch.manual_seed(7)
        x = torch.randn(5, 17)
        sketch = TaskSketchAccumulator(5, 5)
        sketch.add(x[:, :7]); sketch.add(x[:, 7:])
        saved = sketch.finish()
        self.assertEqual(saved['columns'], 17)
        torch.testing.assert_close(saved['factor'] @ saved['factor'].T, x @ x.T, atol=2e-5, rtol=2e-5)
        small = TaskSketchAccumulator(5, 2)
        small.add(x)
        self.assertEqual(small.finish()['factor'].shape, (5, 2))
        self.assertLessEqual(float(small.factor.square().sum()), small.total_energy*(1+1e-5))
        with self.assertRaises(FloatingPointError):
            small.add(x*float('nan'))
        with self.assertRaises(ValueError):
            select_task_memory({'0': entry(x*0)}, 2)

    def test_training_common_rainh_no_replay_bank_and_exact_resume(self):
        fixture = fixtures.CleanSequenceTests()
        fixture.setUp()
        try:
            cfg = fixture.config_at('source')
            cfg['model']['architecture'] = 'stable_v1'
            cfg['train'].update(epochs_per_task=1, epochs_by_task=dict(lowlight=1, rainh=99))
            first = CleanSequenceTrainer(cfg, fixture.manifest)
            first.run(stop_after='lowlight')
            source = Path(first.output_dir)/'task_0_final.pth'

            class Guarded(TaskBalancedTrainer):
                bootstrap = False
                active_task = None
                def bootstrap_memory(self):
                    self.bootstrap = True
                    try: return super().bootstrap_memory()
                    finally: self.bootstrap = False
                def train_task(self, task, *args, **kwargs):
                    self.active_task = task
                    try: return super().train_task(task, *args, **kwargs)
                    finally: self.active_task = None
                def loader(self, task, role):
                    if role == 'train' and not (self.bootstrap and task == 'lowlight'):
                        if task != self.active_task:
                            raise AssertionError('Old task TRAIN accessed')
                    return super().loader(task, role)

            snapshots = []; columns = []; before_fingerprints = []
            config = None
            for mode in ('mean_pca', 'balanced'):
                config = task_balanced_config(source, fixture.root/mode, mode, epochs=2, batch_size=2)
                config['eval']['validation_interval'] = 1
                self.assertNotIn('epochs_by_task', config['train'])
                self.assertEqual(config['odr_lora']['skip_protection'], 'fused')
                trainer = Guarded(config, fixture.manifest)
                trainer.restore(source)
                before_fingerprints.append(non_memory_fingerprint(trainer.model))
                old_hashes = {n: sketch_hash(b['0']) for n, b in trainer.task_sketches.items()}
                trainer.run(stop_after='rainh', resume_path=fixture.root/mode/'task_0_final.pth')
                self.assertEqual(old_hashes, {n: sketch_hash(b['0']) for n, b in trainer.task_sketches.items()})
                snapshots.append(non_memory_fingerprint(trainer.model))
                columns.append(json.loads((Path(trainer.output_dir)/'task_1_memory_columns.json').read_text())['columns'])
                self.assertTrue(all(set(b) == {'0', '1'} for b in trainer.task_sketches.values()))
            self.assertEqual(before_fingerprints[0], before_fingerprints[1])
            self.assertEqual(before_fingerprints[0], non_memory_fingerprint(first.model))
            self.assertEqual(snapshots[0], snapshots[1])
            self.assertEqual(columns[0], columns[1])
            # Continue only Haze4K, without old training data or rebuilding old summaries.
            config['experiment']['output_dir'] = str(fixture.root/'reference')
            initial = fixture.root/'balanced/task_1_final.pth'
            reference = Guarded(config, fixture.manifest)
            reference.run(resume_path=initial)
            interrupted_cfg = copy.deepcopy(config)
            interrupted_cfg['experiment']['output_dir'] = str(fixture.root/'interrupted')
            interrupted = Guarded(interrupted_cfg, fixture.manifest)
            save = interrupted.save
            def disconnect(path, *args, **kwargs):
                save(path, *args, **kwargs)
                if Path(path).name == 'latest.pth': raise RuntimeError('disconnect')
            interrupted.save = disconnect
            with self.assertRaisesRegex(RuntimeError, 'disconnect'):
                interrupted.run(resume_path=initial)
            latest = fixture.root/'interrupted/latest.pth'
            state = torch.load(latest, weights_only=False)
            changed = copy.deepcopy(interrupted_cfg)
            changed['task_memory']['mode'] = 'mean_pca'
            with self.assertRaisesRegex(ValueError, '训练协议'):
                validate_resume(changed, state, state['split_sha256'])
            resumed = Guarded(interrupted_cfg, fixture.manifest)
            resumed.run(resume_path=latest)
            self.assertEqual(reference.history, resumed.history)
            self.assertEqual(bank_hash(reference.task_sketches), bank_hash(resumed.task_sketches))
            for key, value in reference.model.state_dict().items():
                self.assertTrue(torch.equal(value, resumed.model.state_dict()[key]), key)
            bad = copy.deepcopy(state)
            layer = next(iter(bad['task_sketches']))
            bad['task_sketches'][layer]['0']['factor'][0, 0] += 1
            corrupted = fixture.root/'corrupt.pth'
            torch.save(bad, corrupted)
            with self.assertRaisesRegex(ValueError, 'SHA256'):
                Guarded(interrupted_cfg, fixture.manifest).restore(corrupted)
            self.assertEqual(set(next(iter(resumed.task_sketches.values()))), {'0', '1', '2'})
        finally:
            fixture.tearDown()


if __name__ == '__main__': unittest.main()
