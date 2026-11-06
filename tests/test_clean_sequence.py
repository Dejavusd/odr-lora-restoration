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

    def test_batch_size_branch_requires_explicit_flag_and_task_boundary(self):
        sha = sha256_file(self.cfg['data']['split_manifest'])
        saved = dict(clean_protocol_version=1, split_sha256=sha, run_protocol=run_protocol(self.cfg),
                     task_idx=1, in_progress=False, smoke_test=False, gpm_updated=True)
        changed = copy.deepcopy(self.cfg)
        for task in changed['data']['tasks']:
            task['batch_size'] = 64
        changed['experiment']['memory_fraction'] = .9
        with self.assertRaisesRegex(ValueError, '训练协议'):
            validate_resume(changed, saved, sha)
        validate_resume(changed, saved, sha, allow_batch_size_change=True)
        saved['in_progress'] = True
        with self.assertRaisesRegex(ValueError, '边界'):
            validate_resume(changed, saved, sha, allow_batch_size_change=True)
        saved['in_progress'] = False
        changed['train']['lr'] *= .5
        with self.assertRaisesRegex(ValueError, '训练协议'):
            validate_resume(changed, saved, sha, allow_batch_size_change=True)

    def test_stability_replay_train_partition_and_exact_resume(self):
        from src.trainers.stability_trainer import stability_config, StabilityTrainer
        cfg = self.config_at('source_stability')
        cfg['train']['epochs_per_task'] = 1
        source_trainer = CleanSequenceTrainer(cfg, self.manifest)
        source_trainer.run(stop_after='rainh')
        source = Path(cfg['experiment']['output_dir'])/'task_1_final.pth'
        cfg_a = stability_config(source, self.root/'stability_a', epochs=2,
                                 batch_size=2, memory_per_task=2)
        full = StabilityTrainer(cfg_a, self.manifest)
        full.run(stop_after='dehaze', resume_path=source)
        records = json.loads((self.root/'stability_a/replay_manifest.json').read_text())
        for task in ('lowlight','rainh'):
            indices = records['tasks'][task]['train_indices']
            self.assertTrue(set(indices) <= set(self.manifest['tasks'][task]['train_indices']))
            self.assertFalse(set(indices) & set(self.manifest['tasks'][task]['validation_indices']))
        for p in full.teacher.parameters():
            self.assertFalse(p.requires_grad)
            self.assertIsNone(p.grad)
        cfg_b = copy.deepcopy(cfg_a)
        cfg_b['experiment']['output_dir'] = str(self.root/'stability_b')
        interrupted = StabilityTrainer(cfg_b, self.manifest)
        save = interrupted.save
        def disconnect(path, *args, **kwargs):
            save(path, *args, **kwargs)
            if Path(path).name == 'latest.pth':
                raise RuntimeError('stability disconnect')
        interrupted.save = disconnect
        with self.assertRaisesRegex(RuntimeError, 'stability disconnect'):
            interrupted.run(stop_after='dehaze', resume_path=source)
        resumed = StabilityTrainer(cfg_b, self.manifest)
        resumed.run(stop_after='dehaze', resume_path=self.root/'stability_b/latest.pth')
        self.assertEqual(full.history, resumed.history)
        for key, value in full.model.state_dict().items():
            self.assertTrue(torch.equal(value, resumed.model.state_dict()[key]), key)

    def test_bounded_lora_no_old_training_loader_and_exact_resume(self):
        from src.trainers.bounded_lora_trainer import bounded_config, BoundedLoRATrainer
        cfg = self.config_at('source_bound')
        cfg['train']['epochs_per_task'] = 1
        source_trainer = CleanSequenceTrainer(cfg,self.manifest)
        source_trainer.run(stop_after='rainh')
        source = Path(cfg['experiment']['output_dir'])/'task_1_final.pth'
        cfg_a = bounded_config(source,self.root/'bound_a',max_relative_update=1e-6,epochs=2,batch_size=2)
        cfg_a['eval']['validation_interval'] = 1
        class NoReplayTrainer(BoundedLoRATrainer):
            def loader(self, task, role):
                if role == 'train' and task != 'dehaze':
                    raise AssertionError('Old training data accessed')
                return super().loader(task,role)
        full = NoReplayTrainer(cfg_a,self.manifest)
        original_split = full.split_sha
        full.split_sha = 'changed'
        with self.assertRaisesRegex(ValueError,'SHA256'):
            full.restore(source)
        full.split_sha = original_split
        full.run(stop_after='dehaze',resume_path=source)
        self.assertFalse(hasattr(full,'teacher'))
        self.assertFalse(hasattr(full,'memories'))
        for row in full.history:
            self.assertLessEqual(row['update_bound']['max_after'],1e-6*(1+1e-5))
            self.assertEqual(row['validation']['psnr'],row['validation']['tasks']['dehaze']['psnr'])
        cfg_b = copy.deepcopy(cfg_a)
        cfg_b['experiment']['output_dir'] = str(self.root/'bound_b')
        interrupted = NoReplayTrainer(cfg_b,self.manifest)
        save = interrupted.save
        def disconnect(path,*args,**kwargs):
            save(path,*args,**kwargs)
            if Path(path).name == 'latest.pth':
                raise RuntimeError('bound disconnect')
        interrupted.save = disconnect
        with self.assertRaisesRegex(RuntimeError,'bound disconnect'):
            interrupted.run(stop_after='dehaze',resume_path=source)
        latest = self.root/'bound_b/latest.pth'
        saved = torch.load(latest,map_location='cpu',weights_only=False)
        changed = copy.deepcopy(cfg_b)
        changed['update_bound']['max_relative_update'] = .1
        with self.assertRaisesRegex(ValueError,'训练协议'):
            validate_resume(changed,saved,sha256_file(self.cfg['data']['split_manifest']))
        resumed = NoReplayTrainer(cfg_b,self.manifest)
        resumed.run(stop_after='dehaze',resume_path=latest)
        self.assertEqual(full.history,resumed.history)
        for key,value in full.model.state_dict().items():
            self.assertTrue(torch.equal(value,resumed.model.state_dict()[key]),key)

    def test_preparer_rejects_train_test_gt_overlap(self):
        row = self.manifest['tasks']['lowlight']['train_records'][0]
        target = Path(self.manifest['tasks']['lowlight']['test_records'][0]['gt'])
        target.write_bytes(Path(row['gt']).read_bytes())
        cfg = copy.deepcopy(self.cfg)
        cfg['data']['split_manifest'] = str(self.root/'other_split.json')
        with self.assertRaisesRegex(ValueError, '训练/测试存在'):
            prepare_splits(cfg, workers=1)

    def test_memory_source_common_weights_counts_and_no_replay_resume(self):
        from src.trainers.memory_source_trainer import memory_source_config,MemorySourceTrainer,non_memory_fingerprint
        cfg=self.config_at('source_memory')
        cfg['train']['epochs_per_task']=1
        original=CleanSequenceTrainer(cfg,self.manifest)
        original.run(stop_after='lowlight')
        source=Path(cfg['experiment']['output_dir'])/'task_0_final.pth'
        fingerprints=[];budgets=[]
        class NoReplayTrainer(MemorySourceTrainer):
            bootstrap=False
            def bootstrap_memory(self):
                self.bootstrap=True
                try:return super().bootstrap_memory()
                finally:self.bootstrap=False
            def loader(self,task,role):
                if role=='train' and task=='lowlight' and not self.bootstrap:
                    raise AssertionError('Old training images accessed after task0 boundary')
                return super().loader(task,role)
        for kind in ('raw','frequency'):
            config=memory_source_config(source,self.root/f'memory_{kind}',kind,epochs=2,batch_size=2)
            config['eval']['validation_interval']=1
            trainer=NoReplayTrainer(config,self.manifest)
            trainer.restore(source)
            fingerprints.append(non_memory_fingerprint(trainer.model))
            budgets.append(json.loads((Path(trainer.output_dir)/'task_0_memory_bootstrap.json').read_text())['columns'])
        self.assertEqual(fingerprints[0],fingerprints[1])
        self.assertEqual(budgets[0],budgets[1])
        self.assertEqual(fingerprints[0],non_memory_fingerprint(original.model))
        # Resume bootstrap checkpoint without reopening the LSRW train loader.
        reference_cfg=memory_source_config(source,self.root/'memory_ref','raw',epochs=2,batch_size=2)
        reference_cfg['eval']['validation_interval']=1
        reference=NoReplayTrainer(reference_cfg,self.manifest)
        reference.run(stop_after='dehaze',resume_path=self.root/'memory_raw/task_0_final.pth')
        cfg_b=copy.deepcopy(reference_cfg);cfg_b['experiment']['output_dir']=str(self.root/'memory_interrupt')
        interrupted=NoReplayTrainer(cfg_b,self.manifest)
        save=interrupted.save
        def disconnect(path,*args,**kwargs):
            save(path,*args,**kwargs)
            if Path(path).name=='latest.pth':raise RuntimeError('memory disconnect')
        interrupted.save=disconnect
        with self.assertRaisesRegex(RuntimeError,'memory disconnect'):
            interrupted.run(stop_after='dehaze',resume_path=self.root/'memory_raw/task_0_final.pth')
        saved=torch.load(self.root/'memory_interrupt/latest.pth',map_location='cpu',weights_only=False)
        changed=copy.deepcopy(cfg_b);changed['odr_lora']['gpm_statistics']='frequency'
        with self.assertRaisesRegex(ValueError,'训练协议'):
            validate_resume(changed,saved,sha256_file(self.cfg['data']['split_manifest']))
        resumed=NoReplayTrainer(cfg_b,self.manifest)
        resumed.run(stop_after='dehaze',resume_path=self.root/'memory_interrupt/latest.pth')
        self.assertEqual(reference.history,resumed.history)
        for key,value in reference.model.state_dict().items():
            self.assertTrue(torch.equal(value,resumed.model.state_dict()[key]),key)

    def test_retention_branch_same_weights_old_span_and_exact_resume(self):
        from src.trainers.memory_source_trainer import memory_source_config,MemorySourceTrainer
        from src.trainers.memory_retention_trainer import retention_config,MemoryRetentionTrainer
        cfg=self.config_at('retention_origin');cfg['train']['epochs_per_task']=1
        first=CleanSequenceTrainer(cfg,self.manifest);first.run(stop_after='lowlight')
        source=Path(first.output_dir)/'task_0_final.pth'
        run=self.root/'retention_raw'
        memory_cfg=memory_source_config(source,run,epochs=1,batch_size=2)
        raw=MemorySourceTrainer(memory_cfg,self.manifest);raw.run(stop_after='rainh',resume_path=source)
        class GuardedTrainer(MemoryRetentionTrainer):
            boundary=False
            def rebuild_rainh_boundary(self,reference):
                self.boundary=True
                try:return super().rebuild_rainh_boundary(reference)
                finally:self.boundary=False
            def loader(self,task,role):
                if role=='train' and task!='dehaze' and not (task=='rainh' and self.boundary):
                    raise AssertionError('Old training images accessed outside boundary')
                return super().loader(task,role)
        baseline_cfg=retention_config(run,self.root/'retention_control','compressed',epochs=2,batch_size=2)
        baseline=GuardedTrainer(baseline_cfg,self.manifest);baseline.restore(run/'task_1_best.pth')
        for n,m in raw.model.named_modules():
            if hasattr(m,'dual_gpms'):
                target=dict(baseline.model.named_modules())[n].dual_gpms['global'].basis
                torch.testing.assert_close(m.dual_gpms['global'].basis,target,rtol=1e-5,atol=1e-5)
        candidate_cfg=retention_config(run,self.root/'retention_a','preserve',epochs=2,batch_size=2)
        candidate_cfg['eval']['validation_interval']=1
        full=GuardedTrainer(candidate_cfg,self.manifest)
        full.run(stop_after='dehaze',resume_path=run/'task_1_best.pth')
        report=json.loads((Path(full.output_dir)/'rainh_boundary_rebuild.json').read_text())
        self.assertTrue(report['same_non_memory_tensors'])
        for layer in report['layers'].values():
            self.assertLess(layer['old_span_residual'],1e-5)
            self.assertLessEqual(layer['rank_after'],layer['capacity'])
        cfg_b=copy.deepcopy(candidate_cfg);cfg_b['experiment']['output_dir']=str(self.root/'retention_b')
        interrupted=GuardedTrainer(cfg_b,self.manifest);save=interrupted.save
        def disconnect(path,*args,**kwargs):
            save(path,*args,**kwargs)
            if Path(path).name=='latest.pth' and args[0]==2:raise RuntimeError('retention disconnect')
        interrupted.save=disconnect
        with self.assertRaisesRegex(RuntimeError,'retention disconnect'):
            interrupted.run(stop_after='dehaze',resume_path=run/'task_1_best.pth')
        latest=self.root/'retention_b/latest.pth'
        saved=torch.load(latest,map_location='cpu',weights_only=False)
        changed=copy.deepcopy(cfg_b);changed['odr_lora']['gpm_update_rule']='bounded_svd'
        with self.assertRaisesRegex(ValueError,'训练协议'):
            validate_resume(changed,saved,sha256_file(self.cfg['data']['split_manifest']))
        resumed=GuardedTrainer(cfg_b,self.manifest);resumed.run(stop_after='dehaze',resume_path=latest)
        self.assertEqual(full.history,resumed.history)
        for n,v in full.model.state_dict().items():self.assertTrue(torch.equal(v,resumed.model.state_dict()[n]),n)

    def test_balanced_quota_same_covariance_and_exact_resume(self):
        from src.trainers.memory_source_trainer import memory_source_config,MemorySourceTrainer
        from src.trainers.memory_retention_trainer import retention_config,MemoryRetentionTrainer
        cfg=self.config_at('balanced_origin');cfg['train']['epochs_per_task']=1
        first=CleanSequenceTrainer(cfg,self.manifest);first.run(stop_after='lowlight')
        source=Path(first.output_dir)/'task_0_final.pth';run=self.root/'balanced_raw'
        raw=MemorySourceTrainer(memory_source_config(source,run,epochs=1,batch_size=2),self.manifest)
        raw.run(stop_after='rainh',resume_path=source)
        a=retention_config(run,self.root/'balanced_a','balanced',epochs=2,batch_size=2,old_fraction=.5)
        a['eval']['validation_interval']=1
        full=MemoryRetentionTrainer(a,self.manifest);full.run(resume_path=run/'task_1_best.pth')
        b=retention_config(run,self.root/'balanced_old100','balanced',epochs=2,batch_size=2,old_fraction=1.)
        baseline=MemoryRetentionTrainer(b,self.manifest);baseline.restore(run/'task_1_best.pth')
        report_a=json.loads((Path(full.output_dir)/'task_1_balanced_memory.json').read_text())['layers']
        report_b=json.loads((Path(baseline.output_dir)/'task_1_balanced_memory.json').read_text())['layers']
        for n,r in report_a.items():
            self.assertEqual(r['covariance_sha256'],report_b[n]['covariance_sha256'])
            self.assertLessEqual(r['rank'],r['cap'])
            self.assertLessEqual(r['retained_old_rank'],r['old_quota'])
        interrupted_cfg=copy.deepcopy(a);interrupted_cfg['experiment']['output_dir']=str(self.root/'balanced_interrupt')
        interrupted=MemoryRetentionTrainer(interrupted_cfg,self.manifest);save=interrupted.save
        def disconnect(path,*args,**kwargs):
            save(path,*args,**kwargs)
            if Path(path).name=='latest.pth' and args[0]==2:raise RuntimeError('balanced disconnect')
        interrupted.save=disconnect
        with self.assertRaisesRegex(RuntimeError,'balanced disconnect'):
            interrupted.run(resume_path=run/'task_1_best.pth')
        latest=self.root/'balanced_interrupt/latest.pth';saved=torch.load(latest,map_location='cpu',weights_only=False)
        changed=copy.deepcopy(interrupted_cfg);changed['memory_retention']['old_fraction']=.75
        with self.assertRaisesRegex(ValueError,'训练协议'):
            validate_resume(changed,saved,sha256_file(self.cfg['data']['split_manifest']))
        resumed=MemoryRetentionTrainer(interrupted_cfg,self.manifest);resumed.run(resume_path=latest)
        self.assertEqual(full.history,resumed.history)
        for n,v in full.model.state_dict().items():self.assertTrue(torch.equal(v,resumed.model.state_dict()[n]),n)


if __name__ == '__main__':
    unittest.main()
