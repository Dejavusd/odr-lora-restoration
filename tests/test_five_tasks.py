import copy,json
from pathlib import Path
import unittest
import numpy as np
from PIL import Image
import torch

from tests import test_clean_sequence as clean_tests
from src.data.five_task_protocol import prepare_extension,ManifestNoiseDataset,new_tasks,load_extension
from src.trainers.memory_retention_trainer import MemoryRetentionTrainer
from src.trainers.five_task_trainer import five_task_config,FiveTaskTrainer,patch_lpips
from src.utils.training_state import atomic_json


class FiveTaskTests(unittest.TestCase):
    setUp=clean_tests.CleanSequenceTests.setUp
    tearDown=clean_tests.CleanSequenceTests.tearDown
    config_at=clean_tests.CleanSequenceTests.config_at

    def extra_data(self):
        rng=np.random.default_rng(451)
        for folder,count in [('train',6),('test',2),('BSD68',1)]:
            d=self.root/'data/DIV2K+Flickr2K'/folder;d.mkdir(parents=True)
            for i in range(count):
                shape=(32,32) if folder=='BSD68' else (32,32,3)
                Image.fromarray(rng.integers(0,256,shape,dtype=np.uint8)).save(d/f'{i}.png')
        for split,groups in [('train',3),('test',1)]:
            for group in range(groups):
                for clip in range(2):
                    d=self.root/'data/GOPRO'/split/f'{split}{group}_01_{clip:02}'
                    (d/'blur').mkdir(parents=True);(d/'sharp').mkdir()
                    for i in range(2):
                        gt=rng.integers(20,230,(32,32,3),dtype=np.uint8)
                        Image.fromarray(gt).save(d/'sharp'/f'{i}.png')
                        Image.fromarray((gt*.8).astype(np.uint8)).save(d/'blur'/f'{i}.png')

    def test_append_only_groups_noise_and_subsets(self):
        self.extra_data();path=self.root/'five.json'
        m=prepare_extension(self.cfg['data']['split_manifest'],path,1)
        cfg=copy.deepcopy(self.cfg);cfg['data']['tasks']+=new_tasks(2);cfg['data']['split_manifest']=str(path)
        load_extension(cfg,self.manifest)
        for n in self.manifest['tasks']:self.assertEqual(m['tasks'][n],self.manifest['tasks'][n])
        g=m['tasks']['deblur']
        a={g['train_records'][i]['sequence_group'] for i in g['train_indices']}
        b={g['train_records'][i]['sequence_group'] for i in g['validation_indices']}
        self.assertFalse(a&b)
        rows=m['tasks']['denoise']['test_records'];task=new_tasks()[0]
        ds=ManifestNoiseDataset(rows,task,False)
        before=torch.get_rng_state();first=ds[len(rows)-1];second=ds[len(rows)-1]
        self.assertTrue(torch.equal(before,torch.get_rng_state()))
        self.assertTrue(torch.equal(first['input'],second['input']))
        self.assertTrue(torch.equal(first['input'][0],first['input'][1]))
        self.assertTrue(first['grayscale'])

    def test_patch_lpips_handles_short_edges_and_area_weights(self):
        shapes=[]
        def fn(a,b):
            shapes.append(a.shape[-2:]);return (a-b).abs().mean().reshape(1)
        value=patch_lpips(fn,torch.ones(1,3,65,67),torch.zeros(1,3,65,67),32)
        self.assertEqual(value,2.)
        self.assertTrue(all(min(s)>=32 for s in shapes))

    def test_five_task_extension_no_old_training_and_exact_resume(self):
        self.extra_data();extended=self.root/'five.json'
        prepare_extension(self.cfg['data']['split_manifest'],extended,1)
        cfg=self.config_at('balanced3');cfg['train']['epochs_per_task']=1
        cfg['memory_retention']=dict(merge_rule='balanced',old_fraction=.5)
        cfg['odr_lora']['gpm_update_rule']='bounded_balanced'
        source_trainer=MemoryRetentionTrainer(cfg,self.manifest);source_trainer.run(stop_after='dehaze')
        source=Path(source_trainer.output_dir)/'task_2_final.pth'
        config=five_task_config(source,extended,self.root/'five_a',epochs=2,batch_size=2)
        config['eval']['validation_interval']=1
        for task in config['data']['tasks'][3:]:task['patch_size']=16
        manifest=load_extension(config,self.manifest)
        class Guarded(FiveTaskTrainer):
            def loader(self,name,role):
                if role=='train' and name not in ('denoise','deblur'):raise AssertionError('Old training read')
                return super().loader(name,role)
        full=Guarded(config,manifest);full.run(stop_after='denoise',resume_path=source)
        interrupted_cfg=copy.deepcopy(config);interrupted_cfg['experiment']['output_dir']=str(self.root/'five_b')
        interrupted=Guarded(interrupted_cfg,manifest);save=interrupted.save
        def disconnect(path,*args,**kwargs):
            save(path,*args,**kwargs)
            if Path(path).name=='latest.pth':raise RuntimeError('five disconnect')
        interrupted.save=disconnect
        with self.assertRaisesRegex(RuntimeError,'five disconnect'):interrupted.run(stop_after='denoise',resume_path=source)
        resumed=Guarded(interrupted_cfg,manifest);resumed.run(stop_after='denoise',resume_path=self.root/'five_b/latest.pth')
        self.assertEqual(full.history,resumed.history)
        for n,v in full.model.state_dict().items():self.assertTrue(torch.equal(v,resumed.model.state_dict()[n]),n)
        # Separate stage-final continuation reaches GOPRO without old train data.
        final=Path(full.output_dir)/'task_3_final.pth'
        tail_cfg=copy.deepcopy(config);tail_cfg['experiment']['output_dir']=str(self.root/'five_tail')
        tail=Guarded(tail_cfg,manifest);tail.run(stop_after='deblur',resume_path=final)
        self.assertIn('4',tail.performance_matrix)


if __name__=='__main__':unittest.main()
