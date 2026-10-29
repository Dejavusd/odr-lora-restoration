"""Real GPU/4-worker epoch-resume audit; 2 updates only, never a formal model."""
import argparse
import gc
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch
import yaml

from src.data.clean_protocol import load_manifest
from src.scripts.train_clean_sequence import configure_method
from src.trainers.clean_sequence_trainer import CleanSequenceTrainer
from src.utils.training_state import atomic_json, seed_all


class AuditTrainer(CleanSequenceTrainer):
    disconnect = False

    def save(self, path, idx, epoch, *args, **kwargs):
        super().save(path, idx, epoch, *args, **kwargs)
        if Path(path).name == 'latest.pth':
            if epoch == 2:
                # Capture actual last optimization state BEFORE best selection,
                # memory calibration or merging can obscure optimizer errors.
                self.training_tail = torch.load(path, map_location='cpu', weights_only=False)
            if epoch == 1 and self.disconnect:
                raise RuntimeError('audit simulated disconnect')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/clean_lsrw_rainh_haze4k.yaml')
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    cfg = configure_method(yaml.safe_load(Path(args.config).read_text()), 'effective_r16')
    cfg['smoke_test'] = True
    cfg['train'].update(epochs_per_task=2, max_steps_per_epoch=1, calibration_batches=1, gpm_batches=1)
    cfg['eval'].update(validation_interval=1, max_images=1)
    cfg['data']['num_workers'] = 4
    for task in cfg['data']['tasks']:
        task['batch_size'] = 2
    manifest = load_manifest(cfg)
    torch.set_num_threads(4)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.cuda.set_per_process_memory_fraction(.5)
    cfg['experiment']['output_dir'] = str(output/'reference')
    seed_all(cfg['experiment']['seed'])
    reference = AuditTrainer(cfg, manifest)
    reference.run(stop_after='lowlight')
    expected = reference.training_tail
    del reference
    gc.collect()
    torch.cuda.empty_cache()
    cfg['experiment']['output_dir'] = str(output/'interrupted')
    seed_all(cfg['experiment']['seed'])
    interrupted = AuditTrainer(cfg, manifest)
    interrupted.disconnect = True
    try:
        interrupted.run(stop_after='lowlight')
    except RuntimeError as error:
        if str(error) != 'audit simulated disconnect':
            raise
    else:
        raise RuntimeError('Audit did not interrupt at epoch 1')
    del interrupted
    gc.collect()
    torch.cuda.empty_cache()
    resumed = AuditTrainer(cfg, manifest)
    resumed.run(stop_after='lowlight', resume_path=output/'interrupted/latest.pth')
    actual = resumed.training_tail
    model_error = max(float((v-actual['model'][k]).abs().max())
                      for k,v in expected['model'].items() if v.dtype != torch.bool and v.numel())
    optimizer_error = max(float((v-actual['optimizer']['state'][k][name]).abs().max())
                          for k,state in expected['optimizer']['state'].items()
                          for name,v in state.items() if torch.is_tensor(v))
    history_equal = expected['history'] == actual['history']
    scheduler_equal = expected['scheduler'] == actual['scheduler']
    result = dict(passed=model_error == 0 and optimizer_error == 0 and history_equal and scheduler_equal,
        workers=4, updates=2, model_max_abs_error=model_error, optimizer_max_abs_error=optimizer_error,
        history_equal=history_equal, scheduler_equal=scheduler_equal, smoke_test=True,
        note='Real LSRW/NAFNet/VGG; interruption after epoch 1. Last optimization state compared before '
             'best restore and GPM update. NOT a quality benchmark; cannot resume into formal training.')
    atomic_json(output/'verification.json', result)
    print(result, flush=True)
    if not result['passed']:
        raise RuntimeError('GPU/worker resume consistency check failed')


if __name__ == '__main__':
    main()
