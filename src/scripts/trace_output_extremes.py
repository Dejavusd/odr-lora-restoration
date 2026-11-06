"""Trace full-image and center-crop activations without changing checkpoints."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch

from src.scripts.audit_output_stability import read_model
from src.data.clean_protocol import load_manifest, make_loader, sha256_file
from src.utils.training_state import atomic_json


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--checkpoints', nargs='+', required=True)
    args = parser.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    results = []
    for path in args.checkpoints:
        model, state = read_model(path, 'cuda')
        cfg = state['run_protocol']
        manifest = load_manifest(cfg)
        for task, index in [('lowlight', 97), ('rainh', 123)]:
            image = make_loader(cfg, manifest, task, 'validation').dataset[index]['input'][None].cuda()
            h, w = image.shape[-2:]
            for mode, x in [('full', image), ('center128', image[..., (h-128)//2:(h+128)//2,
                                                                           (w-128)//2:(w+128)//2])]:
                trace = []
                def hook(name):
                    def capture(module, inputs, output):
                        if isinstance(output, torch.Tensor):
                            trace.append(dict(name=name, type=type(module).__name__,
                                shape=list(output.shape), minimum=float(output.min()),
                                maximum=float(output.max()), rms=float(output.square().mean().sqrt())))
                    return capture
                handles = [m.register_forward_hook(hook(n)) for n,m in model.named_modules() if n]
                try:
                    pred = model(x, -1)
                finally:
                    for handle in handles:
                        handle.remove()
                row = dict(checkpoint=path, sha256=sha256_file(path), task=task, index=index,
                    mode=mode, minimum=float(pred.min()), maximum=float(pred.max()), trace=trace)
                results.append(row)
                first = next((t for t in trace if max(abs(t['minimum']),abs(t['maximum']))>1000), None)
                print(path,task,mode,row['minimum'],row['maximum'],'first>1000',first,flush=True)
        del model, state
        torch.cuda.empty_cache()
    atomic_json(out/'trace.json', results)


if __name__ == '__main__':
    main()
