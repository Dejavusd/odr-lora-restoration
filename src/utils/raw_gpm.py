"""Re-estimate activation memories from a fixed OLD TRAIN calibration subset."""
import torch
from torch.utils.data import DataLoader, Subset

from src.models.odr_lora import ODRLoRAConv2d


@torch.no_grad()
def fit_raw_gpm_banks(model, dataset, indices, device, capacities=(64, 128), batch_size=8):
    """Use the same covariance and preserve the old span at both capacities.

    This is offline memory calibration, not replay or model optimization. The
    caller controls RNG for the dataset's random crops and augmentations.
    """
    layers = {name: layer for name, layer in model.named_modules()
              if isinstance(layer, ODRLoRAConv2d)}
    covariances, counts, hooks = {}, {}, []

    def make_hook(name):
        def hook(layer, args):
            H = layer.raw_input_matrix(args[0]).float()
            contribution = H @ H.T
            if name not in covariances:
                covariances[name] = contribution
                counts[name] = H.shape[1]
            else:
                covariances[name].add_(contribution)
                counts[name] += H.shape[1]
        return hook

    was_training = model.training
    model.eval()
    for name, layer in layers.items():
        hooks.append(layer.register_forward_pre_hook(make_hook(name)))
    try:
        loader = DataLoader(Subset(dataset, indices), batch_size=batch_size,
                            shuffle=False, num_workers=0)
        for step, batch in enumerate(loader, 1):
            model(batch['input'].to(device), -1)
            print(f'RAW GPM calibration batch {step}/{len(loader)}', flush=True)
    finally:
        for hook in hooks:
            hook.remove()
        model.train(was_training)
    banks = {cap: {'basis': {}, 'diagnostics': {}} for cap in capacities}
    for index, (name, layer) in enumerate(layers.items(), 1):
        C = covariances.pop(name).cpu()
        for cap in capacities:
            M, info = layer.dual_gpms['global'].extend_from_covariance(C, max_basis=cap)
            banks[cap]['basis'][name] = M.cpu()
            banks[cap]['diagnostics'][name] = {**info, 'columns': counts[name]}
        if index % 20 == 0:
            print(f'RAW GPM residual PCA {index}/{len(layers)} layers', flush=True)
    return banks


@torch.no_grad()
def install_gpm_bank(model, bank, capacity):
    layers = {name: layer for name, layer in model.named_modules()
              if isinstance(layer, ODRLoRAConv2d)}
    if layers.keys() != bank['basis'].keys():
        raise ValueError('Raw GPM bank and model layer names differ')
    for name, layer in layers.items():
        M = bank['basis'][name]
        if M.shape[0] != layer.d or M.shape[1] > min(capacity, layer.d):
            raise ValueError(f'Invalid raw GPM shape for {name}')
        gpm = layer.dual_gpms['global']
        gpm.max_basis = min(capacity, layer.d)
        gpm.basis = M.to(device=layer.weight.device, dtype=layer.weight.dtype)
        gpm.singular_values = None
