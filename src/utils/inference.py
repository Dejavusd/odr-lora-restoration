"""Optional overlapping tiles for memory-bounded restoration evaluation."""
import torch


@torch.no_grad()
def predict_image(model, x, task_idx, tile_size=0, overlap=64):
    if tile_size <= 0 or max(x.shape[-2:]) <= tile_size:
        return model(x, task_idx)
    if not 0 <= overlap < tile_size:
        raise ValueError('tile_overlap must satisfy 0 <= overlap < tile_size')
    height, width = x.shape[-2:]
    stride = tile_size - overlap

    def starts(length):
        return sorted(set(list(range(0, max(length - tile_size, 0), stride))
                          + [max(length - tile_size, 0)]))

    output = torch.zeros_like(x)
    weights = torch.zeros_like(x[:, :1])
    for top in starts(height):
        for left in starts(width):
            patch = x[..., top:top + tile_size, left:left + tile_size]
            prediction = model(patch, task_idx)
            output[..., top:top + tile_size, left:left + tile_size] += prediction
            weights[..., top:top + tile_size, left:left + tile_size] += 1
    return output / weights
