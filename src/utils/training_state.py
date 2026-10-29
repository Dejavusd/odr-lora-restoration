"""Fail-loud atomic checkpoints and complete epoch-boundary RNG state."""
import json
import os
from pathlib import Path
import random
import tempfile

import numpy as np
import torch


def atomic_write(path, writer):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + '.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as handle:
            writer(handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_json(path, value):
    atomic_write(path, lambda handle: handle.write(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False).encode('utf-8')))


def atomic_checkpoint(path, state):
    atomic_write(path, lambda handle: torch.save(state, handle))


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def capture_rng():
    return dict(python=random.getstate(), numpy=np.random.get_state(), cpu=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else [])


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['cpu'].cpu())
    if state['cuda']:
        if len(state['cuda']) != torch.cuda.device_count():
            raise ValueError('恢复时可见 GPU 数量与保存时不同；请保持 CUDA_VISIBLE_DEVICES 一致')
        torch.cuda.set_rng_state_all([s.cpu() for s in state['cuda']])
