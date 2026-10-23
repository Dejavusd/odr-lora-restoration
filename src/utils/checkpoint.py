"""
Checkpoint 保存与加载工具。

包含：
    - save_checkpoint: 保存模型、优化器、epoch、task_idx 及可选元信息
    - load_checkpoint: 加载 checkpoint，返回 epoch 和 task_idx
    - load_model_weights: 只加载模型权重，用于评估
"""

import os
import re
import torch


def _task_count(model):
    for module in model.modules():
        if hasattr(module, 'task_A'):
            return len(module.task_A)
    return 0


def _prepare_dynamic_tasks(model, model_state, requested_count=None):
    """Create ParameterList entries before loading their checkpoint tensors."""
    if requested_count is None:
        indices = []
        for key in model_state:
            match = re.search(r'(?:^|\.)task_A\.(\d+)$', key)
            if match:
                indices.append(int(match.group(1)))
        requested_count = max(indices) + 1 if indices else 0
    while _task_count(model) < requested_count:
        model.add_task()


def _restore_runtime_state(model, state):
    current_idx = int(state.get('current_task_idx', -1))
    if hasattr(model, 'set_current_task'):
        model.set_current_task(current_idx)


# ============================================================
# 保存
# ============================================================

def save_checkpoint(model, optimizer, epoch, task_idx, path, extra=None):
    """
    保存 checkpoint。

    参数：
        model: torch.nn.Module
        optimizer: torch.optim.Optimizer 或 None
        epoch: int
        task_idx: int
        path: str，保存路径
        extra: dict，可选，额外元信息（如 cfg、freq_level、freq_bands）
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    state = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict() if optimizer is not None else None,
        'epoch': int(epoch),
        'task_idx': int(task_idx),
        'num_tasks': _task_count(model),
        'current_task_idx': next(
            (int(m.current_task_idx) for m in model.modules()
             if hasattr(m, 'current_task_idx')),
            -1,
        ),
    }

    if extra is not None and isinstance(extra, dict):
        # 避免覆盖核心字段
        for k, v in extra.items():
            if k not in state:
                state[k] = v

    try:
        torch.save(state, path)
    except Exception as e:
        print(f"保存 checkpoint 失败: {path}, 错误: {e}")


# ============================================================
# 加载
# ============================================================

def load_checkpoint(
    model, path, optimizer=None, map_location='cpu', return_state=False
):
    """
    加载 checkpoint。

    参数：
        model: torch.nn.Module
        path: str，checkpoint 路径
        optimizer: torch.optim.Optimizer 或 None
        map_location: str 或 torch.device

    返回：
        epoch: int
        task_idx: int
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"checkpoint 不存在: {path}")

    # 兼容 PyTorch 2.6+ 的 weights_only 默认值
    try:
        state = torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        # 旧版本 PyTorch 不支持 weights_only 参数
        state = torch.load(path, map_location=map_location)

    # 加载模型
    if 'model' in state:
        model_state = state['model']
    else:
        # 兼容直接保存 state_dict 的情况
        model_state = state

    _prepare_dynamic_tasks(model, model_state, state.get('num_tasks'))
    model.load_state_dict(model_state, strict=False)
    _restore_runtime_state(model, state)

    # 加载优化器
    if optimizer is not None and state.get('optimizer') is not None:
        try:
            optimizer.load_state_dict(state['optimizer'])
        except Exception as e:
            print(f"加载优化器状态失败: {e}")

    epoch = state.get('epoch', 0)
    task_idx = state.get('task_idx', -1)
    if return_state:
        return epoch, task_idx, state
    return epoch, task_idx


# ============================================================
# 只加载模型权重
# ============================================================

def load_model_weights(model, path, map_location='cpu'):
    """
    只加载模型权重，用于评估。

    参数：
        model: torch.nn.Module
        path: str，checkpoint 路径
        map_location: str 或 torch.device

    返回：
        state: dict，完整 checkpoint 字典（可能包含 epoch、task_idx 等）
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"checkpoint 不存在: {path}")

    try:
        state = torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        state = torch.load(path, map_location=map_location)

    if 'model' in state:
        model_state = state['model']
    else:
        model_state = state

    _prepare_dynamic_tasks(model, model_state, state.get('num_tasks'))
    model.load_state_dict(model_state, strict=False)
    _restore_runtime_state(model, state)

    return state


# ============================================================
# 导出
# ============================================================

__all__ = [
    "save_checkpoint",
    "load_checkpoint",
    "load_model_weights",
]
