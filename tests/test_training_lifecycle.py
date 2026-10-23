import os
import sys
import tempfile

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.backbone import NAFNet
from src.models.odr_lora import ODRLoRAConv2d
from src.utils.checkpoint import load_checkpoint, save_checkpoint


def _layer():
    return ODRLoRAConv2d(
        3, 4, kernel_size=3, share_rank=2, private_rank=2,
        rank_min=2, rank_max=2, spatial_blocks=(1, 1), freq_level=1,
        spatial_sample_ratio=0.5,
    )


def test_merge_is_equivalent_and_idempotent():
    torch.manual_seed(0)
    layer = _layer()
    layer.add_task()
    layer.set_current_task(0)
    with torch.no_grad():
        layer.task_A[0].normal_()
        layer.task_B[0].normal_()

    x = torch.randn(2, 3, 12, 12)
    before = layer(x, 0)
    layer.merge_task(0)
    after = layer(x, -1)
    assert torch.allclose(before, after, atol=1e-5)

    weight = layer.weight.detach().clone()
    layer.merge_task(0)
    assert torch.equal(weight, layer.weight), "same adapter was merged twice"


def test_checkpoint_restores_dynamic_tasks_and_gpm():
    torch.manual_seed(1)
    layer = _layer()
    layer.add_task()
    layer.set_current_task(0)
    x = torch.randn(2, 3, 16, 16)
    layer.calibrate(x, 0)
    layer.update_gpm(x, 0)

    with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(directory, 'state.pth')
        save_checkpoint(layer, None, epoch=1, task_idx=0, path=path)
        restored = _layer()
        _, task_idx = load_checkpoint(restored, path)

    assert task_idx == 0
    assert len(restored.task_A) == 1
    assert bool(restored.share_initialized)
    assert torch.allclose(restored.task_B[0], layer.task_B[0])
    for key in layer.dual_gpms:
        expected = layer.dual_gpms[key].basis
        actual = restored.dual_gpms[key].basis
        assert expected is not None and actual is not None
        assert torch.allclose(actual, expected)


def test_only_first_task_trains_shared_backbone():
    model = NAFNet(
        base_channels=4, num_blocks=(1, 1), share_rank=2,
        private_rank=2, rank_min=2, rank_max=2,
        spatial_blocks=(1, 1), freq_level=1,
    )
    model.add_task()
    model.freeze_base_and_old(0)
    assert model.downs[0].weight.requires_grad
    assert model.input_conv.weight.requires_grad
    assert not model.input_conv.task_B[0].requires_grad

    model.add_task()
    model.freeze_base_and_old(1)
    trainable = [name for name, p in model.named_parameters() if p.requires_grad]
    assert trainable
    assert all('.task_A.1' in name for name in trainable), trainable


if __name__ == '__main__':
    test_merge_is_equivalent_and_idempotent()
    test_checkpoint_restores_dynamic_tasks_and_gpm()
    test_only_first_task_trains_shared_backbone()
    print("所有训练闭环测试通过。")
