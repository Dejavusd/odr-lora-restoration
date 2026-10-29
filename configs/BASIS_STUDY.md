# 投影后有效基与新任务私有秩对照

后续分组留出、多增量种子和 1000 步验证见 [BASIS_VALIDATION.md](BASIS_VALIDATION.md)。
后续审计发现 LSRW 历史配对存在补零编号碰撞，已修复；以下保留原始短程结果，
不视为数据修复后从头训练的基准。各组固定同一个旧 checkpoint。

2026-10-29 的单种子筛查已完成，每组 200 步、batch=8。完整测试集结果：

| 方案 | LSRW PSNR | RainH PSNR | 平均 PSNR | 平均 SSIM ↑ | 平均 LPIPS ↓ |
|---|---:|---:|---:|---:|---:|
| current_basis | 19.0650 | 17.0319 | 18.0485 | 0.4608 | 0.4578 |
| effective_r8 | 18.9741 | 17.4793 | 18.2267 | 0.4758 | 0.4487 |
| effective_r16 | 18.8889 | 17.9081 | 18.3985 | 0.4886 | 0.4408 |

候选配置为 configs/lsrw_rainh_effective_r16.yaml，输出为独立目录。
它保留原训练配置的 400 epoch / cosine 设置，**并非本次 200 步实验的复现命令**，
也没有经过长程或多种子验证；未自动启动正式训练。
本轮仅检验固定旧 GPM 下的 LSRW→RainH，未验证任务末记忆压缩及更多任务。
完整数据与曲线见 outputs/basis_study_20261029/REPORT.md。

在 odr-lora 环境、项目根目录执行：

```bash
CUDA_VISIBLE_DEVICES=1 python src/scripts/run_forgetting_probe.py \
  --study basis \
  --checkpoint outputs/lsrw/task_0_final.pth \
  --config configs/lsrw_rainh.yaml \
  --output-dir outputs/basis_study_new \
  --rounds 5 --steps-per-round 40 --batch-size 8
```

默认顺序运行三组，均固定旧 GPM-32、完整更新保护、网络结构和 L1/LPIPS 损失。
不重建 GPM、不使用旧训练图。相同种子保证 RainH 校准/训练采样、裁剪与增强
一致；每组 200 个优化器 step，不是 200 个完整 epoch。

| 组别 | 基构造 | 新任务私有基础秩 |
|---|---|---:|
| current_basis | 原软投影/SVD 后，将整个 B 严格投影到旧记忆正交补 | 8 |
| effective_r8 | 先投影，再提取有效单位共享/私有方向 | 8 |
| effective_r16 | 同上，仅扩展 RainH 新分支 | 16 |

新构造对共享基先做严格投影和 SVD，剔除小于 max(1e-6, 最大奇异值×1e-5)
的方向，再提取单位正交共享基。将新任务特征投影到旧记忆及有效共享基的
共同正交补，再提取私有基。最后再次投影以消除浮点残差。无效行保持零，
不会把近零残差归一化或用随机方向填满。

private rank 按层尺度增长，上限仍为 model.rank_max=32。GPM 容量由独立的
odr_lora.gpm_max_basis=32 控制。第三组分配更多 A 参数，不是等参数比较。
第一、二组分配参数数相同，但方向、有效秩及范数会改变。

## 正式代码中的选项

```yaml
model:
  private_rank_base: 8          # 保持已有 task_0 checkpoint 的结构
  new_task_private_rank_base: 16 # 仅对随后新建的任务分支生效
odr_lora:
  full_update_protection: true
  basis_construction: projected_svd
  basis_rank_tolerance: 1.0e-5
```

未指定 basis_construction 时仍使用 legacy。加载动态 checkpoint 时会按保存
的每个任务分支形状恢复不同秩，而非扩展或裁剪旧分支；旧任务参数不被修改。
恢复优化器时同步更新参数引用。新分支参数数、有效行数、有效秩和 Gram 误差
记录在每轮 diagnostics 中。

## 输出与解读

输出目录必须尚不存在，中断后加 --resume-experiment 保持其余参数一致。
REPORT.md / summary.json 的终点指标使用完整 LSRW 50 张、RainH 100 张；
曲线是固定的 20/30 张子集。旧测试图只用于诊断，不用于优化、早停或挑选模型。
各组 arm_config.yaml 保存实际配置，curve.json 保存逐层诊断。

probe_*.pth 是短程诊断模型，未做任务末 GPM 更新，不作为正式下一任务起点。
结论限于本次单种子、固定预算筛查，应在独立验证集、多种子和长程训练中确认。
