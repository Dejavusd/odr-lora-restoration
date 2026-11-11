# 无回放稳定骨干对照（stable_v1）

逐任务平衡摘要与等权PCA的无回放匹配对照，参见 [TASK_BALANCED_MEMORY.md](TASK_BALANCED_MEMORY.md)。

在同一LSRW权重上单独测试跳跃分支感知保护（B），参见 [SKIP_BRANCH.md](SKIP_BRANCH.md)。

目标是减少骨干对LoRA更新的非线性放大，而不是声称已消除遗忘。
此前的日志显示新任务后旧任务下降，并伴随极端原始输出；这支持进行结构对照，
但尚不足以证明具体层是遗忘的根因。

## 改动

| 位置 | legacy | stable_v1 |
| --- | --- | --- |
| 块内归一化 | GroupNorm(1,C)，统计跨通道与空间 | 每像素通道归一化，带可学习仿射 |
| 乘法门控 | a×b | a×tanh(b) |
| 通道注意力 | 无界线性增益 | 2×sigmoid增益，范围(0,2) |
| 残差系数 | beta、gamma直接使用 | tanh(beta)、tanh(gamma)，范围(-1,1) |

仍是同一共享模型和输入/输出头，无回放、无教师、无任务ID路由、无独立任务网络。
新任务仍只训练当前LoRA系数，基座和归一化/门控参数在任务0后冻结，任务末合并方式不变。
保护秩与私有秩不增加。门控限幅不等于全网Lipschitz约束，也不保证原始输出有界；
卷积、可学习归一化仿射和注意力输入统计仍会影响敏感性。
这是组合结构消融，如有效，还需分别移除各改动、增加种子复验，而不能单独归功于某一组件。

必须从LSRW重新训练。旧检查点不兼容新的归一化参数形状与前向语义；
不要通过strict=False强行加载。旧代码路径默认为legacy，既有实验不被切换。
精确恢复会检查architecture和分任务预算，禁止跨结构续训。

## 匹配协议

配置：`configs/stable_backbone_lsrw_rainh_haze4k.yaml`。
固定原始干净配对与划分、batch64、patch128、有效基/私有秩16、空间输入统计、
GPM-32 bounded_svd、相同种子/损失/优化器；LSRW400轮，RainH/Haze4K各20轮。
这轮不使用balanced50配额，避免与结构因素混杂。
旧balanced50结果仅作为研究背景，不是该新结构的严格匹配对照。
任务0也必须分别训练；不能让legacy组共用stable组的LSRW权重。

```bash
conda activate odr-lora
cd /media/zhappy-701/新加卷/odr-lora/odr-lora-restoration-main
export PYTHONNOUSERSITE=1

# 改进组，先停在LSRW检查基础性能
CUDA_VISIBLE_DEVICES=1 python -u src/scripts/train_clean_sequence.py \
  --config configs/stable_backbone_lsrw_rainh_haze4k.yaml \
  --method effective_r16 --architecture stable_v1 \
  --output-dir outputs/stable_v1_b64 --stop-after lowlight

# 同一改进组接续RainH，先观察旧任务是否保持
CUDA_VISIBLE_DEVICES=1 python -u src/scripts/train_clean_sequence.py \
  --config configs/stable_backbone_lsrw_rainh_haze4k.yaml \
  --method effective_r16 --architecture stable_v1 \
  --resume outputs/stable_v1_b64/task_0_final.pth \
  --output-dir outputs/stable_v1_b64 --stop-after rainh

# 匹配对照组：只有architecture不同，也必须从头训练
CUDA_VISIBLE_DEVICES=1 python -u src/scripts/train_clean_sequence.py \
  --config configs/stable_backbone_lsrw_rainh_haze4k.yaml \
  --method effective_r16 --architecture legacy \
  --output-dir outputs/stable_control_b64 --stop-after lowlight
```

对照组接续RainH时使用legacy参数及其自身`task_0_final.pth`，不要共用改进组断点。
之后分别使用各自`task_1_final.pth`并指定`--stop-after dehaze`，输出目录保持各自不变。
如中断，用各自`latest.pth`恢复，其余协议参数不变。
无需在运行时改400/20预算，`epochs_by_task`已记录在协议中，断点恢复不会重置调度器。

首先比较任务0PSNR，避免把基础性能下降误判为遗忘改善；然后比较各组从其自身边界
到下一任务的旧任务PSNR下降、原始输出异常、以及新任务PSNR。
当前阶段测试集仅用于预定任务边界报告，验证最优仍只按当前任务验证PSNR选择。
未通过三任务对照前，不把当前五任务入口直接用于新结构；那个入口仍要求旧balanced50源。

## 验证范围

新增测试覆盖像素归一化的空间局部性、门控幅值、legacy默认行为不变、
stable精确epoch恢复、增量任务合并与拒绝跨结构恢复。
冒烟测试只验证连通性，不可用作正式质量证据。没有启动正式400轮训练。

64项自动测试通过；真实LSRW→RainH→Haze4K冒烟已完成训练、校准、GPM更新和合并。
RTX4090实际batch64、patch128、真实VGG损失的任务0优化更新峰值已分配显存约20.11 GiB，
预留约20.23 GiB；请使用有足够空闲显存的单卡，不要与其他训练共享。
同次容量检查接续RainH的batch64 LoRA优化更新峰值已分配约16.99 GiB；
任务0到任务1的GPM更新/合并/校准均通过，容量检查未导出正式模型。
