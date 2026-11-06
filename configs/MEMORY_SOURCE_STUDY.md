# 无回放 GPM 统计来源单变量对照

入口：`src/scripts/train_memory_source.py`。
两组从**同一份干净 LSRW task_0_final 权重**分支，先重建任务0记忆，再依次训练 RainH→Haze4K。
LSRW网络权重不重训；任务0记忆重建不更新任何权重。
该步骤模拟从共同首任务训练终点、进入第二任务前的记忆构造阶段分支，
不是在已经训练完RainH后使用旧LSRW图像修补记忆。

## 唯一实验因素

| 项目 | frequency 对照 | raw 空间统计组 |
|---|---|---|
| GPM输入矩阵 | 分块DWT子带patch | 真实卷积输入patch |
| 记忆容量上限 |每层32|每层32|
| 合并规则 |DualGPM.update 有损SVD合并|相同|
| 每批每层列预算 |相同，最多512列|相同，最多512列|
| LoRA B校准来源 |已有频域校准|相同，未修改|
| 完整更新硬正交 |开启|开启|
| LoRA新任务私有基础秩 |继承共同起点，当前为16|相同|
| 更新范数约束 / 输出范围损失 |关闭|关闭|
| 回放 / 教师 / 蒸馏 |无|无|

为了深层小特征图也严格匹配列数，两组使用
`min(512, 可用频域列数, 可用原始卷积列数)`，再均匀抽列。
raw分支计算频域矩阵只用于确定共同列预算，记忆拟合矩阵全部来自真实patch，未混入频域列。
frequency输入仍包含原有子带能量选列步骤；频域与空间域特征的统计差异属于本实验因素。
实际保留秩由相同能量阈值和容量上限决定，允许因统计不同而不同。

采用新显式选项 `gpm_update_rule: bounded_svd`，避免旧raw路径的update_preserving与frequency
路径合并算法不同。未设置时继续使用原有legacy路径；已有普通配置不自动切换。

## 训练数据与协议

共同起点：`outputs/clean_lsrw/task_0_final.pth`，不是旧RainH或Haze4K断点。
首次启动时仅清空GPM，固定种子20261104，用LSRW实际TRAIN分区的6个batch构造任务0记忆。
两组使用相同的输入采样和增强，记录前后非记忆张量指纹，要求前向误差为0。
该阶段不创建优化器，不进行参数训练，不读取验证/测试图像拟合记忆。

此后训练/校准/GPM更新只读取当前任务的TRAIN数据，不再次打开旧任务训练loader。
旧任务验证图像仅用于前向诊断，选模只看当前任务验证PSNR。
batch64、patch128、lr2e-4、AdamW、cosine、L1+0.1 LPIPS，其他设置继承共同LSRW起点。
默认RainH和Haze4K各20轮：RainH每轮18次更新，Haze4K每轮43次更新。
每5轮及末轮完整验证，任务结束测试已见任务；严格FP32。
这是短程单种子筛查，不应直接与历史400轮或其他选模规则的结果做方法排名。
保留旧数据的离线核对基没有用于本次训练。

## 命令

在仓库根目录执行：

```bash
conda activate odr-lora
cd /media/zhappy-701/新加卷/odr-lora/odr-lora-restoration-main
export PYTHONNOUSERSITE=1
```

空间统计组：

```bash
CUDA_VISIBLE_DEVICES=1 python -u src/scripts/train_memory_source.py \
  --source outputs/clean_lsrw/task_0_final.pth \
  --statistics raw --epochs 20 --batch-size 64 \
  --output-dir outputs/memory_raw32_b64
```

频域匹配对照，依次运行即可：

```bash
CUDA_VISIBLE_DEVICES=1 python -u src/scripts/train_memory_source.py \
  --source outputs/clean_lsrw/task_0_final.pth \
  --statistics frequency --epochs 20 --batch-size 64 \
  --output-dir outputs/memory_frequency32_b64
```

默认自动执行RainH→Haze4K，不重新训练LSRW。若先检查RainH，在命令末尾加 `--stop-after rainh`。
随后保持其他参数一致，加 `--resume 对应目录/task_1_final.pth`，去掉stop-after即可接Haze4K。
两组目前均应使用GPU1：实施时GPU0有其他作业。运行前自行检查最新显存占用。
默认进程显存额度0.9；正式20轮训练未由实施验证自动启动。

## 断点恢复

空间组断线后：

```bash
CUDA_VISIBLE_DEVICES=1 python -u src/scripts/train_memory_source.py \
  --source outputs/clean_lsrw/task_0_final.pth \
  --statistics raw --epochs 20 --batch-size 64 \
  --resume outputs/memory_raw32_b64/latest.pth \
  --output-dir outputs/memory_raw32_b64
```

每个完整epoch保存模型、优化器、scheduler、随机状态及完整协议；未完成epoch重跑。
初始化任务0记忆完成时也保存此实验的task_0_final和latest，可直接恢复进入RainH，无需再次读取LSRW训练图。
首个任务0记忆构造期间若中断，尚未有latest，则保留失败目录并在新的输出目录重启；不会覆盖原目录。
恢复时不可更换统计来源、轮数、batch、容量、源断点或数据划分。

## 检查结果

```bash
tail -f outputs/memory_raw32_b64/trainer.log
```

- `task_0_memory_bootstrap.json`：共同权重指纹、前向误差、每层实际采样列数。
- `task_1/2_memory_columns.json`：任务末记忆更新采样列数。
- `history.json`：当前与旧任务PSNR/SSIM/LPIPS、原始输出最大值和越界比例；validation.psnr仅为当前任务。
- `task_1/2_protection.json`：完整更新正交泄漏。
- `task_1/2_gpm_transition.json`：有损合并丢弃旧方向的诊断。
- `continual_metrics.json`：各阶段标准测试指标、平均性能、BWT和遗忘。

两组都仍会受到32维容量和有损合并限制。出现压缩警告不等于实验代码失败。
优先比较相同轮次的旧任务保持与新任务学习，再比较各组选中模型；不能只看某一个旧任务。
本次没有同时修改B的校准统计，若空间记忆有效，再将“空间B校准”作为独立因素检验。

## 已完成验证

49项unittest和50个独立测试函数通过（环境未安装pytest，未为此修改环境）。
新测试检查同一更新函数、相同列数、共同权重不变、后续不访问旧训练loader、断点恢复逐张量一致及统计切换拒绝恢复。
两组真实NAFNet/VGG的任务0记忆→RainH→Haze4K两步冒烟均通过，
起点非记忆指纹一致、记忆构造前向误差0，三个阶段实际列预算一致。
冒烟为每任务两步及每验证/测试集一张图，不能作为质量证据或正式续训起点。
产物：`outputs/memory_source_raw_smoke_20261104`、`outputs/memory_source_freq_smoke_20261104`。
另有batch64真实单步容量与列预算检查，结果在 `outputs/memory_source_b64_capacity_20261104/verification.json`。
