# 下一步：stable_v1 逐任务记忆接续去噪

配对 Haze4K seed7/42/2026 平均 PSNR 收益 +0.148888 dB，新增两 seed 平均 +0.162632 dB。
这是固定前两任务的条件复验，不代表完整五任务多 seed。本轮不再叠加新模块。

## 目的与边界

- 固定原 seed42 配对继续第四任务，不按三个 seed 的测试分数择优选源。
- balanced 从 `outputs/task_balanced_b64/task_2_final.pth` 接续；
  mean_pca 从 `outputs/task_mean_pca_b64/task_2_final.pth` 接续。
- 两组的 Haze4K 权重和摘要已经不同。本轮是**各自序列继续的匹配对照**，
  不是从共同物理权重出发、只改第四阶段选择器的消融。
- 保留 stable_v1、尺度秩、有效基、GPM-32 和各组三个旧任务摘要，不重选起始 GPM，
  不重读旧 TRAIN 图像构建摘要，不启用回放、教师或额外旧任务损失。
- source 中的 RainH/Haze4K epoch 曲线和阶段0/1/2测试矩阵原样继承；
  source 本身没有的 LSRW400轮曲线不伪造继承，原 stable_v1 目录仍保留。
- 完整配置包含五任务，但默认停在去噪。先看四任务遗忘，不自动启动 GOPRO。

旧 `train_five_tasks.py` 仍只支持早期 balanced50 历史池方案，不要把当前断点交给它。
新入口是 `src/scripts/train_task_memory_extension.py`。

## 数据、训练与评估

复用 `outputs/five_task_splits_v1/manifest.json`，并核对原父清单 SHA 和旧三任务条目。
不移动原始数据，不重新生成划分。旧 TRAIN 文件只做清单元数据 stat 核验，不解码其图像。

去噪训练3106、验证344；测试 DIV2K 100张 + BSD68 68张。在线生成 AWGN25/255，含噪输入不裁剪；
验证/测试噪声固定于 seed42 和干净文件 SHA。BSD68灰度噪声单通道生成再复制RGB，预测取RGB均值。
两组统一 batch64、patch128、严格FP32、20轮；沿用优化器、损失和6批摘要预算，新增摘要不改写旧摘要。
每5轮验证已见任务，但只按当前去噪验证 PSNR 选模，不按旧任务测试结果选模。
任务末加载验证最佳权重、更新摘要/GPM并合并，再评估四任务。

原三任务仍是原有全图指标；去噪使用512窗口/64重叠推理、有效面积加权512块LPIPS。
分块LPIPS不等价于全图VGG评分，不与其他入口混作同协议比较。
总体去噪结果按168张图平均，同时分别报告 DIV2K/BSD68。

## 正式训练（两条命令顺序执行）

```bash
conda activate odr-lora
cd /media/zhappy-701/新加卷/odr-lora/odr-lora-restoration-main
export PYTHONNOUSERSITE=1

CUDA_VISIBLE_DEVICES=1 python -u src/scripts/train_task_memory_extension.py \
  --source outputs/task_balanced_b64/task_2_final.pth \
  --manifest outputs/five_task_splits_v1/manifest.json \
  --epochs 20 --batch-size 64 --stop-after denoise \
  --output-dir outputs/task_memory_denoise_b64/balanced

CUDA_VISIBLE_DEVICES=1 python -u src/scripts/train_task_memory_extension.py \
  --source outputs/task_mean_pca_b64/task_2_final.pth \
  --manifest outputs/five_task_splits_v1/manifest.json \
  --epochs 20 --batch-size 64 --stop-after denoise \
  --output-dir outputs/task_memory_denoise_b64/mean_pca
```

请勿同时运行两条命令争用GPU1；建议在tmux中依次执行。启动前确认GPU1有足够空闲显存。
本次仅做冒烟与测试，未代为启动正式20轮训练；不要使用smoke断点作为正式训练源。

## 中断恢复

源参数一直指向原 Haze4K 最终断点，`--resume` 才指向扩展入口的断点：

```bash
CUDA_VISIBLE_DEVICES=1 python -u src/scripts/train_task_memory_extension.py \
  --source outputs/task_balanced_b64/task_2_final.pth \
  --manifest outputs/five_task_splits_v1/manifest.json \
  --resume outputs/task_memory_denoise_b64/balanced/latest.pth \
  --epochs 20 --batch-size 64 --stop-after denoise \
  --output-dir outputs/task_memory_denoise_b64/balanced
```

PCA组替换为自身的source和输出目录；不要交叉加载。恢复优化器、scheduler、随机状态和CPU摘要，
按完整epoch边界恢复；未完成epoch重跑。任务末摘要采集若中断，从最后完整epoch重做边界。
mode/source/seed/batch/预算/清单/实现代码变化均拒绝中途恢复；正式运行中不要改代码或切换分支。
若还没有latest但已有失败目录，保留诊断并选新输出目录，不覆盖。

## 完成后汇总

```bash
python src/scripts/summarize_task_memory_extension.py \
  --balanced outputs/task_memory_denoise_b64/balanced \
  --control outputs/task_memory_denoise_b64/mean_pca \
  --output-dir outputs/task_memory_denoise_b64/summary
```

汇总拒绝未完成、smoke、协议/代码/种子不匹配或缺少测试子集的结果。
表中同时报告各任务PSNR、去噪子集、四任务宏平均、旧三任务均值，以及相对各组自身Haze4K的旧任务下降。
`old_mean_drop` 越小越好；不能只看四任务平均被新去噪任务高分抬升。
遗忘指标还会比较各任务历史最佳阶段，不等同于仅本次的平均下降。

新产物包括 `haze_boundary_import.json`、扩展协议的 `task_2_final.pth`、
`task_3_final.pth`、每任务摘要/保护诊断、历史曲线、测试矩阵与 `latest.pth`。
`merged.pth` 只用于推理，没有完整记忆，不能用于后续训练。

先审查去噪的匹配结果，再决定是否继续GOPRO。
已支持同目录增加 `--resume .../task_3_final.pth --stop-after deblur`，其余source和预算不变，
但这不是要求现在就运行第五任务。

## 本次验证（2026-11-11）

- 两组正式Haze4K源到去噪的真实数据/GPU1冒烟通过：每组仅1次小批更新、每评估集1张。
  这只验证连通性、VGG损失、分块推理、摘要/GPM更新与模型合并，不能用作质量比较。
- 单元测试覆盖两组源与GPM校验、禁止旧TRAIN和bootstrap、历史摘要/曲线/测试矩阵保留、
  逐张量精确epoch恢复、任务末摘要采集断线重做、可选GOPRO继续，以及配对汇总/不匹配拒绝。
- 独立GPU小模型epoch恢复测试通过，恢复前后逐张量一致；GPM与权重同设备，历史摘要保持CPU。
- 正式配置为batch64/patch128/20轮。冒烟实际使用batch2，不伪称已经验证此入口batch64峰值显存；
  此前同stable/task-memory骨干已测allocated约17.53 GiB、reserved约19.67 GiB，仅作容量参考。
- 两个正式Haze4K源的SHA在检查前后不变。未启动两组正式20轮训练。
