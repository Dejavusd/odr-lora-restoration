# 旧任务输出稳定性验证（2026-11-04）

结论：当前 effective-r16 / frequency GPM-32 在 Haze4K 阶段不能保持旧任务输出稳定。
这次使用历史 batch16 正式断点做配对审计；没有进行 batch64 的完整重训。

## 方法与范围

- 起点：`outputs/clean_effective_r16/task_1_final.pth`。
- 终点：`outputs/clean_effective_r16/task_2_final.pth`，对应 Haze4K 验证最优 epoch260。
- 全部旧任务训练留出验证集：LSRW 564 张、RainH 126 张，原始完整分辨率。
- 严格 FP32，关闭 TF32，相同输入逐张比较；PSNR/SSIM 继续使用已有 clamp 协议。
- 原始输出额外记录 MAE/RMSE/均值偏移/最大绝对变化，不通过 clamp 隐藏漂移。
- 两个源断点的 SHA256 审计前后一致；原始数据与源模型没有修改。
- 逐图结果：`outputs/output_stability_20261104_full/stability.json`。

## 全量验证结果

| 验证集 | Haze4K 前 PSNR | Haze4K 后 PSNR | PSNR 变化 | SSIM 前→后 | clamp 后输出 MAE | 下降超过1dB的图像 |
|---|---:|---:|---:|---|---:|---:|
| LSRW |18.8511|11.7644|-7.0867|0.5874→0.3283|0.212845|483/564|
| RainH |21.9316|12.6750|-9.2566|0.6973→0.3859|0.210673|123/126|

输出 MAE 使用 [0,1] 单位，约相当于 8-bit 的 54.3 / 53.7 灰度级。
原始输出 MAE 分别为 0.303811 / 0.249155；平均有符号漂移分别为 -0.215118 / +0.194721。
有符号漂移只描述全局统计，不能概括每幅图像所有区域的变化。

## 子空间保护与实际特征

每个任务前8张验证图像、每层最多128个真实卷积输入 patch 的抽样：

| 旧任务 | 各层 GPM 能量覆盖率的算术均值 | 局部增量响应/原权重响应 RMS 比的层中位数 | 最大层比值 |
|---|---:|---:|---:|
| LSRW |54.13%|0.2020|1.3514|
| RainH |78.46%|0.1049|0.5062|

这里使用进入 Haze4K 前的真实激活和 GPM，增量为 Haze4K 选中模型合并后的权重差。
局部响应不包括上游激活随后改变的影响。覆盖率只代表指定抽样，不是全数据、全网络的能量比例。
已有完整更新投影泄漏约1e-7，但投影外的真实输入分量仍有明显响应。
这说明“对保存的 GPM 正交”没有转化为“旧任务输出保持不变”。

额外核对：普通冻结参数没有变化；合并权重差与选中 A@B 的最大差为5.96e-8。
原记录中 GPM 更新前向误差为0，合并前后像素误差约1e-6。
这些检查不支持将这次退化归因于冻结失效或重复合并。

## 原始输出极端值

对每任务输出变化最大的3张图像补查，结果见 `extreme_values.json`。
LSRW 验证索引97：原始输出范围在 Haze4K 前为 [-4900.77,7757.81]，之后为 [-25407.15,29168.84]。
RainH 验证索引123：原始输出范围由 [-0.346,1.152] 变为 [-2622.30,2033.89]。
另一个 LSRW 样本在 Haze4K 之前已有极端值，之后恢复正常范围；不能将全部异常归因于新任务。
这些是选定样本的范围，不是所有图像的范围统计；未定位到具体首个异常层。
clamp 后的 PSNR/SSIM 无法充分反映这种原始输出异常。

下一步优先定位极端值最先出现的层，并对旧任务训练样本加入输出一致性约束做独立对照。
回放/蒸馏需要明确标注旧数据使用量；当前审计没有实现或宣称这类改进已解决问题。

## batch64 修改与实际容量检查

`configs/clean_lsrw_rainh_haze4k.yaml` 三任务均改为 batch64，显存额度改为0.9。
原0.5额度在真实 Haze4K+VGG 单步前向时 OOM；0.9额度完成前向、反向和 AdamW 更新。
峰值 allocated 为14.15 GiB，reserved 为16.11 GiB，loss=0.194502。
使用128裁剪、单个校准 batch、similarity=0；这是容量检查，不是质量或正式训练证据。
结果：`outputs/batch64_capacity_20261104_quota90/capacity.json`。
这不保证任意其他图像尺寸或全分辨率验证的峰值显存。

新增 `--allow-batch-size-change` 允许从已完成任务显式分支，放宽 batch size/显存额度检查，
必须使用新输出目录。任务中途仍禁止更改，其他方法、学习率、数据等配置仍严格核对。
新分支中途恢复时使用其 batch64 配置与 `latest.pth`，不再加该分支选项。

仅需重跑 batch64 Haze4K 时的命令（本次没有启动400轮训练）：

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONNOUSERSITE=1 python -u src/scripts/train_clean_sequence.py \
  --config configs/clean_lsrw_rainh_haze4k.yaml \
  --resume outputs/clean_effective_r16/task_1_final.pth \
  --output-dir outputs/clean_effective_r16_b64_haze4k \
  --method effective_r16 --allow-batch-size-change
```

改变 batch size 不能视为已修复遗忘；固定400 epoch时优化步数也会改变。
新增与相关的14项测试通过，覆盖批量分支边界、其他协议拒绝、精确恢复及真实卷积响应抽样计算。
