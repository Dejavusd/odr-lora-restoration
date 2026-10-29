# 真实卷积输入 GPM 的容量对照

数据审计补充：LSRW 旧配对曾因补零编号碰撞将 5600 张展开成 10500 对，现已修复。
本实验的 raw 记忆只用输入、不用误配 GT，但固定索引的旧训练采样仍受旧列表重复/顺序影响；
修复后重跑不保证相同的 32 张输入。应按原 manifest 路径解释历史结果。
旧起点也需在干净配对下重训后复核，详见 [BASIS_VALIDATION.md](BASIS_VALIDATION.md)。

2026-10-29 的单种子短程筛查（每组 200 步）结果：

| 方案 | LSRW PSNR | RainH PSNR | 两任务平均 PSNR | 旧测试图抽样覆盖率 |
|---|---:|---:|---:|---:|
| current32 | 19.06 | 17.03 | 18.05 | 72.0% |
| raw64 | 19.08 | 15.92 | 17.50 | 83.3% |
| raw128 | 19.08 | 15.22 | 17.15 | 88.5% |

在本轮固定训练预算下，增加容量提升了覆盖估计，却降低了新任务及整体指标。
因此没有把 raw64/raw128 设为默认推荐。后续应先检查投影后有效基的构造、
归一化与私有秩，再用独立验证集、多种子和更长训练验证。
完整结果保存在 outputs/gpm_study_20261029/REPORT.md；此结论不能推断
更长训练后的最终排名，也不代表所有数据集都应使用相同容量。

在项目根目录、odr-lora 环境中运行：

```bash
CUDA_VISIBLE_DEVICES=1 python src/scripts/run_forgetting_probe.py \
  --study gpm \
  --checkpoint outputs/lsrw/task_0_final.pth \
  --config configs/lsrw_rainh.yaml \
  --output-dir outputs/gpm_study_new \
  --rounds 5 --steps-per-round 40 --batch-size 8 \
  --gpm-calibration-samples 32
```

输出目录必须尚不存在。中断后增加 `--resume-experiment`，保持其余参数一致。
默认顺序运行以下三组，均启用完整更新保护：

| 组别 | 保护空间来源 | 每层容量上限 |
|---|---|---:|
| current32 | task_0_final.pth 中现有的频带统计记忆 | 32 |
| raw64 | 保留旧记忆，补充真实卷积输入的残差 PCA 方向 | 64 |
| raw128 | 同一份原始输入协方差，保留相同旧方向，再补充残差方向 | 128 |

旧记忆从同一个 LSRW checkpoint 读取。仅使用固定索引的 32 张 LSRW **训练图**
进行离线记忆校准（seed=777，128 随机裁剪）。此步骤不更新模型权重，不使用
旧测试图或测试 GT，也不把旧图混入 RainH 优化器。manifest 记录索引与路径。
这是用旧训练数据重估已有记忆的实验，不应宣称整个流程完全不访问旧数据。

每层从真实卷积输入展开 patch，均匀抽取每批最多 512 列，累计未中心化的
H H^T。对旧子空间之外的协方差做 PCA，并按累计原始输入能量阈值 0.99 添加
方向；64/128 是上限，达到阈值后不会强行填满。两种容量使用相同协方差，
子空间嵌套，不通过截断旧记忆给新方向腾位置。

网络结构、LoRA 秩、损失权重、学习率及训练步数固定。每轮相同随机种子保证
新任务采样与增强相同。校准不消耗新任务训练的随机序列。两组测试集分别
用于评估，完整终点报告 LSRW 50 张、RainH 100 张；每轮曲线为固定 20/30 张。
覆盖诊断使用独立于拟合样本的旧测试图中心裁剪，只是逐层抽样覆盖估计。

输出包括 REPORT.md、curves.png、summary.json、逐组 curve.json 和 coverage_*.json、
raw_bank_64/128.pth 与 raw_fit_64/128.json。诊断模型未做任务末 GPM 更新，
不作为正式增量训练起点。

## 普通训练可用的选项

```yaml
odr_lora:
  full_update_protection: true
  gpm_statistics: raw
  gpm_max_basis: 64
  gpm_sample_columns: 512
```

GPM 容量已经独立于 model.rank_max（LoRA 秩预算）。未设置时继续使用旧行为。
普通训练选择 raw 时，任务末从真实输入更新 GPM，并保留已有方向。
仅在配置里写 raw/64 不会追溯重建旧 checkpoint 的记忆；本实验脚本才会
使用上述固定旧训练子集做离线补充。

保护空间覆盖越高，新任务自由度可能越低，所以必须同时比较新旧任务指标。
current32→raw64 还改变了统计方法和数据访问，不能单独归因于容量；
raw64→raw128 才更直接反映上限扩展。此实验仍是单种子短程筛查，不代替
长期训练、多种子与独立验证集上的模型选择。
