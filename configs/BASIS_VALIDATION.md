# 有效基/私有秩的进一步验证

入口：`src/scripts/validate_effective_basis.py`。
本轮输出：`outputs/basis_validation_20261029/`。

## 协议

- 固定同一个 `outputs/lsrw/task_0_final.pth`，比较 `current_basis` 与 `effective_r16`。
- 二者均完整更新保护，GPM-32/frequency 不变；后者改用投影后有效基，并仅增大新任务私有秩。
- 增量随机种子为 7、42、123；每组 1000 步、batch 8、128 裁剪、固定 LR 2e-4、L1 + 0.1 LPIPS。
- RainH 训练集有 1254 张输入，但只有 209 个不同的真实图像，每个真实图像有 6 张雨图。
  按解码后的 GT 内容 SHA256 分组，固定留出 21 个 GT（126 张），剩下 1128 张用于相似度、校准和优化。
  校准不能读取留出的图像。训练/验证分组无交集，RainH 原测试集也没有完全相同的训练 GT。
- 每 200 步计算完整分组验证集 PSNR，仅据此选择最优 checkpoint（同分取较早）。
  另记录固定 10 张 LSRW 测试图的保留曲线，但它不参与选模。
- 完整 LSRW 50 张、RainH 100 张测试图只在末步及验证最优 checkpoint 评估。
  保持既有 RGB PSNR/SSIM、LPIPS 协议，不使用额外亮度缩放或测试时自适应。
- 每次运行保存后实际恢复模型、AdamW 与 RNG，重放一步更新，对比 loss、权重和优化器矩。
  主训练随即恢复到检查前，不增加任何一次优化器更新。
- 断点以每 200 步的完整段为单位。每段预声明独立数据/增强种子，主进程加载数据，避免隐含 worker 游标。
  重跑同一命令会继续未完成的段并跳过完成的 seed。
- 检查旧参数、基座、共享基和 GPM 逐位不变；保存逐层有效秩、参数量和完整更新泄漏。
- 在三张预定 129×131 奇数尺寸裁剪上比较合并前后 max/mean 像素误差和 PSNR 差。
  合并代数检查显式禁用 cuDNN/matmul TF32，保持 max<1e-3、mean<1e-4、PSNR差<0.01 的原阈值。
  第二个基线种子的默认 TF32 路径在 mean=1.0003e-4 时触发边界失败；严格 FP32 下
  同图 max<5e-7、mean<7e-8。保留默认精度失败日志与单独诊断，不把它隐藏为“训练错误”。
  effective-r16 的 seed=123 默认路径也超出阈值（mean=1.77e-4、PSNR差=0.0151 dB），
  同样保存原始失败并对完全相同的 checkpoint 做严格 FP32 复核。
  主训练与完整测试指标的后端设置不变；检查结束恢复原后端设置。

## 运行

在仓库根目录执行。输出目录必须是新的，不能覆盖此前实验。

```bash
conda activate odr-lora
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
python src/scripts/validate_effective_basis.py \
  --output-dir outputs/basis_validation_20261029 --prepare
```

若两张卡均空闲，可在两个终端分别执行；每个进程显存上限默认为可见卡的 32%。

```bash
CUDA_VISIBLE_DEVICES=0 python src/scripts/validate_effective_basis.py \
  --output-dir outputs/basis_validation_20261029 --arms current_basis
```

```bash
CUDA_VISIBLE_DEVICES=1 python src/scripts/validate_effective_basis.py \
  --output-dir outputs/basis_validation_20261029 --arms effective_r16
```

也可只用一张卡，省略 `--arms`，两种方法会依次运行。两组均完成后汇总：

```bash
python src/scripts/validate_effective_basis.py \
  --output-dir outputs/basis_validation_20261029 --report
```

所有运行完成后，统一复查两种方法、三个种子的 final/验证最优共 12 份导出：

```bash
CUDA_VISIBLE_DEVICES=0 python src/scripts/check_validation_exports.py \
  --output-dir outputs/basis_validation_20261029
```

输出 `strict_fp32_export_checks.json`；默认 TF32 边界失败单独保存在
`default_backend_merge_failure_seed42.json`、
`default_backend_merge_failure_effective_r16_seed123.json`，不修改原阈值。

`REPORT.md`、`summary.json`、`curves.png` 是汇总；`manifest.json` 记录源 checkpoint 哈希、
配置、数据 SHA256 和实际划分；各组/各 seed 目录保存曲线、checkpoint、逐图测试指标、
断点重放检查和冻结/合并检查。标准输出可重定向到日志。

固定示例使用首个预声明种子（7），每个测试集等间隔三张，不据结果挑图；
完成该 seed 后可以导出对比图（用空闲卡运行）：

```bash
CUDA_VISIBLE_DEVICES=0 python src/scripts/export_validation_examples.py \
  --output-dir outputs/basis_validation_20261029 --seed 7
```

对应 `fixed_examples_seed_7.png/json` 和 `VISUAL_REVIEW.md`。

## 本轮新增发现：LSRW 文件名碰撞

LSRW `train/input`、`train/gt` 各 5600 张。`00011.jpg` 与 `11.jpg` 是不同图像，
但旧配对函数去掉补零后把它们当成同一个 ID，再对输入/GT 做笛卡尔积，产生 10500 对，
其中 4900 对并非同名对应。还有跨尺寸 GT 被数据集静默 resize 的风险。

现在 `pair_images` 优先完整文件名，其次唯一文件主名，最后只允许无歧义的数字 ID。
仍支持 haze/input + gt 目录、图像扩展名不同、多个退化输入对应一个 GT；歧义、缺配、
未使用 GT 均报错，不再按排序猜测。原始数据没有改名、删除或覆盖。
实测 LSRW 5600/50、RainH 1254/100、Haze4K 3000/1000 对均可加载。

无法仅从已有权重确认当时使用的具体数据代码。历史 task-0 checkpoint 可能受到这一问题影响；
本轮保留旧起点以做公平的算法条件比较，并没有声称历史训练已修正。
后续正式基准必须从修复后的配对重新训练任务0，另用训练集划分验证集后再做 RainH 与 Haze4K。
不要在旧 `outputs/lsrw/` 上覆盖重训，也不要把旧 checkpoint 当作已干净重训的结果。

## 解释边界

3 个增量种子共用一个旧模型；并非从头多种子训练。1000 步是原筛查预算的 5 倍，
并非 400 轮正式训练。effective-r16 多参数，不是等参数消融。
按完全相同 GT 分组可避免明显泄漏，但不保证相近场景完全隔离。
测试集此前已被用于开发，结果应视为额外复现证据，不能宣称独立未触碰测试集确认。

本轮刻意不更新任务末 GPM。`latest.pth`/`best_validation.pth` 是诊断 checkpoint，
不直接接续 Haze4K；也没有自动替换正式配置或上传 GitHub。
