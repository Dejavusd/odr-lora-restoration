# balanced50 无回放五任务续训

任务顺序：LSRW → RainH → Haze4K → DIV2K+Flickr2K 去噪 → GOPRO 去模糊。
从 `outputs/memory_balanced50_b64/task_2_final.pth` 接续，不重训前三任务，
不启用回放、教师或旧任务训练损失。仍使用真实空间统计、GPM-32、有效基和私有秩16。
50% 指历史子空间池/当前任务残差的容量配额，不是每个历史任务各占50%，也不保证旧任务输出不变。
本次扩展用于检查五任务容量与遗忘，并不表示已解决三任务中的遗忘问题。

## 数据与评估协议

当前工作区已生成 `outputs/five_task_splits_v1/manifest.json`，不要覆盖。
原三任务记录、划分、评估方式原样保留；新任务数据不移动、不重命名。

| 新任务 | 训练 | 验证 | 测试 |
| --- | ---: | ---: | --- |
| 去噪 | 3106 | 344 | DIV2K 100 + BSD68 68 |
| GOPRO | 1893 | 210 | 1111 |

去噪3450张原训练图含重复GT（3391个唯一内容），按解码后内容分组划分。
训练在线生成独立 AWGN，sigma=25/255，不裁剪含噪输入。
验证/测试噪声由 seed42 和干净文件SHA确定，不随任务阶段或遍历顺序变化。
BSD68灰度图使用单通道噪声再复制为RGB，输出取RGB均值后评估。
去噪测试分别报告DIV2K/BSD68，也报告168张按图像数加权的总体结果。

GOPRO采用blur/sharp配对，TRAIN内部按录像名前两段分组，训练/验证不共享来源组；
验证组为GOPR0378_13、GOPR0385_11。官方TRAIN/TEST本身共享部分录像前缀，
因此不能宣称三种划分完全视频/场景独立。本轮保持官方测试集不变，已检查无完全相同GT跨TRAIN/TEST。

新任务采用512窗口、64重叠的分块推理；LPIPS采用不重叠512块、按有效面积加权，
不足32像素的短边复制填充。该LPIPS不是完整图像VGG评分的严格等价替代；
不可直接与旧去噪/GOPRO入口的全图协议作同协议比较。前三任务仍沿用原全图评估。
每5轮验证已见任务，但只按当前新任务验证PSNR选权重；恢复最佳权重后更新GPM并合并。

新工作区缺少扩展清单时，仅运行一次（需要现有三任务父清单与原始数据）：

```bash
python src/scripts/prepare_five_task_splits.py
```

## 正式训练：先完成去噪

```bash
conda activate odr-lora
cd /media/zhappy-701/新加卷/odr-lora/odr-lora-restoration-main
export PYTHONNOUSERSITE=1
CUDA_VISIBLE_DEVICES=1 python -u src/scripts/train_five_tasks.py \
  --source outputs/memory_balanced50_b64/task_2_final.pth \
  --manifest outputs/five_task_splits_v1/manifest.json \
  --epochs 20 --batch-size 64 \
  --stop-after denoise \
  --output-dir outputs/balanced50_five_tasks
```

去噪后检查 `task_3_final.pth` 对应的四任务测试结果，尤其是前三任务PSNR变化。
然后接续GOPRO：

```bash
CUDA_VISIBLE_DEVICES=1 python -u src/scripts/train_five_tasks.py \
  --source outputs/memory_balanced50_b64/task_2_final.pth \
  --manifest outputs/five_task_splits_v1/manifest.json \
  --resume outputs/balanced50_five_tasks/task_3_final.pth \
  --epochs 20 --batch-size 64 \
  --output-dir outputs/balanced50_five_tasks
```

如要自动连续完成两个新任务，首次命令删除 `--stop-after denoise` 即可。
中断时将 `--resume` 改为 `outputs/balanced50_five_tasks/latest.pth`，其他协议参数保持不变。
去噪阶段若仍需停在去噪，续训时也传 `--stop-after denoise`。
恢复为完整epoch边界恢复，不是batch级恢复；源三任务断点、父清单、扩展清单均须保留。
不要把 `--source` 改为五任务断点，五任务断点只传给 `--resume`。

## 已完成验证

59项unittest通过，包含固定噪声、灰度处理、录像分组与精确断点恢复。
真实数据冒烟已走完两个新增任务及GPM更新/合并，但只用每任务一次小批更新、少量评估图，
不能用作正式质量结果。正式20轮训练未启动。
冒烟输出在 `outputs/five_task_smoke_20261105`，不可用作正式续训源。

RTX4090单卡实测两个新增任务各一次真实batch64（3×128×128）优化更新：
去噪峰值已分配显存14.16 GiB、去模糊14.18 GiB，缓存预留峰值均16.12 GiB。
容量检查只校准/更新一次GPM，不是正式训练，未导出正式模型，三任务源断点SHA保持不变。
显存结果不表示可以安全与其他大显存进程共享同一张卡。
