# 2026-11-11 实验产物清理记录

仅清理已结束的 smoke / capacity 临时检查点，不清理正式质量实验、数据、源断点和划分。

操作：使用系统回收站，可恢复。目标 24 个诊断目录中的 121 个权重文件，共 12.2903 GiB。
移入回收站不会立即释放所在磁盘的空间；未清空回收站，也未永久删除。

保留全部诊断 JSON、CSV、日志；保留 pre_model 汇报、clean/stable 正式源、所有正式对照结果和三组 seed。
特别保留 task_balanced_b64/task_1_final.pth（共同 RainH 源）、两组 task_2_final.pth、所有复验边界及正式最终断点、clean_splits_v1 与 five_task_splits_v1。

状态：已全部移入系统回收站（gio trash 成功返回）；正式实验未删除，诊断记录留在原目录。

| 目录 | 权重数 | MiB |
|---|---:|---:|
| outputs/balanced_capacity_20261105 | 4 | 353.40 |
| outputs/balanced_half_smoke_20261105 | 5 | 456.35 |
| outputs/bounded_lora_batch64_smoke_20261104 | 4 | 385.43 |
| outputs/bounded_lora_smoke_20261104 | 4 | 383.02 |
| outputs/clean_smoke_20261029 | 8 | 854.32 |
| outputs/clean_smoke_branch_20261029 | 6 | 507.84 |
| outputs/five_task_smoke_20261105 | 6 | 677.45 |
| outputs/memory_source_b64_capacity_20261104 | 4 | 707.06 |
| outputs/memory_source_freq_smoke_20261104 | 7 | 690.07 |
| outputs/memory_source_raw_smoke_20261104 | 7 | 689.57 |
| outputs/retention_capacity_20261105 | 4 | 370.02 |
| outputs/retention_preserve_smoke_20261105 | 2 | 176.70 |
| outputs/retention_preserve_smoke_v2_20261105 | 5 | 456.79 |
| outputs/skip_branch_smoke_20261109 | 7 | 690.36 |
| outputs/skip_branch_v1_capacity_20261109 | 2 | 353.53 |
| outputs/skip_fused_capacity_20261109 | 2 | 353.53 |
| outputs/stability_control_smoke_20261104 | 4 | 383.04 |
| outputs/stability_replay_smoke_20261104 | 4 | 383.02 |
| outputs/stable_v1_smoke_20261106 | 8 | 852.41 |
| outputs/task_balanced_capacity_20261110 | 2 | 168.11 |
| outputs/task_balanced_smoke_20261110 | 7 | 722.77 |
| outputs/task_mean_pca_capacity_20261110 | 2 | 168.11 |
| outputs/task_mean_pca_smoke_20261110 | 7 | 722.77 |
| outputs/task_replicate_smoke_20261110 | 10 | 1079.56 |

## 精确目标

- `outputs/balanced_capacity_20261105/0.5/latest.pth` (92636452 bytes)
- `outputs/balanced_capacity_20261105/0.5/task_1_final.pth` (92636452 bytes)
- `outputs/balanced_capacity_20261105/1.0/latest.pth` (92648420 bytes)
- `outputs/balanced_capacity_20261105/1.0/task_1_final.pth` (92648420 bytes)
- `outputs/balanced_half_smoke_20261105/latest.pth` (105766432 bytes)
- `outputs/balanced_half_smoke_20261105/merged.pth` (62260846 bytes)
- `outputs/balanced_half_smoke_20261105/task_1_final.pth` (92599204 bytes)
- `outputs/balanced_half_smoke_20261105/task_2_best.pth` (112128314 bytes)
- `outputs/balanced_half_smoke_20261105/task_2_final.pth` (105766432 bytes)
- `outputs/bounded_lora_batch64_smoke_20261104/latest.pth` (114771296 bytes)
- `outputs/bounded_lora_batch64_smoke_20261104/merged.pth` (62259886 bytes)
- `outputs/bounded_lora_batch64_smoke_20261104/task_2_best.pth` (112349498 bytes)
- `outputs/bounded_lora_batch64_smoke_20261104/task_2_final.pth` (114771296 bytes)
- `outputs/bounded_lora_smoke_20261104/latest.pth` (113508192 bytes)
- `outputs/bounded_lora_smoke_20261104/merged.pth` (62259886 bytes)
- `outputs/bounded_lora_smoke_20261104/task_2_best.pth` (112349562 bytes)
- `outputs/bounded_lora_smoke_20261104/task_2_final.pth` (113508192 bytes)
- `outputs/clean_smoke_20261029/latest.pth` (108686560 bytes)
- `outputs/clean_smoke_20261029/merged.pth` (62259246 bytes)
- `outputs/clean_smoke_20261029/task_0_best.pth` (194403590 bytes)
- `outputs/clean_smoke_20261029/task_0_final.pth` (98125352 bytes)
- `outputs/clean_smoke_20261029/task_1_best.pth` (117653886 bytes)
- `outputs/clean_smoke_20261029/task_1_final.pth` (93238948 bytes)
- `outputs/clean_smoke_20261029/task_2_best.pth` (112767226 bytes)
- `outputs/clean_smoke_20261029/task_2_final.pth` (108686560 bytes)
- `outputs/clean_smoke_branch_20261029/latest.pth` (98877152 bytes)
- `outputs/clean_smoke_branch_20261029/merged.pth` (62259246 bytes)
- `outputs/clean_smoke_branch_20261029/task_1_best.pth` (85390206 bytes)
- `outputs/clean_smoke_branch_20261029/task_1_final.pth` (88339364 bytes)
- `outputs/clean_smoke_branch_20261029/task_2_best.pth` (98760506 bytes)
- `outputs/clean_smoke_branch_20261029/task_2_final.pth` (98877152 bytes)
- `outputs/five_task_smoke_20261105/latest.pth` (132359704 bytes)
- `outputs/five_task_smoke_20261105/merged.pth` (62261998 bytes)
- `outputs/five_task_smoke_20261105/task_3_best.pth` (125425398 bytes)
- `outputs/five_task_smoke_20261105/task_3_final.pth` (119212060 bytes)
- `outputs/five_task_smoke_20261105/task_4_best.pth` (138741106 bytes)
- `outputs/five_task_smoke_20261105/task_4_final.pth` (132359704 bytes)
- `outputs/memory_source_b64_capacity_20261104/frequency/latest.pth` (185351208 bytes)
- `outputs/memory_source_b64_capacity_20261104/frequency/task_0_final.pth` (185351208 bytes)
- `outputs/memory_source_b64_capacity_20261104/raw/latest.pth` (185351208 bytes)
- `outputs/memory_source_b64_capacity_20261104/raw/task_0_final.pth` (185351208 bytes)
- `outputs/memory_source_freq_smoke_20261104/latest.pth` (113475616 bytes)
- `outputs/memory_source_freq_smoke_20261104/merged.pth` (62259950 bytes)
- `outputs/memory_source_freq_smoke_20261104/task_0_final.pth` (98125672 bytes)
- `outputs/memory_source_freq_smoke_20261104/task_1_best.pth` (117654782 bytes)
- `outputs/memory_source_freq_smoke_20261104/task_1_final.pth` (99537636 bytes)
- `outputs/memory_source_freq_smoke_20261104/task_2_best.pth` (119066170 bytes)
- `outputs/memory_source_freq_smoke_20261104/task_2_final.pth` (113475616 bytes)
- `outputs/memory_source_raw_smoke_20261104/latest.pth` (113010656 bytes)
- `outputs/memory_source_raw_smoke_20261104/merged.pth` (62259950 bytes)
- `outputs/memory_source_raw_smoke_20261104/task_0_final.pth` (98125672 bytes)
- `outputs/memory_source_raw_smoke_20261104/task_1_best.pth` (117654782 bytes)
- `outputs/memory_source_raw_smoke_20261104/task_1_final.pth` (99737124 bytes)
- `outputs/memory_source_raw_smoke_20261104/task_2_best.pth` (119265722 bytes)
- `outputs/memory_source_raw_smoke_20261104/task_2_final.pth` (113010656 bytes)
- `outputs/retention_capacity_20261105/compressed/latest.pth` (101344228 bytes)
- `outputs/retention_capacity_20261105/compressed/task_1_final.pth` (101344228 bytes)
- `outputs/retention_capacity_20261105/preserve/latest.pth` (92650340 bytes)
- `outputs/retention_capacity_20261105/preserve/task_1_final.pth` (92650340 bytes)
- `outputs/retention_preserve_smoke_20261105/latest.pth` (92640100 bytes)
- `outputs/retention_preserve_smoke_20261105/task_1_final.pth` (92640100 bytes)
- `outputs/retention_preserve_smoke_v2_20261105/latest.pth` (105950752 bytes)
- `outputs/retention_preserve_smoke_v2_20261105/merged.pth` (62260654 bytes)
- `outputs/retention_preserve_smoke_v2_20261105/task_1_final.pth` (92644644 bytes)
- `outputs/retention_preserve_smoke_v2_20261105/task_2_best.pth` (112173690 bytes)
- `outputs/retention_preserve_smoke_v2_20261105/task_2_final.pth` (105950752 bytes)
- `outputs/skip_branch_smoke_20261109/latest.pth` (113265312 bytes)
- `outputs/skip_branch_smoke_20261109/merged.pth` (62262830 bytes)
- `outputs/skip_branch_smoke_20261109/task_0_final.pth` (98128616 bytes)
- `outputs/skip_branch_smoke_20261109/task_1_best.pth` (117657726 bytes)
- `outputs/skip_branch_smoke_20261109/task_1_final.pth` (99892772 bytes)
- `outputs/skip_branch_smoke_20261109/task_2_best.pth` (119421370 bytes)
- `outputs/skip_branch_smoke_20261109/task_2_final.pth` (113265312 bytes)
- `outputs/skip_branch_v1_capacity_20261109/latest.pth` (185354152 bytes)
- `outputs/skip_branch_v1_capacity_20261109/task_0_final.pth` (185354152 bytes)
- `outputs/skip_fused_capacity_20261109/latest.pth` (185354152 bytes)
- `outputs/skip_fused_capacity_20261109/task_0_final.pth` (185354152 bytes)
- `outputs/stability_control_smoke_20261104/latest.pth` (113517600 bytes)
- `outputs/stability_control_smoke_20261104/merged.pth` (62259886 bytes)
- `outputs/stability_control_smoke_20261104/task_2_best.pth` (112349562 bytes)
- `outputs/stability_control_smoke_20261104/task_2_final.pth` (113517600 bytes)
- `outputs/stability_replay_smoke_20261104/latest.pth` (113508192 bytes)
- `outputs/stability_replay_smoke_20261104/merged.pth` (62259886 bytes)
- `outputs/stability_replay_smoke_20261104/task_2_best.pth` (112349562 bytes)
- `outputs/stability_replay_smoke_20261104/task_2_final.pth` (113508192 bytes)
- `outputs/stable_v1_smoke_20261106/latest.pth` (107511584 bytes)
- `outputs/stable_v1_smoke_20261106/merged.pth` (62261614 bytes)
- `outputs/stable_v1_smoke_20261106/task_0_best.pth` (194410438 bytes)
- `outputs/stable_v1_smoke_20261106/task_0_final.pth` (98127720 bytes)
- `outputs/stable_v1_smoke_20261106/task_1_best.pth` (117656190 bytes)
- `outputs/stable_v1_smoke_20261106/task_1_final.pth` (93404388 bytes)
- `outputs/stable_v1_smoke_20261106/task_2_best.pth` (112932602 bytes)
- `outputs/stable_v1_smoke_20261106/task_2_final.pth` (107511584 bytes)
- `outputs/task_balanced_capacity_20261110/latest.pth` (88138246 bytes)
- `outputs/task_balanced_capacity_20261110/task_0_final.pth` (88138246 bytes)
- `outputs/task_balanced_smoke_20261110/latest.pth` (131260026 bytes)
- `outputs/task_balanced_smoke_20261110/merged.pth` (62262894 bytes)
- `outputs/task_balanced_smoke_20261110/task_0_final.pth` (87482758 bytes)
- `outputs/task_balanced_smoke_20261110/task_1_best.pth` (107011932 bytes)
- `outputs/task_balanced_smoke_20261110/task_1_final.pth` (109535520 bytes)
- `outputs/task_balanced_smoke_20261110/task_2_best.pth` (129064246 bytes)
- `outputs/task_balanced_smoke_20261110/task_2_final.pth` (131260026 bytes)
- `outputs/task_mean_pca_capacity_20261110/latest.pth` (88138246 bytes)
- `outputs/task_mean_pca_capacity_20261110/task_0_final.pth` (88138246 bytes)
- `outputs/task_mean_pca_smoke_20261110/latest.pth` (131260026 bytes)
- `outputs/task_mean_pca_smoke_20261110/merged.pth` (62262894 bytes)
- `outputs/task_mean_pca_smoke_20261110/task_0_final.pth` (87482758 bytes)
- `outputs/task_mean_pca_smoke_20261110/task_1_best.pth` (107011932 bytes)
- `outputs/task_mean_pca_smoke_20261110/task_1_final.pth` (109535520 bytes)
- `outputs/task_mean_pca_smoke_20261110/task_2_best.pth` (129064182 bytes)
- `outputs/task_mean_pca_smoke_20261110/task_2_final.pth` (131260026 bytes)
- `outputs/task_replicate_smoke_20261110/balanced_seed7/latest.pth` (131916346 bytes)
- `outputs/task_replicate_smoke_20261110/balanced_seed7/merged.pth` (62263726 bytes)
- `outputs/task_replicate_smoke_20261110/balanced_seed7/task_1_final.pth` (110188192 bytes)
- `outputs/task_replicate_smoke_20261110/balanced_seed7/task_2_best.pth` (129716726 bytes)
- `outputs/task_replicate_smoke_20261110/balanced_seed7/task_2_final.pth` (131916346 bytes)
- `outputs/task_replicate_smoke_20261110/mean_pca_seed7/latest.pth` (131916346 bytes)
- `outputs/task_replicate_smoke_20261110/mean_pca_seed7/merged.pth` (62263726 bytes)
- `outputs/task_replicate_smoke_20261110/mean_pca_seed7/task_1_final.pth` (110188192 bytes)
- `outputs/task_replicate_smoke_20261110/mean_pca_seed7/task_2_best.pth` (129716726 bytes)
- `outputs/task_replicate_smoke_20261110/mean_pca_seed7/task_2_final.pth` (131916346 bytes)
