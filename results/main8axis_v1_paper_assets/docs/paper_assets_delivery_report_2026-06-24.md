# StyleBlend-Bench main8axis_v1 论文图表交付报告

日期：2026-06-24

## 1. 交付概览

本次任务已完成 `main8axis_v1` 主实验结果的论文资产整理。所有交付物均从已有 final analysis 结果中可复现生成，没有重新运行 generation、judge 或 embedding。

总脚本：

`/home/data_cpfs/lizihua/sbe/scripts/14_prepare_paper_assets.py`

论文资产目录：

`/home/data_cpfs/lizihua/sbe/results/main8axis_v1_paper_assets/`

脚本运行结果：

```text
created_figures: 20
created_tables: 16
missing_inputs: []
missing_count = 0
```

说明：20 个 figure 文件对应 10 张图，每张图都有 PDF 和 PNG；16 个 table 文件对应 8 组表格，每组包含 CSV 和 Markdown。

## 2. 核心文件

- Manifest: `/home/data_cpfs/lizihua/sbe/results/main8axis_v1_paper_assets/docs/asset_manifest.md`
- 写作摘要: `/home/data_cpfs/lizihua/sbe/results/main8axis_v1_paper_assets/docs/paper_results_digest.md`
- 图 caption: `/home/data_cpfs/lizihua/sbe/results/main8axis_v1_paper_assets/captions/figure_captions.md`
- 表 caption: `/home/data_cpfs/lizihua/sbe/results/main8axis_v1_paper_assets/captions/table_captions.md`
- 图目录: `/home/data_cpfs/lizihua/sbe/results/main8axis_v1_paper_assets/figures/`
- 表目录: `/home/data_cpfs/lizihua/sbe/results/main8axis_v1_paper_assets/tables/`

## 3. 主要实验结论

主结论：

`Qwen3.5-9B > Qwen3.5-35B-A3B > Qwen3.5-4B`

该排序在 bge-m3 和 text2vec 两套 embedding projection 指标下保持一致。

### 3.1 主模型对比

来源：

`tables/table_1_main_model_comparison.md`

| Model | Projection MAE mean | Projection Spearman mean | Normalized drift | NN mean rank | Length violation rate |
|---|---:|---:|---:|---:|---:|
| Qwen3.5-9B | 0.118 | 0.890 | 0.485 | 2.944 | 0.138 |
| Qwen3.5-35B-A3B | 0.134 | 0.864 | 0.473 | 2.959 | 0.200 |
| Qwen3.5-4B | 0.157 | 0.810 | 0.523 | 3.253 | 0.242 |

解释：

- 9B 的 Projection MAE 最低，说明其对目标风格比例的 placement 最准确。
- 35B-A3B 的 normalized drift 更低，但 projection MAE 高于 9B，说明它更靠近 A-B interpolation line，却没有更准确地落在目标比例位置。
- 4B 在 projection、nearest-neighbor 和 length compliance 上整体最弱。

### 3.2 Embedding 稳健性

来源：

`tables/table_2_embedding_robustness.md`

关键数字：

- bge-m3: 9B MAE `0.126`, 35B-A3B MAE `0.140`, 4B MAE `0.164`
- text2vec: 9B MAE `0.111`, 35B-A3B MAE `0.127`, 4B MAE `0.151`

解释：

两套 embedding 都给出相同模型排序，因此主结论不是某一个 embedding backend 的偶然结果。

### 3.3 风格轴难度

来源：

`tables/table_3_axis_projection_mae_pivot.md`

| Axis | Qwen3.5-4B | Qwen3.5-9B | Qwen3.5-35B-A3B |
|---|---:|---:|---:|
| Detail | 0.211 | 0.187 | 0.191 |
| Emotion | 0.154 | 0.103 | 0.116 |
| Empathy | 0.141 | 0.118 | 0.128 |
| Expertise | 0.123 | 0.085 | 0.118 |
| Formality | 0.137 | 0.082 | 0.089 |
| Humor | 0.157 | 0.136 | 0.145 |
| Objectivity | 0.147 | 0.080 | 0.133 |
| Politeness | 0.189 | 0.155 | 0.151 |

解释：

- Detail / concision-detail 是最难轴之一，三个模型误差都偏高。
- Formality 和 Objectivity 对 9B 最友好，9B 在这些轴上误差最低。
- 轴级别差异明显，论文正文中应避免只报告全局平均。

### 3.4 长度诊断

来源：

`tables/table_5_length_diagnostics.md`

关键数字：

- 9B length violation rate: `0.138`
- 35B-A3B length violation rate: `0.200`
- 4B length violation rate: `0.242`

解释：

9B 长度合规性最好。35B-A3B 更容易过长，4B 更容易过短。长度问题需要作为诊断项报告，但当前结果不支持把主 ranking 解释为长度污染。

### 3.5 Scalar / Pairwise 辅助评价

来源：

`tables/table_6_scalar_and_pairwise_auxiliary.md`

关键数字：

- Scalar judge MAE: 9B `0.175`, 35B-A3B `0.182`, 4B `0.244`
- Higher-alpha pairwise win rate: 9B `0.709`, 35B-A3B `0.751`, 4B `0.645`
- Bidirectional consistency: 9B `0.844`, 35B-A3B `0.706`, 4B `0.827`

解释：

Scalar 和 pairwise 大体支持 projection 结论，但只能作为辅助证据。原因是 judge assignment 不对称，并且 pairwise 存在 order effect / bidirectional inconsistency。

## 4. 推荐放入论文正文的图

建议正文优先使用以下 5 张图：

1. `figures/fig_3_model_comparison_projection_mae_ci.pdf`
   - 最直接展示主 ranking 和 bootstrap CI。
2. `figures/fig_1_model_calibration_curves_bge_m3.pdf`
   - 展示 bge-m3 下 calibration curve。
3. `figures/fig_2_model_calibration_curves_text2vec.pdf`
   - 展示 text2vec 下同样 ranking，支持 embedding robustness。
4. `figures/fig_4_axis_heatmap_projection_mae.pdf`
   - 展示 8 个风格轴的难度差异。
5. `figures/fig_6_geometry_decomposition_bar.pdf`
   - 支持“9B 优势来自 ratio placement，不是 lower off-axis drift”的解释。

可放入 appendix 或 diagnostics 的图：

- `figures/fig_5_axis_heatmap_off_axis_drift.pdf`
- `figures/fig_7_length_violation_heatmap.pdf`
- `figures/fig_8_projection_vs_scalar_judge_scatter.pdf`
- `figures/fig_9_nearest_neighbor_rank_by_model.pdf`
- `figures/fig_10_pairwise_auxiliary_audit.pdf`

## 5. 推荐放入论文正文的表

建议正文优先使用以下 4 张表：

1. `tables/table_1_main_model_comparison.md`
   - 主结果表。
2. `tables/table_2_embedding_robustness.md`
   - 支持 embedding robustness。
3. `tables/table_3_axis_projection_mae_pivot.md`
   - 简洁展示轴级结果。
4. `tables/table_4_geometry_decomposition.md`
   - 支持几何解释。

建议 appendix 使用：

- `tables/table_3_axis_level_projection.md`
- `tables/table_5_length_diagnostics.md`
- `tables/table_6_scalar_and_pairwise_auxiliary.md`
- `tables/table_7_nearest_neighbor_and_out_of_range.md`

## 6. 可以写入论文的 claim

可以写：

1. Qwen3.5-9B 在本实验的三种 Qwen-family 模型中有最佳 style-ratio calibration。
2. bge-m3 和 text2vec 给出一致的模型排序，支持 measurement robustness。
3. 35B-A3B 的 off-axis drift 更低，但 ratio placement 不如 9B。
4. 4B 在 projection calibration 和 length compliance 上都最弱。
5. Scalar judge 和 pairwise audit 支持主趋势，但只能作为辅助验证。
6. 不同风格轴难度差异显著，Detail / concision-detail 是较难轴之一。

## 7. 不应写入论文的 claim

暂时不要写：

1. 不要声称 9B 对所有模型家族都最优；cross-family 实验尚未运行。
2. 不要声称结果已经 human-validated；human validation sample 已准备但尚未标注。
3. 不要把 scalar judge 或 pairwise score 当作主指标。
4. 不要声称 temperature robustness；temperature ablation 只是计划，尚未执行。
5. 不要声称模型越大 style control 越强；当前结果恰好反驳这一简单结论。

## 8. 质量检查记录

已完成检查：

- `python sbe/scripts/14_prepare_paper_assets.py` 运行成功。
- `missing_count = 0`。
- `created_figures = 20`，即 10 张图的 PDF + PNG。
- `created_tables = 16`，即 8 组表的 CSV + Markdown。
- 所有图表文件非空。
- PNG 图像分辨率均超过 1300 px 宽/高。
- 抽查图：
  - `fig_3_model_comparison_projection_mae_ci.png`
  - `fig_4_axis_heatmap_projection_mae.png`
  - `fig_10_pairwise_auxiliary_audit.png`
- 修复过热图深色格文字不可读的问题，已重跑脚本。
- `compileall` 已通过：
  - `/home/data_cpfs/lizihua/sbe/scripts/14_prepare_paper_assets.py`
  - `/home/data_cpfs/lizihua/sbe/results/main8axis_v1_paper_assets/scripts/14_prepare_paper_assets.py`

## 9. 下一步建议

优先级建议：

1. 使用 `paper_results_digest.md` 起草论文 Results 段落。
2. 将 Figure 3、Figure 1/2、Figure 4、Figure 6 作为正文主图。
3. 将 Table 1、Table 2、Table 3 pivot、Table 4 作为正文主表。
4. 如果要增强论文可信度，下一步应先做人类标注，而不是继续扩大自动生成。
5. 如果审稿风险集中在模型泛化，再执行 cross-family subset。
6. 如果审稿风险集中在采样温度，再执行 temperature ablation。
