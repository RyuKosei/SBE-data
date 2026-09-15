# MDPI Computers 扩展实验结果摘要

## 实验覆盖

- Qwen 主实验：5 个 checkpoint × 320 scenarios × 7 ratios × 3 seeds，共 33,600 条；唯一键完整，无空响应，50 条达到长度截断条件并已标记。
- 稳健性实验：40 scenarios × 4 代表模型 × 5 内部比例 × 3 seeds × 4 条件，共 9,600 条，其中 P1/T0.7 的 2,400 条复用主实验，新增 7,200 条。
- 三编码器：BGE-M3、原实验 exact text2vec、multilingual-E5-large-instruct，共 100,800 条样本级几何记录、4,800 条 scenario 级汇总。
- 自动评价：主实验 33,600 条和稳健性 9,600 条均由固定 Qwen3.6-35B-A3B evaluator、temperature 0、固定 schema 完成。
- 所有置信区间均以 scenario 为统计单位，按风格轴分层做 10,000 次 bootstrap；同一 scenario/ratio/model 内先聚合 seed 信息。

## 核心结果

### 1. 端点偏差

旧实现把两个机械零误差端点计入 MAE。对于完整七点轨迹，7-point MAE 恒等于 5-point interior MAE 的 5/7，因此会低估内部点误差 28.57%。旧 BGE-M3 和 text2vec 的修正值均在预设 0.002 容差内复现。

### 2. 主模型排名

三编码器给出的 ICE 排名完全相同。三编码器描述性均值为：

| 模型 | ICE | Drift | Monotonicity |
|---|---:|---:|---:|
| Qwen3.5-9B | 0.1561 | 0.6802 | 0.7995 |
| Qwen3.6-35B-A3B | 0.1728 | 0.6914 | 0.7515 |
| Qwen3.5-35B-A3B | 0.1826 | 0.6748 | 0.6902 |
| Qwen3.6-27B | 0.1892 | 0.6735 | 0.6473 |
| Qwen3.5-4B | 0.2018 | 0.7456 | 0.5429 |

规模增大不保证 ICE 下降。Qwen3.5-35B-A3B 的 drift 略低于 Qwen3.5-9B，但 ICE 更高，说明比例误差和轴外漂移反映不同失败模式。

### 3. 风格轴差异

Detail 轴最难：总体 ICE 0.2366、drift 0.8960、monotonicity 0.1375。Formality 轴相对容易：ICE 0.1396、monotonicity 0.8852。Detail、Humor、Empathy、Expertise 的长度、信息密度和语用策略耦合结果见 `metrics/content_style_coupling_focus_axes.csv`。

### 4. 端点质量与自锚点限制

固定 evaluator 判断的端点风格强度差为：Qwen3.5-4B 49.46、Qwen3.5-9B 66.75、Qwen3.5-35B-A3B 71.51、Qwen3.6-27B 77.26、Qwen3.6-35B-A3B 72.10。4B 的端点范围明显压缩，但 self-anchor 会把该局部范围重新缩放为 0–1，因此 self-anchor 结果不能单独证明绝对控制能力。

### 5. 线性近似的边界

七点轨迹的模型级 PC1 解释率只有 0.349–0.408，端点方向与 PC1 的绝对余弦为 0.647–0.755，ordered path length ratio 为 4.64–5.20，平均离散曲率为 116.4°–118.3°。这不支持“风格在表示空间中天然线性”的强结论，只支持把端点投影作为局部诊断量，并同时报告 drift、曲率和经验轨迹结果。

### 6. 编码器与端点分离度

增加第三编码器没有改变模型 ICE 排名，但绝对数值存在编码器依赖。端点分离度与 ICE 的 Spearman 为 −0.317 至 −0.413；与 normalized drift 的相关更强，为 −0.662 至 −0.811。低分离度端点明显放大测量不稳定性。

### 7. 自动内容与 scalar judge

自动 content-preservation pass：

- Qwen3.5-4B：98.05%
- Qwen3.5-9B：97.72%
- Qwen3.5-35B-A3B：98.15%
- Qwen3.6-27B：96.79%
- Qwen3.6-35B-A3B：98.17%

固定 scalar judge 的 MAE 排名与几何 ICE 不完全一致：Qwen3.6-35B-A3B 0.2224 最低，Qwen3.5-4B 0.2951 最高。自动 content pass 很高，但在真实人工评分返回前不得当作已验证的人工一致性证据。

### 8. Prompt/temperature 稳健性

P2 相对 P1：

- scalar style MAE 增加 0.0167，95% CI 0.0065–0.0274；
- content pass 增加 0.0106，95% CI 0.0031–0.0202；
- 平均缩短 11.58 个字符，95% CI −17.57 至 −4.61。

温度从 0.7 降至 0.2 对 style MAE 没有清晰影响（0.0010，CI −0.0034 至 0.0056），但自然度增加 0.0410（CI 0.0171–0.0650）。style-error 交互项不显著偏离零。说明语义等价的 prompt 改写比本次温度变化更容易改变控制误差和长度。

### 9. 失败类型

| 类型 | 比例 |
|---|---:|
| late under-response | 58.7% |
| curved trajectory | 51.2% |
| early overshoot | 49.8% |
| non-monotonic reversal | 26.6% |
| near-constant response | 12.0% |
| content–style entanglement | 7.6% |
| length-dominated control | 4.0% |
| out-of-range extrapolation | 2.8% |

一个重要负面发现是 late under-response 超过一半，因此不能用较好的平均 ICE 掩盖大量场景级失败。

### 10. 推理效率

只在两张 L20Z、TP=2、BF16、concurrency=32 的相同条件下作直接速度比较：

- Qwen3.5-35B-A3B：P95 3.46 s；
- Qwen3.6-35B-A3B：P95 3.67 s；
- Qwen3.6-27B：P95 5.35 s。

4B 和 9B 使用单 GPU 且请求并发不同，只保留描述性位置，不作直接速度优劣结论。

## 预注册假设状态

- H1：支持。排除端点后误差按定义升高。
- H2：支持。低端点分离度与更高误差/漂移显著相关。
- H3：external pending；等待共享端点人工改写与双作者检查。
- H4：支持为描述性结论；规模增大不保证 ICE 单调下降。
- H5：支持。ICE 与 drift 的相关为中等强度（0.338–0.492），不是同一指标。
- H6：编码器依赖已经完成；人类相关性 external pending。

## 不允许写入论文的过度结论

- 不得声称风格在人类感知空间或句向量空间中全局线性。
- 不得把 Qwen 家族结果直接推广到所有大模型。
- 不得把 self-anchor 当作绝对跨模型控制能力证明。
- 不得把五个模型上的描述性关系称为 scaling law。
- 不得把尚未用真人验证的 LLM evaluator 当作人工金标准。
