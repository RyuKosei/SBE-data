# Benchmark construction facts

Status: asset-grounded reconstruction for Computers major revision (2026-10-09). This report does not infer unrecorded human review or provenance.

## Frozen artifact and scope

- Scenario artifact: `data/scenarios/styleblend_8axis_scenarios_v1.jsonl` (320 JSONL records; 40 per axis).
- Language actually represented: Chinese. No translated prompt set or cross-language validation is present.
- Each record fixes a core semantic request (`content`), 2–4 atomic preservation checks, an A/B style pair, one domain tag, and an allowed output-character interval.
- The selected scope is everyday short-text style control. It is not an exhaustive style taxonomy. Persuasive generation, open-ended creative writing, safety-sensitive tasks, and specialist-domain task suites are not represented as dedicated axes.

## Eight axes and intended scene types

| axis_id | measured attribute | endpoint A | endpoint B | recorded scene-type instruction | final N |
|---|---|---|---|---|---:|
| emotion_apology | 愤怒强硬程度 | 冷静、客气、宽容，不指责对方 | 愤怒、强硬、明确表达投诉意愿，但不使用脏话或人身攻击 | 服务延误、售后、等待、误会等回应 | 40 |
| expertise_explain | 专业术语密度 | 面向五岁儿童的幼儿园老师，生动、简单、带比喻 | 面向物理学博士后的学术解释，术语密集、严谨、可含公式或专业概念 | 可用浅显或专业方式解释的科学/常识问题 | 40 |
| objectivity_cat | 抒情修辞程度 | 客观观察记录，像医学或生物学记录，克制、无感情色彩 | 抒情现代散文诗，意象丰富、情感浓、辞藻华丽 | 动物、天气、室内物体、自然景象观察 | 40 |
| formality_request | 正式程度 | 口语、随意、像朋友之间聊天 | 正式、公文式、措辞规范、礼貌但有距离感 | 请求、通知、说明、预约、确认 | 40 |
| politeness_refusal | 礼貌委婉程度 | 直白、不委婉、略显生硬，但不侮辱 | 委婉、礼貌、照顾对方面子，给出缓冲理由 | 拒绝邀请、请求或方案 | 40 |
| concision_detail | 详细程度 | 极简摘要，只保留必要信息 | 详尽说明，包含背景、原因、步骤或补充细节 | 可简写或详细展开的说明 | 40 |
| humor_neutral | 幽默吐槽程度 | 中性、平实、无玩笑 | 幽默、轻微吐槽、有俏皮表达，但不跑题 | 日常小麻烦、产品体验、学习工作场景 | 40 |
| empathy_clinical | 共情温度 | 临床冷静、客观描述、少情绪安慰 | 高共情、安慰性强、承认对方感受并鼓励 | 压力、失落、身体不适、担忧回应，避免医疗诊断 | 40 |

The source definitions are in `configs/style_axes.yaml`, lines 1–56.

## Observed record distributions

- Atomic checks: 32 scenarios have 2 checks, 207 have 3, and 81 have 4.
- Output intervals: 80 scenarios use 70–130, 40 use 70–160, 80 use 80–140, 80 use 80–150, and 40 use 90–160 characters.
- Core semantic text contains 14–42 Chinese characters in the final pool.
- Domain tags (N): travel 33, public_service 32, technology 32, family 30, daily_life 30, education 29, consumer 28, community 28, workplace 27, environment 14, science 13, housing 13, and health_non_diagnostic 11.
- IDs are deterministic axis-prefixed identifiers such as `concision_detail_001`; the frozen artifact is labeled `styleblend_8axis_scenarios_v1`.

## Construction and screening that are actually recorded

1. The candidate script requested JSON candidates with 12–80 Chinese characters, 2–4 atomic checks, a controlled domain tag, a risk note, and endpoint compatibility (`scripts/07_generate_scenario_candidates.py`, lines 129–170). It defaults to 80 candidates per axis (`scripts/07_generate_scenario_candidates.py`, line 264).
2. The surviving asset report records 650 candidates total: 80 for seven axes and 90 for `objectivity_cat`; 0 generation failures; then 50 filtered records per axis and 40 final records per axis (`docs/scenario_pool_review_8axis_v1.md`, lines 13–22; `docs/next_stage_blocking_diagnostics_summary_2026-06-23.md`, section 3).
3. Rule validation checked required fields, 12–80 Chinese characters, 2–4 nonempty checks, output interval bounds of 40–220 with lower < upper, distinct nonempty A/B endpoints, domain tag, and risk note (`scripts/06_curate_scenarios.py`, lines 47–101).
4. The curation code rejected exact normalized duplicates and within-axis bigram-Jaccard near duplicates at the CLI default threshold 0.45 (`scripts/06_curate_scenarios.py`, lines 104–154).
5. Final selection round-robined across recorded domains to improve domain coverage, then reassigned stable final IDs and a code-generated `review_rank` (`scripts/09_review_scenario_pool.py`, lines 61–82). `review_rank` is an algorithmic ordering field, not evidence of human approval.
6. The final report found no rule-based fact/safety flags and separately warned that this did not replace manual review (`docs/scenario_pool_review_8axis_v1.md`, line 185). It also listed sparse-content cautions for some `concision_detail` and `expertise_explain` scenes.

The candidate-generation model/checkpoint, who (if anyone) performed a human spot-check, any human adjudication protocol, and human acceptance/rejection decisions are **not recorded in the available construction assets**. A contemporaneous summary explicitly describes the pool as “LLM-generated and rule-reviewed, not deeply human-reviewed.” These details must not be reconstructed from memory or presented as verified.

## Complete frozen example

Scenario `concision_detail_001`:

```json
{
  "scenario_id": "concision_detail_001",
  "axis_id": "concision_detail",
  "content": "说明如何选购适合的冬季保暖衣物。",
  "style_a": "极简摘要，只保留必要信息",
  "style_b": "详尽说明，包含背景、原因、步骤或补充细节",
  "attribute_name": "详细程度",
  "content_checks": ["选购冬季衣物", "保暖功能"],
  "length_range_zh_chars": [70, 160],
  "domain_tag": "consumer",
  "risk_note": "普通低风险场景。"
}
```

The P1 prompt at the midpoint is rendered from the locked implementation (`revision_computers_major_r1/scripts/prompts.py`, lines 28–45):

```text
/no_think
严禁输出 Thinking Process、Analysis、推理过程、步骤说明、标题或 Markdown 列表。
你只能输出最终中文正文，第一句话就必须是正文内容。

你将根据一个固定核心语义生成中文短文本。

核心语义：
说明如何选购适合的冬季保暖衣物。

风格A：
极简摘要，只保留必要信息

风格B：
详尽说明，包含背景、原因、步骤或补充细节

目标风格比例：
alpha = 3/6（50% 风格A + 50% 风格B）。

硬性要求：
1. 必须保留核心语义，不添加与核心语义冲突的新事实。
2. 正文必须至少 70 个汉字，最多 160 个汉字；少于 70 个汉字不合格。
3. 只输出正文，不要解释比例，不要列标题。
4. 不要输出任何英文分析，不要输出 Thinking Process。
```

For the external GPT-6-Luna route, only the leading Qwen-specific `/no_think` control line is removed; the original and sent prompt hashes are stored separately. The Chinese task content is not translated. This is the author-approved replacement for the DeepSeek route that failed before inference.
