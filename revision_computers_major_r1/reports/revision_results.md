# Revision results

Generated from final CSV/JSON artifacts at 2026-10-09T15:54:40.940611Z. Numbers below are traceable to the named tables; no page or line numbers are inferred for the manuscript.

## Completion and denominators

The retained local-Qwen main experiment contains 33,600 outputs. The 2×2 robustness matrix contains 96,000 planned condition records, of which 95,994 contain valid text and 6 are explicitly recorded failures. The deduplicated evaluation index contains 112,313 usable outputs; 7 planned identities are explicit generation failures, so all 112,320 planned identities are accounted for without imputation.

The cross-family GPT-6-Luna generation run is completed: 6,720/6,720 request identities were attempted, yielding 6,719 usable outputs; the non-usable count is 1, with status retained. It is a different model family from Qwen, but it shares the GPT family/provider with the GPT-6-Sol judge; the local Qwen judge supplies the cross-family evaluator comparison. Six-model tables retain valid denominators and never impute non-usable outputs.

## Anchor protocols and encoder sensitivity

Under BGE-M3, mean internal ICE with self anchors was: qwen35_9b 0.161, gpt6_luna_api 0.163, qwen36_35b_a3b 0.178, qwen35_35b_a3b 0.186, qwen36_27b 0.190, qwen35_4b 0.209. With the fixed GPT reference pair it was: qwen35_9b 0.176, gpt6_luna_api 0.178, qwen36_35b_a3b 0.190, qwen35_35b_a3b 0.191, qwen36_27b 0.196, qwen35_4b 0.212. The model-order correlation between those two protocols was Spearman 1.000 with 0 pairwise flips among 15 model pairs. The pooled-Qwen and other-encoder results are reported without requiring identical rankings in `table_anchor_encoder_sensitivity.csv`.

All 320 GPT reference pairs were generated once and frozen. Automatic Qwen screening was valid for 639/640 texts; the text-level length-pass rate was 98.0%, both endpoints passed content checks in 317/320 scenes, and the mean automatic B-minus-A intensity span was 78.81 points. These are automatic diagnostics, not human certification.

## Independent judge

Across matched outputs, Qwen and GPT intensity scores had Pearson r=0.910 and Spearman rho=0.870; after demeaning within target-ratio × axis strata, the corresponding correlations were 0.862 and 0.820. The mean absolute intensity difference was 9.59 points.

Content-pass raw agreement was 76.2% and Cohen's kappa was 0.078 (Qwen pass prevalence 98.2%; GPT 75.1%). Kappa is interpreted alongside the confusion table because high prevalence can depress it.

The descriptive six-model style-MAE ordering had Spearman correlation 0.943 across judges; 2/6 models changed ordinal position. Exact judge-specific ranks and scenario-matched pairwise differences are retained in `dual_judge_model_rankings.csv` and `dual_judge_model_pairwise_style_mae.csv` rather than reduced to a universal ranking claim.

- qwen geometry–judge residual association: r=0.576 [0.546, 0.604]; after residualizing character length: r=0.576 [0.546, 0.604] (320 scenario clusters).
- gpt geometry–judge residual association: r=0.618 [0.588, 0.647]; after residualizing character length: r=0.619 [0.588, 0.647] (320 scenario clusters).

## Full robustness

- external_reference_ice: prompt_effect=0.005 [0.002, 0.008]; temperature_effect=-0.000 [-0.001, 0.001]; interaction=0.004 [0.002, 0.007].
- external_reference_drift: prompt_effect=-0.050 [-0.057, -0.042]; temperature_effect=-0.014 [-0.016, -0.012]; interaction=0.003 [-0.001, 0.007].
- external_reference_out_of_range: prompt_effect=0.001 [-0.000, 0.003]; temperature_effect=-0.000 [-0.001, 0.000]; interaction=0.000 [-0.001, 0.001].
- qwen_style_mae: prompt_effect=0.013 [0.009, 0.016]; temperature_effect=0.001 [-0.001, 0.003]; interaction=0.003 [-0.001, 0.007].
- qwen_content_pass: prompt_effect=0.004 [0.001, 0.007]; temperature_effect=0.001 [-0.000, 0.003]; interaction=-0.000 [-0.003, 0.003].
- qwen_naturalness: prompt_effect=-0.034 [-0.058, -0.011]; temperature_effect=0.055 [0.047, 0.062]; interaction=-0.006 [-0.022, 0.010].
- gpt_style_mae: prompt_effect=0.015 [0.011, 0.019]; temperature_effect=0.002 [0.000, 0.004]; interaction=0.003 [-0.000, 0.006].
- gpt_content_pass: prompt_effect=0.104 [0.090, 0.117]; temperature_effect=0.013 [0.007, 0.018]; interaction=-0.006 [-0.015, 0.004].
- gpt_naturalness: prompt_effect=-0.096 [-0.116, -0.077]; temperature_effect=0.076 [0.069, 0.084]; interaction=-0.025 [-0.041, -0.008].
- character_length: prompt_effect=-12.379 [-13.700, -10.951]; temperature_effect=-1.533 [-2.239, -0.794]; interaction=0.680 [-0.710, 2.127].

Effects use the prespecified signs: mean-temperature(P2−P1), mean-prompt(low−high), and the difference-in-differences interaction. Three seeds and five internal ratios are first averaged within scenario; the five Qwen models are then equally weighted.

## Efficiency and scope limits

The corrected local efficiency figure contains only qwen35_35b_a3b, qwen36_27b, and qwen36_35b_a3b because their logs match on two NVIDIA L20Z GPUs, TP=2, BF16, no quantization, and concurrency 32. The one-GPU 4B/9B deployments remain in a descriptive table. API latency is not placed on the local-GPU efficiency frontier.

The evidence supports an operational, embedding-and-automatic-judge diagnostic for Chinese short-text style control. It does not establish human perceptual linearity, absolute style truth, cross-language generalization, universal behavior across LLM families, or elimination of evaluator bias.

## Budget

Recorded GPT token usage implies an estimated cumulative cost of USD 183.66 under the author-supplied prices. Adding conservative allowances for calls that may have been in flight at two concurrency restarts and for two inexpensive comparison pilots whose raw usage was not retained gives USD 183.94, below the USD 240 operational cap. The relay did not expose actual wallet balance. DeepSeek dollar cost is not imputed without an internal price or billable response.
