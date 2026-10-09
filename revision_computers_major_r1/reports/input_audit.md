# Input audit

Generated: 2026-10-09T14:33:02.319585Z
Git commit: `b97dc8500e3e10c98c54806dfcdbfe9596394921`
Project: `/home/data_cpfs/lizihua/sbe`
Legacy input (read only): `/home/data_cpfs/lizihua/sbe/revision_computers_final`

## Scenario source

- SHA-256: `c6b7ba3e514eff070ab7ccd7d6a5929dc2a3ee0147db2b7eed802b859f6b16fc`
- Rows: 320; duplicate IDs: 0
- Axes: {"concision_detail": 40, "emotion_apology": 40, "empathy_clinical": 40, "expertise_explain": 40, "formality_request": 40, "humor_neutral": 40, "objectivity_cat": 40, "politeness_refusal": 40}
- Atomic checks per scenario: min 2, max 4, missing 0

## Main Qwen experiment

- Rows: 33,600; expected 33,600; duplicate keys: 0; missing combinations: 0
- Models: {"qwen35_35b_a3b": 6720, "qwen35_4b": 6720, "qwen35_9b": 6720, "qwen36_27b": 6720, "qwen36_35b_a3b": 6720}
- Ratios: [0.0, 0.16666666666666666, 0.3333333333333333, 0.5, 0.6666666666666666, 0.8333333333333334, 1.0]; seeds: [1, 2, 3]
- Prompt: ['computers_explicit_ratio_v1']; temperature: [0.7]; top-p: [0.9]; max new tokens: [256]
- Reuse decision: all valid rows are reused; the complete original main matrix is not regenerated.

## Existing robustness experiment

- Rows: 9,600; expected 9,600; scenarios: 40; duplicate keys: 0
- Reused main rows: 2,400; other-condition rows: 7,200
- Existing models: {"qwen35_4b": 2400, "qwen35_9b": 2400, "qwen36_27b": 2400, "qwen36_35b_a3b": 2400}
- `computers_explicit_ratio_v1` at temperature `0.2`: 2,400 rows
- `computers_explicit_ratio_v1` at temperature `0.7`: 2,400 rows
- `computers_explicit_ratio_v2` at temperature `0.2`: 2,400 rows
- `computers_explicit_ratio_v2` at temperature `0.7`: 2,400 rows
- Reuse decision: reuse only exact configuration keys and verified prompt/text hashes; expand to 320 scenarios and all five Qwen models in the new output tree.

## Evaluations and embeddings

- Main fixed-Qwen evaluations: 33,600; missing generation IDs: 0
- Robustness fixed-Qwen evaluations: 9,600; missing generation IDs: 0
- Embedding archives: 15 files; every archive records shape and duplicate run-ID count in `manifests/input_audit.json`.

## External API status

- The attempted `deepseek-r1` and `deepseek-v4-pro-0813` routes were blocked before inference by `missing exchange rate from null to CNY`; both returned zero usage tokens.
- The author replaced that planned condition with lightweight `gpt-6-luna`. A separate 20-output real-prompt smoke test passed 20/20 before formal generation; selection and cost evidence are in `manifests/external_generator_selection.json`.
- Formal GPT-6-Luna outcome: 6,720/6,720 request identities attempted; 6,719 usable outputs; 1 content failures; 0 transport failures; estimated cost USD 0.476949.
- The independent second judge is the distinct `gpt-6-sol` alias. Generator and second judge share the GPT family/provider, while the local Qwen judge supplies the cross-family evaluator comparison. Credentials are read only from `SBE_IAI_API_KEY` and are excluded from this audit.
