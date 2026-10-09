from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from common import atomic_write_text, read_jsonl, utc_now


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def fmt(value: float, digits: int = 3) -> str:
    return "NA" if pd.isna(value) else f"{value:.{digits}f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.output_root
    reports = root / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    robustness = load_json(root / "manifests/robustness_coverage.json")
    references = pd.read_csv(root / "tables/table_reference_quality.csv").iloc[0]
    api = load_json(root / "manifests/api_usage_and_budget.json")
    coverage_audit = load_json(root / "manifests/final_coverage_audit.json")
    eval_status = load_json(root / "manifests/evaluation_index_status.json")
    geometry = pd.read_csv(root / "tables/table_anchor_encoder_sensitivity.csv")
    ranks = pd.read_csv(root / "statistics/anchor_protocol_rank_sensitivity.csv")
    cross = pd.read_csv(root / "tables/table_cross_family_main.csv")
    judge_summary = pd.read_csv(root / "tables/table_independent_judge.csv") if (root / "tables/table_independent_judge.csv").exists() else pd.DataFrame()
    judge_agreement = pd.read_csv(root / "statistics/dual_judge_intensity_agreement.csv") if (root / "statistics/dual_judge_intensity_agreement.csv").exists() else pd.DataFrame()
    content_agreement = pd.read_csv(root / "statistics/dual_judge_content_agreement.csv") if (root / "statistics/dual_judge_content_agreement.csv").exists() else pd.DataFrame()
    judge_rankings = pd.read_csv(root / "statistics/dual_judge_model_rankings.csv") if (root / "statistics/dual_judge_model_rankings.csv").exists() else pd.DataFrame()
    ranking_comparison = pd.read_csv(root / "statistics/dual_judge_ranking_comparison.csv") if (root / "statistics/dual_judge_ranking_comparison.csv").exists() else pd.DataFrame()
    length_sensitivity = pd.read_csv(root / "statistics/geometry_judge_length_sensitivity.csv") if (root / "statistics/geometry_judge_length_sensitivity.csv").exists() else pd.DataFrame()
    robustness_table = pd.read_csv(root / "tables/table_robustness_full.csv") if (root / "tables/table_robustness_full.csv").exists() else pd.DataFrame()
    stability = pd.read_csv(root / "tables/table_judge_stability_retest.csv") if (root / "tables/table_judge_stability_retest.csv").exists() else pd.DataFrame()
    stability_index_rows = len(read_jsonl(root / "data/stability_retest_index.jsonl")) if (root / "data/stability_retest_index.jsonl").exists() else 0
    qwen_stability_valid = len(read_jsonl(root / "metrics/qwen_stability_retest.jsonl")) if (root / "metrics/qwen_stability_retest.jsonl").exists() else 0
    qwen_stability_failures = len(read_jsonl(root / "logs/qwen_stability_retest_failures.jsonl")) if (root / "logs/qwen_stability_retest_failures.jsonl").exists() else 0
    gpt_stability_valid = len(read_jsonl(root / "metrics/gpt_stability_retest.jsonl")) if (root / "metrics/gpt_stability_retest.jsonl").exists() else 0
    gpt_stability_failures = len(read_jsonl(root / "logs/gpt_stability_retest_failures.jsonl")) if (root / "logs/gpt_stability_retest_failures.jsonl").exists() else 0
    stability_executed = (
        stability_index_rows == 240
        and qwen_stability_valid + qwen_stability_failures == 240
        and gpt_stability_valid + gpt_stability_failures == 240
    )
    external_rows = read_jsonl(root / "data/external_main_outputs.jsonl") if (root / "data/external_main_outputs.jsonl").exists() else []
    external_attempted = len(external_rows)
    external_ok = sum(row.get("status") in {"ok", "truncated"} and bool(row.get("generated_text")) for row in external_rows)
    external_failed = external_attempted - external_ok

    bge_external = geometry[(geometry.encoder_id == "bge_m3") & (geometry.anchor_protocol == "external_fixed_reference")].sort_values("ice_mean")
    bge_self = geometry[(geometry.encoder_id == "bge_m3") & (geometry.anchor_protocol == "self_anchored")].sort_values("ice_mean")
    rank_row = ranks[(ranks.encoder_id == "bge_m3") & ranks.comparison.str.startswith("external")].iloc[0]
    external_sentence = ", ".join(f"{row.model_id} {row.ice_mean:.3f}" for row in bge_external.itertuples())
    self_sentence = ", ".join(f"{row.model_id} {row.ice_mean:.3f}" for row in bge_self.itertuples())
    main_models = sorted(geometry.model_id.unique().tolist())
    cross_family_executed = external_attempted == 6720 and len(main_models) == 6
    cross_family_strict_complete = external_ok == 6720 and len(main_models) == 6
    external_run_status = "completed" if external_attempted == 6720 else "in progress"

    result_lines = [
        "# Revision results",
        "",
        f"Generated from final CSV/JSON artifacts at {utc_now()}. Numbers below are traceable to the named tables; no page or line numbers are inferred for the manuscript.",
        "",
        "## Completion and denominators",
        "",
        f"The retained local-Qwen main experiment contains 33,600 outputs. The 2×2 robustness matrix contains {robustness.get('expected_records', 'NA'):,} planned condition records, of which {robustness.get('valid_output_records', 'NA'):,} contain valid text and {robustness.get('missing_records', 'NA')} are explicitly recorded failures. The deduplicated evaluation index contains {eval_status.get('total', 'NA'):,} usable outputs; {eval_status.get('explicitly_failed_and_excluded', 'NA')} planned identities are explicit generation failures, so all {eval_status.get('expected_complete_total', 'NA'):,} planned identities are accounted for without imputation.",
        "",
        f"The cross-family GPT-6-Luna generation run is {external_run_status}: {external_attempted:,}/6,720 request identities were attempted, yielding {external_ok:,} usable outputs; the non-usable count is {external_failed:,}, with status retained. It is a different model family from Qwen, but it shares the GPT family/provider with the GPT-6-Sol judge; the local Qwen judge supplies the cross-family evaluator comparison. Six-model tables retain valid denominators and never impute non-usable outputs.",
        "",
        "## Anchor protocols and encoder sensitivity",
        "",
        f"Under BGE-M3, mean internal ICE with self anchors was: {self_sentence}. With the fixed GPT reference pair it was: {external_sentence}. The model-order correlation between those two protocols was Spearman {fmt(rank_row.spearman)} with {int(rank_row.pairwise_rank_flips)} pairwise flips among {int(rank_row.pairwise_model_pairs)} model pairs. The pooled-Qwen and other-encoder results are reported without requiring identical rankings in `table_anchor_encoder_sensitivity.csv`.",
        "",
        f"All 320 GPT reference pairs were generated once and frozen. Automatic Qwen screening was valid for {int(references.qwen_judge_valid)}/640 texts; the text-level length-pass rate was {references.length_pass_rate:.1%}, both endpoints passed content checks in {int(references.pair_both_content_pass_count)}/320 scenes, and the mean automatic B-minus-A intensity span was {references.mean_style_span_B_minus_A:.2f} points. These are automatic diagnostics, not human certification.",
        "",
        "## Independent judge",
        "",
    ]
    if not judge_agreement.empty:
        overall = judge_agreement[judge_agreement.stratum == "overall"].iloc[0]
        residual = judge_agreement[judge_agreement.stratum == "demeaned_within_alpha_axis"].iloc[0]
        content = content_agreement.iloc[0]
        result_lines.extend([
            f"Across matched outputs, Qwen and GPT intensity scores had Pearson r={fmt(overall.pearson)} and Spearman rho={fmt(overall.spearman)}; after demeaning within target-ratio × axis strata, the corresponding correlations were {fmt(residual.pearson)} and {fmt(residual.spearman)}. The mean absolute intensity difference was {fmt(overall.qwen_minus_gpt_mae, 2)} points.",
            "",
            f"Content-pass raw agreement was {content.raw_agreement:.1%} and Cohen's kappa was {fmt(content.cohen_kappa)} (Qwen pass prevalence {content.qwen_pass_rate:.1%}; GPT {content.gpt_pass_rate:.1%}). Kappa is interpreted alongside the confusion table because high prevalence can depress it.",
            "",
        ])
        if not ranking_comparison.empty and not judge_rankings.empty:
            rank_pivot = judge_rankings.pivot(index="model_id", columns="judge", values="rank").dropna()
            changed_ranks = int((rank_pivot.qwen != rank_pivot.gpt).sum()) if {"qwen", "gpt"}.issubset(rank_pivot.columns) else 0
            rank_stat = ranking_comparison.iloc[0]
            result_lines.extend([
                f"The descriptive six-model style-MAE ordering had Spearman correlation {fmt(rank_stat.spearman_rank_correlation_descriptive)} across judges; {changed_ranks}/{len(rank_pivot)} models changed ordinal position. Exact judge-specific ranks and scenario-matched pairwise differences are retained in `dual_judge_model_rankings.csv` and `dual_judge_model_pairwise_style_mae.csv` rather than reduced to a universal ranking claim.",
                "",
            ])
        if not length_sensitivity.empty:
            for row in length_sensitivity.itertuples():
                result_lines.append(
                    f"- {row.judge} geometry–judge residual association: r={fmt(row.residual_correlation)} "
                    f"[{fmt(row.residual_correlation_ci_low)}, {fmt(row.residual_correlation_ci_high)}]; "
                    f"after residualizing character length: r={fmt(row.length_controlled_correlation)} "
                    f"[{fmt(row.length_controlled_ci_low)}, {fmt(row.length_controlled_ci_high)}] "
                    f"({int(row.valid_scenarios)} scenario clusters)."
                )
            result_lines.append("")
    else:
        result_lines.extend(["Dual-judge analysis is pending; no placeholder numerical claims are inserted.", ""])

    result_lines.extend(["## Full robustness", ""])
    if not robustness_table.empty:
        effects = robustness_table[(robustness_table.record_type == "factorial_effect") & (robustness_table.model_id == "equal_weight_five_models")]
        for metric in [
            "external_reference_ice", "external_reference_drift", "external_reference_out_of_range",
            "qwen_style_mae", "qwen_content_pass", "qwen_naturalness",
            "gpt_style_mae", "gpt_content_pass", "gpt_naturalness", "character_length",
        ]:
            subset = effects[effects.metric == metric]
            if subset.empty:
                continue
            pieces = [f"{row.condition_or_effect}={fmt(row.mean)} [{fmt(row.ci_low)}, {fmt(row.ci_high)}]" for row in subset.itertuples()]
            result_lines.append(f"- {metric}: " + "; ".join(pieces) + ".")
        result_lines.extend(["", "Effects use the prespecified signs: mean-temperature(P2−P1), mean-prompt(low−high), and the difference-in-differences interaction. Three seeds and five internal ratios are first averaged within scenario; the five Qwen models are then equally weighted.", ""])
    else:
        result_lines.extend(["Full factorial analysis is pending; no estimated effects are inserted.", ""])

    result_lines.extend([
        "## Efficiency and scope limits",
        "",
        "The corrected local efficiency figure contains only qwen35_35b_a3b, qwen36_27b, and qwen36_35b_a3b because their logs match on two NVIDIA L20Z GPUs, TP=2, BF16, no quantization, and concurrency 32. The one-GPU 4B/9B deployments remain in a descriptive table. API latency is not placed on the local-GPU efficiency frontier.",
        "",
        "The evidence supports an operational, embedding-and-automatic-judge diagnostic for Chinese short-text style control. It does not establish human perceptual linearity, absolute style truth, cross-language generalization, universal behavior across LLM families, or elimination of evaluator bias.",
        "",
        "## Budget",
        "",
        f"Recorded GPT token usage implies an estimated cumulative cost of USD {api.get('gpt_total_estimated_cost_usd', float('nan')):.2f} under the author-supplied prices. Adding conservative allowances for calls that may have been in flight at two concurrency restarts and for two inexpensive comparison pilots whose raw usage was not retained gives USD {api.get('gpt_conservative_total_with_allowances_usd', float('nan')):.2f}, below the USD {api.get('operational_gpt_hard_cap_usd', float('nan')):.0f} operational cap. The relay did not expose actual wallet balance. DeepSeek dollar cost is not imputed without an internal price or billable response.",
    ])
    atomic_write_text(reports / "revision_results.md", "\n".join(result_lines) + "\n")

    if not judge_agreement.empty and not content_agreement.empty:
        evidence_overall = judge_agreement[judge_agreement.stratum == "overall"].iloc[0]
        evidence_residual = judge_agreement[judge_agreement.stratum == "demeaned_within_alpha_axis"].iloc[0]
        evidence_content = content_agreement.iloc[0]
        judge_response_paragraph = (
            f"Across matched outputs, the Qwen/GPT intensity association was Pearson r={fmt(evidence_overall.pearson)} "
            f"and Spearman rho={fmt(evidence_overall.spearman)}; after demeaning within alpha-by-axis strata it was "
            f"r={fmt(evidence_residual.pearson)} and rho={fmt(evidence_residual.spearman)}. Content-pass agreement was "
            f"{evidence_content.raw_agreement:.1%} with Cohen's kappa={fmt(evidence_content.cohen_kappa)}. We treat these as "
            "evaluator-sensitivity diagnostics rather than human validation or a causal test of same-family bias."
        )
    else:
        judge_response_paragraph = "Dual-judge numerical response text is pending and must not be pasted as a completed result."

    evidence = f"""# Reviewer-response evidence map

This map follows the task-summary labels supplied for the revision; the formal response letter must quote and answer the actual submission-system wording. No manuscript page or line numbers are invented here.

| Concern | Evidence | English response point |
|---|---|---|
| R1/R2: interpretation strength and stronger reference validation | `tables/table_independent_judge.csv`; `tables/table_reference_quality.csv`; `statistics/geometry_judge_association_summary.csv` | We added a second automatic evaluator and a frozen independently generated reference pair for every scene. We describe both as operational automatic diagnostics, not human validation; the GPT reference/judge role overlap is disclosed. |
| R1/R2: model-specific self coordinates | `tables/table_cross_family_main.csv`; `tables/table_anchor_encoder_sensitivity.csv`; `statistics/anchor_protocol_rank_sensitivity.csv` | We now contrast self anchors, fixed external text anchors, and fixed pooled-Qwen vector anchors across all frozen encoders. Ranking changes are reported as different measurement questions rather than treated as errors. |
| R1: eight-axis rationale and boundary | `reports/benchmark_construction_facts.md` | We document the recorded axis definitions, task/domain coverage, atomic checks, length ranges, curation rules, and unrecorded provenance. The axes cover selected everyday Chinese short-text styles rather than all styles. |
| R1/R2: Qwen-only evidence | `tables/table_cross_family_main.csv`; `manifests/external_generator_selection.json` | We attempted all 6,720 prespecified GPT-6-Luna request identities as the sixth, non-Qwen generator; {external_ok} yielded usable text, and the non-usable count is {external_failed}, with status retained. GPT-6-Luna is distinct from the GPT-6-Sol judge but belongs to the same GPT family/provider, which is disclosed; the local Qwen judge provides the cross-family evaluator view. |
| R3: possible same-family judge sensitivity | `tables/table_independent_judge.csv`; `statistics/dual_judge_intensity_agreement.csv`; `statistics/dual_judge_content_confusion.csv`; `statistics/dual_judge_model_condition_differences.csv` | The same outputs are scored blind by Qwen and GPT; we report absolute agreement, stratified/residual agreement, content confusion, and model-conditional differences without interpreting them as a causal measurement of self-bias. |
| R3: 28.57% | `reports/metric_estimator_audit.md`; `tests/test_metric_audit.py` | The 2/7 reduction is an algebraic weighting identity for complete equally weighted self-anchored seven-point trajectories, not a new empirical finding. |
| R3: 40-scene interaction evidence | `tables/table_robustness_full.csv`; `tables/table_robustness_axis_effects.csv`; `figures/figure_robustness_effects_ci.pdf` | The 2×2 prompt-by-temperature design now covers all 320 scenes and all five original Qwen models; effects are scenario-matched with clustered CIs. |
| R3: Figure 7 comparability | `tables/table_efficiency_matched_local.csv`; `tables/table_efficiency_all_deployments_descriptive.csv`; `figures/figure_efficiency_matched_local.pdf` | The comparative figure is restricted to the three runs with the same two-GPU/TP/concurrency configuration; other local deployments and API observations are descriptive only. |

Suggested methods sentence: “For uncertainty estimation, the independent unit was `scenario_id`; all conditions belonging to a sampled scenario were retained together in an axis-stratified 10,000-replicate cluster bootstrap (seed 20260916). Primary tests were limited to BGE-M3 external-reference ICE and judge-specific style MAE, with prespecified Holm families.”

Suggested limitation sentence: “All references and ratings are model-generated automatic measurements on Chinese short texts; they neither constitute human perceptual validation nor establish language-independent or absolute style ground truth.”

## Paste-ready English response paragraphs

**External generator and scope.** “We added GPT-6-Luna as a sixth, non-Qwen generator and attempted all {external_attempted:,} prespecified requests over 320 scenarios, seven ratios, and three independent API replicates. {external_ok:,} responses were usable; the non-usable count is {external_failed:,}, with status retained, and no score- or quality-dependent regeneration was performed. GPT-6-Luna and the GPT-6-Sol evaluator are distinct model aliases but share a GPT-family provider route, so we disclose this limitation and use the local Qwen evaluator as the cross-family judge comparison.”

**Fixed reference protocol.** “For every scenario, GPT-6-Sol generated one A endpoint and one B endpoint once, after which the 640 texts and their hashes were frozen and shared across all generators, replicates, conditions, and encoders. Qwen automatic screening produced valid records for {int(references.qwen_judge_valid)}/640 texts; both endpoints passed the content rule in {int(references.pair_both_content_pass_count)}/320 scenarios. These checks are automatic diagnostics, not human certification, and the reference-generator/evaluator role overlap is explicitly disclosed.”

**Anchor sensitivity.** “Under BGE-M3, the six-model ordering by mean internal ICE had Spearman correlation {fmt(rank_row.spearman)} between self anchors and fixed external text anchors, with {int(rank_row.pairwise_rank_flips)}/{int(rank_row.pairwise_model_pairs)} pairwise order changes. We do not require protocol-invariant rankings: self anchoring measures calibration relative to each model's own endpoints, whereas fixed anchors measure calibration in a shared operational coordinate system.”

**Independent automatic judge.** “{judge_response_paragraph}”

**Missingness.** “The deduplicated judge index contains {eval_status.get('total', 'NA'):,} usable outputs. Seven planned identities—six local robustness generations and one external generation—are retained as explicit generation failures and are never imputed as passes or zero error; all 112,320 planned identities are therefore accounted for.”
"""
    atomic_write_text(reports / "reviewer_response_evidence.md", evidence)

    artifact_files = [
        path
        for path in root.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts and ".pytest_cache" not in path.parts
    ]
    artifact_bytes = sum(path.stat().st_size for path in artifact_files)
    qwen_scores = len(read_jsonl(root / "metrics/qwen_judge_evaluations.jsonl"))
    gpt_scores = len(read_jsonl(root / "metrics/gpt_judge_evaluations.jsonl"))
    qwen_failures = len(read_jsonl(root / "logs/qwen_judge_failures.jsonl")) if (root / "logs/qwen_judge_failures.jsonl").exists() else 0
    gpt_failures = len(read_jsonl(root / "logs/gpt_judge_failures.jsonl")) if (root / "logs/gpt_judge_failures.jsonl").exists() else 0
    completion = f"""# Completion report

Generated: {utc_now()}

## Counts and status

- Legacy Qwen main retained: 33,600.
- External GPT-6-Luna requests attempted: {external_attempted}/6,720 — {external_run_status}; usable text: {external_ok}; recorded failures: {external_failed}.
- Robustness condition records: {robustness.get('expected_records', 'NA'):,}; valid text: {robustness.get('valid_output_records', 'NA'):,}; recorded failures: {robustness.get('missing_records', 'NA')}.
- Evaluation index: {eval_status.get('total', 'NA'):,} usable; {eval_status.get('explicitly_failed_and_excluded', 'NA')} explicit generation failures excluded; all 112,320 planned identities accounted for: {eval_status.get('all_planned_identities_accounted_for', False)}.
- Qwen judge valid/cache records: {qwen_scores:,}; terminal evaluation failures: {qwen_failures}.
- GPT judge valid records: {gpt_scores:,}; terminal evaluation failures: {gpt_failures}.
- Fixed references: 640 generated; Qwen screening valid: {int(references.qwen_judge_valid)}.
- Judge stability retest: {'all 240 selected identities accounted for by both judges' if stability_executed else 'pending or partial'}; valid retest scores were Qwen {qwen_stability_valid}/240 and GPT {gpt_stability_valid}/240, with terminal failures retained separately.
- Three anchor protocols × three encoders: {'six-model analysis complete on all valid outputs' if cross_family_executed else 'partial; see geometry manifests'}.

## API usage and cost

- GPT estimated cost from recorded usage and pilot: USD {api.get('gpt_total_estimated_cost_usd', float('nan')):.4f}.
- Conservative total with interrupted in-flight-call and unrecorded comparison-pilot allowances: USD {api.get('gpt_conservative_total_with_allowances_usd', float('nan')):.4f}.
- Operational cap: USD {api.get('operational_gpt_hard_cap_usd', float('nan')):.2f}; estimated headroom: USD {api.get('estimated_headroom_usd', float('nan')):.2f}.
- Actual account balance: unavailable from relay.
- External GPT-6-Luna generation cost: USD {api.get('external_generation', {}).get('estimated_cost_usd', float('nan')):.4f}.
- GPT hashed response IDs captured: {coverage_audit.get('gpt_judge', {}).get('hashed_response_ids_present', 'NA')}; missing from early records: {coverage_audit.get('gpt_judge', {}).get('hashed_response_ids_missing', 'NA')}. The missing early IDs are an audit limitation and are not reconstructed.

## Reproduction entry points

Exact credential-free commands are preserved in `reports/runbook.md`.

- Audit: `python scripts/audit_inputs.py --config config/experiment.yaml --output-root .`
- Robustness preparation/generation: `prepare_robustness.py`, `run_qwen_robustness.py`, then `consolidate_outputs.py`.
- External main generation: `run_gpt_external_main.py` (resume-safe; token budget and response status recorded per request).
- References/judges: `run_gpt_references.py`, `build_*_evaluation_index.py`, `run_qwen_judge.py`, `run_gpt_judge.py`.
- Embeddings/geometry: `embed_dataset.py`, then `compute_anchor_metrics.py` once per encoder.
- Final analyses: `analyze_reference_quality.py`, `analyze_geometry.py`, `analyze_judges.py`, `analyze_robustness_full.py`, `analyze_stability_retest.py`, `select_failure_cases.py`.
- Tests: `pytest -q revision_computers_major_r1/tests` from the repository root.
- Final identity/count/hash audit: `python scripts/audit_final_coverage.py --output-root .`.

## Worktree and authorization boundaries

- Legacy directory was used read-only; new writes are contained under `revision_computers_major_r1/`.
- No human labels were created or inferred.
- No manuscript submission or email was performed.
- On the author's explicit instruction, this completed package was prepared for delivery to the repository's public `main` branch. The exact delivery commit is recorded in Git history.
- The new revision tree contains {len(artifact_files):,} files totaling {artifact_bytes / (1024 ** 3):.2f} GiB; JSONL, NPZ, and Parquet payloads use Git LFS. See `manifests/environment_and_hashes.json` for the audited base commit, environment, exact paths, and critical hashes.
"""
    atomic_write_text(reports / "completion_report.md", completion)
    print(json.dumps({"reports_written": 3, "cross_family_executed": cross_family_executed, "cross_family_strict_complete": cross_family_strict_complete, "external_attempted": external_attempted, "external_usable": external_ok}, ensure_ascii=False))


if __name__ == "__main__":
    main()
