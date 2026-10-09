from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
sys.path.insert(0, str(ROOT / "scripts"))

from common import read_jsonl  # noqa: E402
from prompts import render_external_prompt, render_qwen_prompt  # noqa: E402
from run_gpt_external_main import build_jobs as build_external_jobs  # noqa: E402
from run_qwen_robustness import build_jobs as build_robustness_jobs  # noqa: E402


def load_config() -> dict:
    return yaml.safe_load((ROOT / "config/experiment.yaml").read_text(encoding="utf-8"))


def test_qwen_prompt_is_byte_identical_to_all_legacy_rows() -> None:
    config = load_config()
    scenarios = {row["scenario_id"]: row for row in read_jsonl(config["scenario_file"])}
    paths = [
        Path(config["legacy_root"]) / "data/qwen_family_outputs.jsonl",
        Path(config["legacy_root"]) / "data/robustness_outputs.jsonl",
    ]
    for path in paths:
        for row in read_jsonl(path):
            scenario = scenarios[row["scenario_id"]]
            assert render_qwen_prompt(scenario, float(row["target_ratio"]), row["prompt_template_id"]) == row["prompt"]


def test_external_wrapper_removes_only_no_think_control_line() -> None:
    config = load_config()
    scenario = read_jsonl(config["scenario_file"])[0]
    original, sent = render_external_prompt(scenario, 0.5, "computers_explicit_ratio_v1")
    assert original == "/no_think\n" + sent
    assert not sent.startswith("/no_think")


def test_locked_matrix_sizes() -> None:
    config = load_config()
    scenarios = read_jsonl(config["scenario_file"])
    external = build_external_jobs(config, scenarios)
    assert len(external) == 6720
    assert len({row["run_id"] for row in external}) == 6720
    assert len({(row["scenario_id"], row["target_ratio"], row["replicate_id"]) for row in external}) == 6720
    model = {"model_id": "qwen35_4b", "family_version": "Qwen3.5", "architecture": "Dense"}
    assert len(build_robustness_jobs(config, scenarios, model)) == 19200
